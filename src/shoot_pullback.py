import utils
import numpy as np


def delta(u):
    return div(grad(u))


def bihelmholtz(v, dv):
    return (inner(v, dv)
          + inner(grad(v), grad(dv))
          + inner(delta(v), delta(dv))) * dx


def velocity_lhs(v, dv, phi):
    J = grad(phi)
    J_inv = inv(J)
    def gradJ(w):
        return dot(grad(w), J_inv)
    alp1 = 0.5
    vx, vy = v
    dvx, dvy = dv
    a = inner(v, dv) + alp1*inner(gradJ(vx), gradJ(dvx)) + alp1*inner(gradJ(vy), gradJ(dvy))
    a += gradJ(vx)
    return a * det(J) * dx


class GeodesicShooter:

    def __init__(self, mesh, timesteps):
        # time stuff
        self.timesteps = timesteps
        self.dt = Constant(1/timesteps)

        # set up mesh
        self.scale = 5  # to fit unit circle to gmsh field

        # Function spaces
        self.mesh = mesh
        self.VCG1 = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)
        self.VU = VectorFunctionSpace(self.mesh, "WXRobH3NC", degree=7, dim=2)
        self.U = FunctionSpace(self.mesh, "CG", 1)
        self.DG = FunctionSpace(self.mesh, "DG", 0)
        self.DGT = FunctionSpace(self.mesh, "DGT", 1)
        self.DGT0 = FunctionSpace(self.mesh, "DGT", 0)
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=1, dim=2)
        self.shape_tag = 10

        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             'pc_factor_mat_solver_type': 'mumps',
                             'pc_type': 'lu'}

        # set up the Functions we need
        self.orig_coords = project(SpatialCoordinate(self.mesh), self.VU)
        _bcs = DirichletBC(self.VU, 0, "on_boundary")
        self.u = Function(self.VU)
        self.q = Function(self.VU)
        self.p = Function(self.VDGT)
        self.h = Function(self.DGT0)

        # set up FacetArea recomputation problem
        h_trial, h_test = TrialFunction(self.DGT0), TestFunction(self.DGT0)
        h_lhs = inner(h_trial, h_test)('+')*dS + inner(h_trial, h_test) * ds
        h_rhs = inner(FacetArea(self.mesh), h_test)('+') * dS \
              + inner(FacetArea(self.mesh), h_test) * ds
        facetarea_problem = LinearVariationalProblem(h_lhs, h_rhs, self.h,
            constant_jacobian=False)
        self.facetarea_solver = LinearVariationalSolver(facetarea_problem,
            solver_parameters=solver_parameters)
        self.facetarea_solver.solve()  # warm up

        # compute number of edges that form the initial curve q_{0h}
        self._N = assemble((1./self.h('+'))*dS(10))  # reference interval has measure 1

        # set up velocity problem for u & w
        v = TrialFunction(self.VU)
        dv = TestFunction(self.VU)
        a = velocity_lhs(v, dv, self.q)

        det_grad_q_inv = self.h * Constant(2 * pi / self._N)
        x, y = SpatialCoordinate(self.mesh)
        ux, uy = self.u
        magnitude = sqrt((x + ux * self.dt) ** 2 + (y + uy * self.dt) ** 2)
        integrand = det_grad_q_inv * magnitude * self.u
        self.L = inner(integrand, dv)('+') * dS(self.shape_tag)
        velocity_problem_u = LinearVariationalProblem(
            a, self.L, self.u, bcs=DirichletBC(self.VU, 0, "on_boundary"), constant_jacobian=False
        )
        self.velocity_solver_u = LinearVariationalSolver(velocity_problem_u,
            solver_parameters=solver_parameters)
        #self.velocity_solver_u.solve()  # warm up

    def shoot(self, p, ts):
        self.q.assign(self.orig_coords)
        template_points = self.extract_points(ts)
        self.p.assign(utils.trace_interpolate(self.VDGT, Constant(7) * as_vector((1, 1))))

        utils.pprint(f"Shooting... t = 0")
        self.update_velocity(rhs=self.p)
        self.update_diffeomorphism()
        self.update_momentum()

        for t in range(1, self.timesteps):
            utils.pprint(f"Shooting... t = {t}")

            self.update_velocity()
            self.update_diffeomorphism()
            self.update_momentum()

            #template_points += np.array(self.u.at(template_points, tolerance=1e-06))
            #self._check_points(template_points)
        coords = project(self.q, self.VCG1)
        self.update_mesh(coords)
        File("shape.pvd").write(coords)
        return self.q, template_points

    def update_diffeomorphism(self):
        self.q.assign(self.q + self.u * self.dt)

    def update_momentum(self):
        pass

    def update_velocity(self, rhs=None):
        """ Inverts one or more Helmholtz operators onto the momentum. """

        # since the mesh has been moved since computing the RHS vector we need
        # to recompute the DG0 Function containing the FacetArea on the edges.
        self.recompute_facetarea()
        if rhs:
            dv = TestFunction(self.VU)
            v = TrialFunction(self.VU)
            a = velocity_lhs(v, dv, self.q)
            solve(
                a == inner(rhs, dv)('+') * dS(10), self.u, bcs=DirichletBC(self.VU, 0, "on_boundary")
            )
        else:
            self.velocity_solver_u.solve()

    def recompute_facetarea(self):
        self.facetarea_solver.solve()

    def cg_interpolate(self, u):
        """ Interpolates a Vector WuXu function into a Vector CG function. """

        ux, uy = u
        gx = Function(self.U).interpolate(ux)
        gy = Function(self.U).interpolate(uy)
        return Function(self.VCG1).interpolate(as_vector((gx, gy)))

    def _check_points(self, points):
        """ Sanity check whether the points given by `points` lies within the
        spatial domain. """
        eps = 12.5 + 1e-05
        within_bounds = (points > -eps).all() and (points < eps).all()
        if not within_bounds:
            raise Exception("Template points moved outside domain: {}"
                .format(points))

    def update_mesh(self, coords=None):
        """ Updates the mesh according to some `coords` (if provided, else
        defaults to the physical coords) and does some book-keeping.
        """
        if coords is None:
            coords = self.orig_coords

        coords = project(coords, self.VCG1)
        self.mesh.coordinates.assign(coords)
        self.mesh.clear_spatial_index()

    def extract_points(self, thetas):
        """ Get the points on the discretised shape (i.e. the "circle" for
        now) corresponding to the parameterisation `thetas`, here
        (cos(t), sin(t)).
        """
        return self.scale * np.array([(np.cos(t), np.sin(t)) for t in thetas])
