from firedrake import *
import numpy as np

import src.utils as utils


class AxisAlignedDirichletBC(DirichletBC):
    axis_aligned = True


def velocity_lhs(v, dv, phi):
    alpha = 0.5
    alpha_1 = alpha
    alpha_2 = alpha**2
    alpha_3 = alpha**3

    J = grad(phi)
    JT = transpose(J)
    J_inv = inv(J)

    def gradJ(w):
        return dot(grad(w), J_inv)

    def deltaJ(w):
        grad_w = grad(w)
        H = grad(grad_w)
        return dot(JT, dot(H, J)) + dot(grad_w, grad(J))

    def triJ(w):
        grad_w = grad(w)
        H = grad(grad_w)
        T = grad(H)
        return 2 * dot(grad(JT), dot(H, J)) + dot(JT, dot(T, J)) + dot(H, grad(J)) + dot(grad_w, grad(grad(J)))

    vx, vy = v
    dvx, dvy = dv

    a = inner(v, dv)
    a += alpha_1*(inner(gradJ(vx), gradJ(dvx)) + inner(gradJ(vy), gradJ(dvy)))
    a += alpha_2*(inner(deltaJ(vx), deltaJ(dvx)) + inner(deltaJ(vy), deltaJ(dvy)))
    a += alpha_3*(inner(triJ(vx), triJ(dvx)) + inner(triJ(vy), triJ(dvy)))

    return a * det(J) * dx


class GeodesicShooter:

    def __init__(self, _mesh: Mesh, logger: utils.Logger):
        self.mesh = _mesh
        self._logger = logger

        # set up mesh
        self._scale = 5  # to fit unit circle to gmsh field
        self._shape_tag = 10
        self._domain_size = 12.5  # TODO: fix this
        self._inside_tag = 16
        solver_parameters = {'mat_type': 'aij', 'ksp_type': 'preonly', 'pc_factor_mat_solver_type': 'mumps', 'pc_type': 'lu'}

        # Function spaces
        self.XW = VectorFunctionSpace(self.mesh, "WXH3NC", degree=4, dim=2)
        self.DG = FunctionSpace(self.mesh, "DG", 0)  # for plotting inside the curve
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=1, dim=2)  # for momentum
        self.VCG = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.DGT0 = FunctionSpace(self.mesh, "DGT", 0)  # for trace sizes

        # shape function for plotting
        self.shape_function = utils.shape_function(self.DG, self._inside_tag)

        # set up the Functions we need
        self.orig_coords = project(SpatialCoordinate(self.mesh), self.VCG)
        self.phi = project(SpatialCoordinate(self.mesh), self.XW)

        # set up FacetArea recomputation problem
        self.h = Function(self.DGT0)
        h_trial, h_test = TrialFunction(self.DGT0), TestFunction(self.DGT0)
        h_lhs = inner(h_trial, h_test)('+')*dS + inner(h_trial, h_test) * ds
        h_rhs = inner(FacetArea(self.mesh), h_test)('+') * dS + inner(FacetArea(self.mesh), h_test) * ds
        facetarea_problem = LinearVariationalProblem(h_lhs, h_rhs, self.h, constant_jacobian=False)
        self.facetarea_solver = LinearVariationalSolver(facetarea_problem, solver_parameters=solver_parameters)
        self.facetarea_solver.solve()  # warm up

        # set up velocity problem
        self.u = Function(self.XW)
        self.p0 = Function(self.VDGT)
        v, dv = TrialFunction(self.XW), TestFunction(self.XW)
        a = velocity_lhs(v, dv, self.phi)
        # compute number of edges that form the initial curve q_{0h}
        n = assemble((1./self.h('+'))*dS(self._shape_tag))  # reference interval has measure 1
        rhs = (self.h*Constant(2*pi/n)*inner(det(grad(self.phi)) * self.p0, dv))('+')*dS(self._shape_tag)
        bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")
        velocity_problem_u = LinearVariationalProblem(a, rhs, self.u, bcs=bcs, constant_jacobian=False)
        self.velocity_solver_u = LinearVariationalSolver(velocity_problem_u, solver_parameters=solver_parameters)

    def shoot(self, p0, time_steps):
        dt = Constant(1 / time_steps)
        shape_normal = utils.shape_normal(self.mesh, self.VDGT)
        self.p0.assign(utils.trace_interpolate(self.VDGT, p0 * shape_normal))

        for t in range(time_steps):
            utils.pprint(f"Shooting... t = {t}")
            self.velocity_solver_u.solve()
            self.phi += self.u * dt

        # move the mesh for visualisation
        self._update_mesh()
        return self.phi, self.shape_function

    def _update_mesh(self):
        self.mesh.coordinates.assign(project(self.phi, self.VCG))
        self.mesh.clear_spatial_index()

    def _check_points(self, points):
        """ Sanity check whether the points given by `points` lies within the
        spatial domain. """
        eps = self._domain_size + 1e-05
        within_bounds = (points > -eps).all() and (points < eps).all()
        if not within_bounds:
            raise Exception("Template points moved outside domain: {}".format(points))

    def _extract_points(self, thetas):
        """ Get the points on the discretised shape (i.e. the "circle" for
        now) corresponding to the parameterisation `thetas`, here
        (cos(t), sin(t)).
        """
        return self._scale * np.array([(np.cos(t), np.sin(t)) for t in thetas])
