from firedrake import *
import utils
import numpy as np


class AxisAlignedDirichletBC(DirichletBC):
    axis_aligned = True


class GeodesicShooter:

    def __init__(self, _mesh, log_path):
        selfsmesh = _mesh
        self.log_path = log_path
        utils.create_dir_from_path_if_not_exists(self.log_path)

        # set up mesh
        self._scale = 5  # to fit unit circle to gmsh field
        self._shape_tag = 10
        self._inside_tag = 16
        solver_parameters = {'mat_type': 'aij', 'ksp_type': 'preonly', 'pc_factor_mat_solver_type': 'mumps', 'pc_type': 'lu'}

        # Function spaces
        #self.XW = VectorFunctionSpace(self.mesh, "WXRobH3NC", degree=7, dim=2)
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
        a = utils.trihelmholtz(v, dv, alpha=1)
        # compute number of edges that form the initial curve q_{0h}
        n = assemble((1./self.h('+'))*dS(self._shape_tag))  # reference interval has measure 1
        rhs = (self.h*Constant(2*pi/n)*inner(det(grad(self.phi)) * self.p0, dv))('+')*dS(self._shape_tag)
        bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")
        velocity_problem_u = LinearVariationalProblem(a, rhs, self.u, bcs=bcs, constant_jacobian=False)
        self.velocity_solver_u = LinearVariationalSolver(velocity_problem_u, solver_parameters=solver_parameters)

    def shoot(self, p0, timesteps):
        """ Solves the forward problem given momentum `p`, tracking the
            locations of the landmarks given by the parameterisation `ts`.
            `dump_curves` is only for IVP examples.
        """
        dt = Constant(1/timesteps)
        shape_normal = utils.shape_normal(self.mesh, self.VDGT)
        self.p0.assign(utils.trace_interpolate(self.VDGT, p0 * shape_normal))

        for t in range(timesteps):
            utils.pprint("Shooting... t = {}".format(t + 1))
            self.facetarea_solver.solve()
            self.velocity_solver_u.solve()
            self.phi += self.u * dt
            self._update_mesh()

        File(self.log_path / "shape.pvd").write(self.shape_function)

    def _update_mesh(self):
        coords = project(self.phi, self.VCG)
        self.mesh.coordinates.assign(coords)
        self.mesh.clear_spatial_index()

    def _check_points(self, points):
        """ Sanity check whether the points given by `points` lies within the
        spatial domain. """
        eps = 12.5 + 1e-05
        within_bounds = (points > -eps).all() and (points < eps).all()
        if not within_bounds:
            raise Exception("Template points moved outside domain: {}"
                .format(points))

    def extract_points(self, thetas):
        """ Get the points on the discretised shape (i.e. the "circle" for
        now) corresponding to the parameterisation `thetas`, here
        (cos(t), sin(t)).
        """
        return self._scale * np.array([(np.cos(t), np.sin(t)) for t in thetas])

