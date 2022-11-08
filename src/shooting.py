from dataclasses import dataclass, field
from typing import Type, List, Any

import firedrake
from firedrake import *
from pathlib import Path

import src.utils as utils
from src.curves import Curve
from src.manufactured_solutions import Momentum
from src.mesh_generation.mesh_generation import CURVE_TAG, INNER_TAG

__all__ = ["Diffeomorphism", "ShootingParameters", "GeodesicShooter"]


Diffeomorphism = Type[Function]


@dataclass
class ShootingParameters:
    velocity_solver_parameters: dict = field(
        default_factory=lambda:
        {
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_factor_mat_solver_type': 'mumps',
            'pc_type': 'lu'
        }
    )
    momentum_degree: int = 0
    alpha: float = .1
    time_steps: int = 10


@dataclass
class CurveResult:
    diffeo: firedrake.Function
    velocity_norms: List[Any] = None
    momentum_norms: List[Any] = None


class GeodesicShooter:
    def __init__(
        self,
        logger: utils.Logger,
        mesh_path: Path,
        template: Curve = None,
        shooting_parameters: ShootingParameters = None,
        communicator=COMM_WORLD,
    ):
        self.mesh_path = mesh_path
        self.communicator = communicator
        self.mesh = Mesh(str(self.mesh_path), comm=self.communicator)
        self._logger = logger
        self.parameters = shooting_parameters or ShootingParameters()
        self._solver_parameters = self.parameters.velocity_solver_parameters
        self.template = template

        # Function spaces
        self.XW = VectorFunctionSpace(self.mesh, "WXRobH3NC", degree=7, dim=2)
        self.XW_scalar = FunctionSpace(self.mesh, "WXRobH3NC", degree=7)
        self.XW_tensor = TensorFunctionSpace(self.mesh, "WXRobH3NC", degree=7, shape=(2, 2))
        self.DG = FunctionSpace(self.mesh, "DG", 0)
        VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=0, dim=2)  # for shape normal
        self.XW_approx = VectorFunctionSpace(self.mesh, "DG", degree=7, dim=2)
        self.VCG1 = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.MomentumSpace = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal
        self.velocity_bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")

        # Velocity, momentum and diffeo
        self.diffeo = project(SpatialCoordinate(self.mesh), self.XW)
        self.u, self.w, self.z = Function(self.XW), Function(self.XW), Function(self.XW)
        self.momentum = Function(self.MomentumSpace)  # dummy

        # Functions we'll need for the source term/visualisation
        self.shape_function = utils.shape_function(self.mesh, INNER_TAG)
        self.h_inv = inv(utils.compute_facet_area(self.mesh))
        self.n = assemble(self.h_inv('+') * dS(CURVE_TAG))  # reference interval has measure 1
        self.shape_normal = utils.shape_normal(self.mesh, VDGT)
        self.orig_coords = project(SpatialCoordinate(self.mesh), self.XW)

        v, dv = TrialFunction(self.XW), TestFunction(self.XW)
        self.diffeo = Function(self.XW).assign(self.orig_coords)
        self.jacobian = Function(self.XW_tensor)
        self.inv_jacobian = Function(self.XW_tensor)
        self.transp_inv_jacobian = Function(self.XW_tensor)
        self.det_jacobian = Function(self.XW_scalar)

        vx, vy = v
        dvx, dvy = dv

        inner_j = lambda w, z: inner(dot(grad(w), self.inv_jacobian), dot(grad(z), self.inv_jacobian))
        a_form = (inner(v, dv) + self.parameters.alpha * inner_j(vx, dvx) + inner_j(vy, dvy)) * self.det_jacobian * dx

        dt, dvt = TrialFunction(self.XW_tensor), TestFunction(self.XW_tensor)
        h1_form = inner(dt, dvt) * dx
        lvp_jacobian = LinearVariationalProblem(
            a=h1_form, L=inner(grad(self.diffeo), dvt) * dx, u=self.jacobian, constant_jacobian=False
        )
        lvp_inv_jacobian = LinearVariationalProblem(
            a=h1_form, L=inner(inv(self.jacobian), dvt) * dx, u=self.inv_jacobian, constant_jacobian=False,
        )

        dt, dvt = TrialFunction(self.XW_scalar), TestFunction(self.XW_scalar)
        lvp_det_jacobian = LinearVariationalProblem(
            a=inner(dt, dvt) * dx, L=inner(det(self.jacobian), dvt) * dx, u=self.det_jacobian, constant_jacobian=False,
        )
        lvp_z = LinearVariationalProblem(
            a=a_form,
            L=Constant(2*pi) * self.h_inv('+') * dot(dot(transpose(self.inv_jacobian), self.momentum * self.shape_normal), dv)('+') * dS(CURVE_TAG),
            u=self.z, bcs=self.velocity_bcs, constant_jacobian=False,
        )
        lvp_w = LinearVariationalProblem(
            a=a_form, L=inner(self.z, dv) * self.det_jacobian * dx, u=self.w, bcs=self.velocity_bcs, constant_jacobian=False,
        )
        lvp_u = LinearVariationalProblem(
            a=a_form, L=inner(self.w, dv) * self.det_jacobian * dx, u=self.u, bcs=self.velocity_bcs, constant_jacobian=False,
        )

        self.lvs_z = LinearVariationalSolver(lvp_z, solver_parameters=self._solver_parameters)
        self.lvs_w = LinearVariationalSolver(lvp_w, solver_parameters=self._solver_parameters)
        self.lvs_u = LinearVariationalSolver(lvp_u, solver_parameters=self._solver_parameters)

        self.lvs_jacobian = LinearVariationalSolver(lvp_jacobian, solver_parameters=self._solver_parameters)
        self.lvs_inv_jacobian = LinearVariationalSolver(lvp_inv_jacobian, solver_parameters=self._solver_parameters)
        self.lvs_det_jacobian = LinearVariationalSolver(lvp_det_jacobian, solver_parameters=self._solver_parameters)

    def shoot(self, momentum: Momentum) -> CurveResult:
        self.diffeo.assign(self.orig_coords)
        Dt = Constant(1 / self.parameters.time_steps)

        if not isinstance(momentum, Function):
            x, y = SpatialCoordinate(self.mesh)
            momentum = Function(self.MomentumSpace).interpolate(momentum.signal(x, y))

        self.momentum.assign(momentum)

        for t in range(self.parameters.time_steps):
            self.lvs_jacobian.solve()
            self.lvs_inv_jacobian.solve()
            self.transp_inv_jacobian.assign(transpose(self.inv_jacobian))
            self.lvs_det_jacobian.solve()
            self.lvs_z.solve()
            self.lvs_w.solve()
            self.lvs_u.solve()
            self.diffeo.assign(self.diffeo + self.u * Dt)

        # move the mesh for visualisation
        soft_diffeo = project(self.diffeo, self.XW_approx)
        return CurveResult(
            diffeo=soft_diffeo,
        )

    def momentum_function(self):
        """ Used in the inverse problem solver. """
        return Function(self.MomentumSpace)

    def dump_parameters(self):
        self._logger.info(f"{self.parameters}")

    def update_mesh(self):
        soft_diffeo = project(self.diffeo, self.VCG1)
        self.mesh.coordinates.assign(soft_diffeo)
        self.mesh.clear_spatial_index()


class AxisAlignedDirichletBC(DirichletBC):
    axis_aligned = True
