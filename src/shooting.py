from dataclasses import dataclass, field
from typing import Type, List, Any

import firedrake
from firedrake import *
from pathlib import Path

import src.utils as utils
from src.curves import Curve
from src.manufactured_solutions import Momentum
from src.mesh_generation.mesh_generation import CURVE_TAG


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
    alpha: float = 1
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
        self.mesh = Mesh(str(self.mesh_path), comm=communicator)
        self._logger = logger
        self.parameters = shooting_parameters or ShootingParameters()
        self._solver_parameters = self.parameters.velocity_solver_parameters
        self.template = template

        # Function spaces
        self.XW = VectorFunctionSpace(self.mesh, "WXRobH3NC", degree=7, dim=2)
        self.DG = FunctionSpace(self.mesh, "DG", 0)
        VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=0, dim=2)  # for shape normal
        self.XW_approx = VectorFunctionSpace(self.mesh, "DG", degree=7, dim=2)
        self.VCG1 = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.MomentumSpace = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal
        self.velocity_bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")

        # Velocity, momentum and diffeo
        self.diffeo = None
        self.u = Function(self.XW)
        self.momentum = None

        # Functions we'll need for the source term/visualisation
        self.shape_function = utils.shape_function(self.mesh, CURVE_TAG)
        self.h_inv = inv(utils.compute_facet_area(self.mesh))
        self.n = assemble(self.h_inv('+') * dS(CURVE_TAG))  # reference interval has measure 1
        self.shape_normal = utils.shape_normal(self.mesh, VDGT)

    def shoot(self, momentum: Momentum) -> CurveResult:
        self.diffeo = project(SpatialCoordinate(self.mesh), self.XW)

        dt = Constant(1 / self.parameters.time_steps)
        if isinstance(momentum, Function):
            self.momentum = momentum
        else:
            # mostly for manufactured solutions
            x, y = SpatialCoordinate(self.mesh)
            self.momentum = momentum.signal(x, y)

        for t in range(self.parameters.time_steps):
            self.velocity_solve()
            self.diffeo.assign(self.diffeo + self.u * dt)

        # move the mesh for visualisation
        soft_diffeo = project(self.diffeo, self.XW_approx)
        return CurveResult(
            diffeo=soft_diffeo,
        )

    def velocity_solve(self):
        v, dv = TrialFunction(self.XW), TestFunction(self.XW)
        w, z = Function(self.XW), Function(self.XW)

        momentum = dot(transpose(inv(grad(self.diffeo))), self.momentum * self.shape_normal)
        momentum_form = Constant(2*pi) * self.h_inv * momentum
        rhs = dot(momentum_form, dv)('+') * dS(CURVE_TAG)

        J = grad(self.diffeo)
        det_jacobian = det(J)
        inv_jacobian = inv(J)

        a, l = h1_form(v, dv, inv_jacobian, det_jacobian, alpha=self.parameters.alpha)
        solve(a == rhs, z, bcs=self.velocity_bcs, solver_parameters=self._solver_parameters)
        solve(a == l(z), w, bcs=self.velocity_bcs, solver_parameters=self._solver_parameters)
        solve(a == l(w), self.u, bcs=self.velocity_bcs, solver_parameters=self._solver_parameters)

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


def h1_form(v, dv, inv_jacobian, det_jacobian, alpha):
    grad_j = lambda w: dot(grad(w), inv_jacobian)
    inner_j = lambda w, z: inner(grad_j(w), grad_j(z))

    vx, vy = v
    dvx, dvy = dv

    a = (inner(v, dv) + alpha * inner_j(vx, dvx) + inner_j(vy, dvy)) * det_jacobian * dx
    return a, lambda f: inner(f, dv) * det_jacobian * dx
