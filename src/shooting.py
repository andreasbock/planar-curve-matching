from dataclasses import dataclass, field
from typing import Type, List, Any

import firedrake
from firedrake import *
from pathlib import Path

import numpy as np

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
    momentum_degree: int = 1
    alpha: float = 1
    time_steps: int = 10


@dataclass
class CurveResult:
    diffeo: firedrake.Function
    velocity_norms: List[Any]
    momentum_norms: List[Any]

    def eval_diffeo(self, points):
        return self.diffeo.at(points)


class GeodesicShooter:
    def __init__(
        self,
        logger: utils.Logger,
        mesh_path: Path,
        template: Curve = None,
        shooting_parameters: ShootingParameters = None,
        communicator=None,
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
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=self.parameters.momentum_degree, dim=2)  # for momentum
        self.VCG = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.DGT = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal
        self.velocity_bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")

        # Velocity, momentum and diffeo
        self.diffeo = project(SpatialCoordinate(self.mesh), self.XW)
        self.u = Function(self.XW)
        self.momentum = None

        # Functions we'll need for the source term/visualisation
        self.shape_function = utils.shape_function(self.mesh, CURVE_TAG)
        self.h_inv = inv(utils.compute_facet_area(self.mesh))
        self.n = assemble(self.h_inv('+') * dS(CURVE_TAG))  # reference interval has measure 1
        self.shape_normal = utils.shape_normal(self.mesh, self.VDGT)

    def shoot(self, momentum: Momentum) -> CurveResult:
        dt = Constant(1 / self.parameters.time_steps)
        if isinstance(momentum, Function):
            self.momentum = momentum
        else:
            # mostly for manufactured solutions
            x, y = SpatialCoordinate(self.mesh)
            self.momentum = momentum.signal(x, y)

        u_norms, p_norms = [], []

        self._logger.info("Shooting...")
        for t in range(self.parameters.time_steps):
            u_norm, p_norm = self.velocity_solve()
            self.diffeo.assign(self.diffeo + self.u * dt)

            u_norms.append(u_norm)
            p_norms.append(p_norm)

        # move the mesh for visualisation
        self._update_mesh()
        return CurveResult(
            diffeo=self.diffeo,
            velocity_norms=u_norms,
            momentum_norms=p_norms,
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

        momentum_norm = np.sqrt(assemble(inner(momentum_form, momentum)('+') * dS(CURVE_TAG)))
        u_norm = np.sqrt(assemble(h1_form(self.u, self.u, inv_jacobian, det_jacobian, alpha=self.parameters.alpha)[0]))
        return u_norm, momentum_norm

    def evaluate_curve(self, angles):
        if self.template is None:
            raise Exception("No template.pickle `Curve` object was provided.")

        # map angles to template.pickle
        template_points = self.template.at(angles)

        # compose with diffeomorphism

        # TODO: replace with Xu-Wu point evaluation!
        #return self.diffeo.at(template_points)
        cg_diffeo = Function(self.VCG).project(self.diffeo)
        return np.array(cg_diffeo.at(template_points))

    def momentum_function(self):
        """ Used in the inverse problem solver. """
        return Function(self.DGT)

    def dump_parameters(self):
        self._logger.info(f"{self.parameters}")

    def _update_mesh(self):
        self.mesh.coordinates.assign(project(self.diffeo, self.VCG))
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
