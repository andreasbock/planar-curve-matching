from dataclasses import dataclass, field
from typing import Type

from firedrake import *
from pathlib import Path

import src.utils as utils
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
    alpha: float = 0.5
    time_steps: int = 10


class GeodesicShooter:
    def __init__(
        self,
        logger: utils.Logger,
        mesh_path: Path,
        shooting_parameters: ShootingParameters = None,
    ):
        self.mesh_path = mesh_path
        self.mesh = Mesh(str(self.mesh_path))
        self._logger = logger
        self.parameters = shooting_parameters or ShootingParameters()
        self._solver_parameters = self.parameters.velocity_solver_parameters

        # Function spaces
        self.XW = VectorFunctionSpace(self.mesh, "WXH3NC", degree=4, dim=2)
        self.DG = FunctionSpace(self.mesh, "DG", 0)
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=self.parameters.momentum_degree, dim=2)  # for momentum
        self.VCG = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.DGT = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal
        self.velocity_bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")

        # Velocity, momentum and diffeo
        self.u = Function(self.XW)
        self.p = None
        self.phi = project(SpatialCoordinate(self.mesh), self.XW)

        # Functions we'll need for the source term/visualisation
        self.shape_function = utils.shape_function(self.mesh, CURVE_TAG)
        self.h_inv = inv(utils.compute_facet_area(self.mesh))
        self.n = assemble(self.h_inv('+') * dS(CURVE_TAG))  # reference interval has measure 1
        self.shape_normal = utils.shape_normal(self.mesh, self.VDGT)

    def shoot(self, momentum_signal: function):
        dt = Constant(1 / self.parameters.time_steps)
        x, y = SpatialCoordinate(self.mesh)
        self.p = momentum_signal(x, y)
        us = []
        for t in range(self.parameters.time_steps):
            self._logger.info(f"Shooting... t = {t}")
            self.velocity_solver()
            us.append(self.u.copy())
            self.phi.assign(self.phi + self.u * dt)

        # move the mesh for visualisation
        self._update_mesh()
        return self.phi, us

    def velocity_solver(self):
        v, dv = TrialFunction(self.XW), TestFunction(self.XW)
        a = velocity_lhs(v, dv, self.phi, self.parameters.alpha)#, self.mesh)
        rhs = (Constant(2*pi/self.n) * self.h_inv
               * inner(dot(transpose(inv(grad(self.phi))), self.p*self.shape_normal), dv))('+') * dS(CURVE_TAG)
        solve(a == rhs, self.u, bcs=self.velocity_bcs, solver_parameters=self._solver_parameters)

    def momentum_function(self):
        """ Used in the inverse problem solver. """
        return Function(self.DGT)

    def dump_parameters(self):
        self._logger.info(f"{self._solver_parameters}")

    def _update_mesh(self):
        self.mesh.coordinates.assign(project(self.phi, self.VCG))
        self.mesh.clear_spatial_index()


class AxisAlignedDirichletBC(DirichletBC):
    axis_aligned = True


def velocity_lhs(v, dv, phi, alpha):
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
