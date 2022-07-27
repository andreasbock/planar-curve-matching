from dataclasses import dataclass, field
from typing import Type

from firedrake import *
from pathlib import Path

import numpy as np

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
    alpha: float = 1
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
        self.XW = VectorFunctionSpace(self.mesh, "WXRobH3NC", degree=7, dim=2)
        self.DG = FunctionSpace(self.mesh, "DG", 0)
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=self.parameters.momentum_degree, dim=2)  # for momentum
        self.VCG = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.DGT = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal
        self.velocity_bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")

        # Velocity, momentum and diffeo
        self.phi = project(SpatialCoordinate(self.mesh), self.XW)
        self.u = Function(self.XW)
        self.p = None

        # Functions we'll need for the source term/visualisation
        self.shape_function = utils.shape_function(self.mesh, CURVE_TAG)
        self.h_inv = inv(utils.compute_facet_area(self.mesh))
        self.n = assemble(self.h_inv('+') * dS(CURVE_TAG))  # reference interval has measure 1
        self.shape_normal = utils.shape_normal(self.mesh, self.VDGT)

    def shoot(self, momentum_signal: function):
        dt = Constant(1 / self.parameters.time_steps)
        x, y = SpatialCoordinate(self.mesh)
        self.p = momentum_signal(x, y) * self.shape_normal
        u_norms, p_norms = [], []

        self._logger.info("Shooting...")
        for t in range(self.parameters.time_steps):
            u_norm, p_norm = self.velocity_solve()
            self.phi.assign(self.phi + self.u * dt)

            u_norms.append(u_norm)
            p_norms.append(p_norm)

        # move the mesh for visualisation
        self._update_mesh()
        return self.phi, u_norms, p_norms

    def velocity_solve(self):
        v, dv = TrialFunction(self.XW), TestFunction(self.XW)
        w, z = Function(self.XW), Function(self.XW)

        momentum = dot(transpose(inv(grad(self.phi))), self.p)
        momentum_form = Constant(2*pi) * self.h_inv * momentum
        rhs = dot(momentum_form, dv)('+') * dS(CURVE_TAG)

        J = grad(self.phi)
        detJ = det(J)
        J_inv = inv(J)

        a, l = h1_form(v, dv, J_inv, detJ, alpha=self.parameters.alpha)
        solve(a == rhs, z, bcs=self.velocity_bcs, solver_parameters=self._solver_parameters)
        solve(a == l(z), w, bcs=self.velocity_bcs, solver_parameters=self._solver_parameters)
        solve(a == l(w), self.u, bcs=self.velocity_bcs, solver_parameters=self._solver_parameters)

        momentum_norm = np.sqrt(assemble(inner(momentum_form, momentum)('+') * dS(CURVE_TAG)))
        u_norm = np.sqrt(assemble(h1_form(self.u, self.u, J_inv, detJ, alpha=self.parameters.alpha)[0]))
        return u_norm, momentum_norm

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


def h1_form(v, dv, J_inv, detJ, alpha):
    gradJ = lambda w: dot(grad(w), J_inv)

    vx, vy = v
    dvx, dvy = dv

    a = (inner(v, dv) + alpha * (inner(gradJ(vx), gradJ(dvx)) + inner(gradJ(vy), gradJ(dvy)))) * detJ * dx
    return a, lambda f: inner(f, dv) * detJ * dx
