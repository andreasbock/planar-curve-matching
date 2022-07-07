from dataclasses import dataclass, field
from typing import Type

from firedrake import *

import src.utils as utils
from src.curves import Curve
from src.mesh_generation.mesh_generation import MeshGenerationParameters, generate_mesh


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
            template: Curve,
            mesh_parameters: MeshGenerationParameters,
            shooting_parameters: ShootingParameters = None,
    ):
        self.template = template
        self.mesh_path = generate_mesh(mesh_parameters, template, logger.logger_dir)
        self.mesh = Mesh(str(self.mesh_path))
        self._logger = logger
        self.parameters = shooting_parameters or ShootingParameters()
        self._solver_parameters = self.parameters.velocity_solver_parameters

        # Function spaces
        self.XW = VectorFunctionSpace(self.mesh, "WXH3NC", degree=4, dim=2)
        self.DG = FunctionSpace(self.mesh, "DG", 0)  # for plotting inside the curve
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=self.parameters.momentum_degree, dim=2)  # for momentum
        self.VCG = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.DGT = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal

        self.shape_function = utils.shape_function(self.DG, mesh_parameters.curve_tag)
        self.phi = project(SpatialCoordinate(self.mesh), self.XW)
        self.h = utils.compute_facet_area(self.mesh)

        # set up velocity problem
        self.u = Function(self.XW)
        self.p = self.momentum_function()
        v, dv = TrialFunction(self.XW), TestFunction(self.XW)
        a = velocity_lhs(v, dv, self.phi, self.parameters.alpha)

        # compute number of edges that form the initial curve
        h_inv = inv(self.h)
        n = assemble(h_inv('+')*dS(mesh_parameters.curve_tag))  # reference interval has measure 1
        shape_normal = utils.shape_normal(self.mesh, self.VDGT)
        rhs = (Constant(2*pi/n) * h_inv * inner(dot(transpose(inv(grad(self.phi))), self.p * shape_normal), dv))('+') * dS(mesh_parameters.curve_tag)
        bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")
        velocity_problem_u = LinearVariationalProblem(a, rhs, self.u, bcs=bcs, constant_jacobian=False)
        self.velocity_solver_u = LinearVariationalSolver(velocity_problem_u, solver_parameters=self._solver_parameters)

    def shoot(self, momentum_signal: function) -> Diffeomorphism:
        dt = Constant(1 / self.parameters.time_steps)
        x, y = SpatialCoordinate(self.mesh)
        self.p = momentum_signal(x, y)

        for t in range(self.parameters.time_steps):
            self._logger.info(f"Shooting... t = {t}")
            self.velocity_solver_u.solve()
            self.phi += self.u * dt

        # move the mesh for visualisation
        self._update_mesh()
        return self.phi

    def momentum_function(self):
        return Function(self.DGT)

    def dump_parameters(self):
        self._logger.info(f"{self._solver_parameters}")


    def _update_mesh(self):
        self.mesh.coordinates.assign(project(self.phi, self.VCG))
        self.mesh.clear_spatial_index()

    def _check_points(self, points):
        """ Sanity check whether the points given by `points` lies within the
        spatial domain. """
        eps = self._domain_size + 1e-05
        within_bounds = (points > -eps).all() and (points < eps).all()
        if not within_bounds:
            raise Exception(f"Template points moved outside domain: {points}.")


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
