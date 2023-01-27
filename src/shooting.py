from dataclasses import dataclass, field
from numpy import nan_to_num
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
    alpha: float = 0.5
    time_steps: int = 20


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

        self.h_inv = inv(utils.compute_facet_area(self.mesh))  # TODO: used?

        # Function spaces
        self.order_XW = 4
        self.order_mismatch = 1

        # initial mesh, for mismatch
        self.CG_order_mismatch = VectorFunctionSpace(self.mesh, "CG", self.order_mismatch, dim=2)
        self.initial_mesh = Mesh(Function(self.CG_order_mismatch).interpolate(self.mesh.coordinates), comm=self.communicator)
        self.ShapeSpace = FunctionSpace(self.initial_mesh, "CG", self.order_mismatch)

        self.Lagrange_XW = VectorFunctionSpace(self.mesh, "CG", degree=self.order_XW, dim=2)  # for coordinate fields
        self.mesh = Mesh(Function(self.Lagrange_XW).interpolate(self.mesh.coordinates), comm=self.communicator)
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=self.parameters.momentum_degree, dim=2)  # for momentum
        self.XW = VectorFunctionSpace(self.mesh, "WXH3NC", degree=self.order_XW, dim=2)
        self.VCG1 = VectorFunctionSpace(self.mesh, "CG", degree=1, dim=2)  # for coordinate fields
        self.MomentumSpace = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal
        self.velocity_bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")

        self.XW_order_orig_coords = self.mesh.coordinates.copy(deepcopy=True)
        self.diffeo = Function(self.mesh.coordinates.function_space())  # don't use `self.Lagrange_XW` where, libsupermesh complains!
        self.diffeo_xw = Function(self.XW)
        # Velocity, momentum and diffeo
        self.u = Function(self.XW)
        self.w, self.z = Function(self.XW), Function(self.XW)
        self.momentum = Function(self.MomentumSpace)

        # Functions we'll need for the source term/visualisation
        self.shape_function = utils.shape_function(self.mesh, INNER_TAG)

    def shoot(self, momentum: Momentum):
        self.update_mesh(self.XW_order_orig_coords)
        self.diffeo_xw.project(self.XW_order_orig_coords)
        Dt = Constant(1 / self.parameters.time_steps)

        if not isinstance(momentum, Function):
            x, y = SpatialCoordinate(self.mesh)
            momentum = Function(self.MomentumSpace).interpolate(momentum.signal(x, y))
        self.momentum.assign(momentum)

        for t in range(self.parameters.time_steps):
            self.velocity_solve()
            self.diffeo_xw.assign(self.diffeo_xw + self.u * Dt)
            self.diffeo.project(self.diffeo_xw)
            self.update_mesh(self.diffeo)

        # move the mesh for visualisation
        return CurveResult(
            diffeo=self.diffeo,
        )

    def velocity_solve(self):
        v, dv = TrialFunction(self.XW), TestFunction(self.XW)

        momentum_form = dot(transpose(inv(grad(self.diffeo_xw))), self.momentum * utils.shape_normal(self.mesh, self.VDGT))
        rhs = dot(as_vector(Constant(2*pi) * momentum_form), dv)('+') * dS(CURVE_TAG)

        alp0, alp1, alp2, alp3 = 1, self.parameters.alpha, self.parameters.alpha**2, self.parameters.alpha**3
        h3_form = trihelmholtz(v, dv, alp0, alp1, alp2, alp3)
        solve(h3_form == rhs, self.u, bcs=DirichletBC(self.XW, 0, "on_boundary"))

    def momentum_function(self):
        """ Used in the inverse problem solver. """
        return Function(self.MomentumSpace)

    def dump_parameters(self):
        self._logger.info(f"{self.parameters}")

    def update_mesh(self, coords):
        self.mesh.coordinates.assign(coords)
        self.mesh.clear_spatial_index()

    def shape_function_initial_mesh(self):
        # evaluate the moved indicator on the original mesh
        indicator_moved_original_mesh = Function(
            self.ShapeSpace,
            self.shape_function.at(
                self.initial_mesh.coordinates.dat.data_ro,
                tolerance=1e-03,
                dont_raise=True,
            )
        )
        indicator_moved_original_mesh.dat.data[:] = nan_to_num(
            indicator_moved_original_mesh.dat.data[:],
            nan=1.0,
        )
        utils.my_heaviside(indicator_moved_original_mesh)
        return indicator_moved_original_mesh


class AxisAlignedDirichletBC(DirichletBC):
    axis_aligned = True


def trihelmholtz(v, dv, alp0=1., alp1=1., alp2=1., alp3=1.):
    vx, vy = v
    dvx, dvy = dv
    return (
            Constant(alp0)*inner(v, dv)
            + Constant(alp1)*inner(grad(vx), grad(dvx))
            + Constant(alp1)*inner(grad(vy), grad(dvy))
            + Constant(alp2)*inner(delta(vx), delta(dvx))
            + Constant(alp2)*inner(delta(vy), delta(dvy))
            + Constant(alp3)*inner(grad(delta(vx)), grad(delta(dvx)))
            + Constant(alp3)*inner(grad(delta(vy)), grad(delta(dvy)))
            ) * dx


def delta(u):
    return div(grad(u))
