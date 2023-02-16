from dataclasses import dataclass, field

import numpy as np
import scipy.special
from numpy import nan_to_num
from typing import Type

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
        self.template = template

        # Function spaces
        self.order_XW = 4
        self.order_mismatch = 1

        # initial mesh, for mismatch
        self.CG_order_mismatch = VectorFunctionSpace(self.mesh, "CG", self.order_mismatch, dim=2)
        self.initial_mesh = Mesh(Function(self.CG_order_mismatch).interpolate(self.mesh.coordinates), comm=self.communicator)
        self.ShapeSpace = FunctionSpace(self.initial_mesh, "CG", self.order_mismatch)
        self.shape_moved_original_mesh = Function(self.ShapeSpace)
        
        self.Lagrange_XW = VectorFunctionSpace(self.mesh, "CG", degree=self.order_XW, dim=2)  # for coordinate fields
        self.mesh = Mesh(Function(self.Lagrange_XW).interpolate(self.mesh.coordinates), comm=self.communicator)
        self.VDGT = VectorFunctionSpace(self.mesh, "DGT", degree=self.parameters.momentum_degree, dim=2)  # for momentum
        self.XW = VectorFunctionSpace(self.mesh, "WXH3NC", degree=self.order_XW, dim=2)
        self.MomentumSpace = FunctionSpace(self.mesh, "DGT", self.parameters.momentum_degree)  # for momentum signal
        self.velocity_bcs = AxisAlignedDirichletBC(self.XW, Function(self.XW), "on_boundary")
        self.XW_order_orig_coords = self.mesh.coordinates.copy(deepcopy=True)

        # Velocity, momentum and diffeo
        self.u = Function(self.XW)
        self.momentum = Function(self.MomentumSpace)
        self.diffeo = Function(self.mesh.coordinates.function_space())  # don't use `self.Lagrange_XW` where, libsupermesh complains!
        self.diffeo_xw = Function(self.XW)

        # Functions we'll need for the source term/visualisation
        self.ShapeSpaceDG = FunctionSpace(self.mesh, "DG", 0)  # for visualisation
        self.shape_function = utils.shape_function(self.ShapeSpaceDG, INNER_TAG)

        self.kappa = 10  # how much to smoothen
        self.ShapeSpaceMovingMesh = FunctionSpace(self.mesh, "CG", self.order_mismatch)
        self.smooth_shape_function = self.smoothen_shape(self.shape_function)

        # Velocity solver
        self.MomentumTrace = VectorFunctionSpace(self.mesh, "HDiv Trace", degree=self.order_XW - 1, dim=2)  # for coordinate fields
        v, dv = TrialFunction(self.XW), TestFunction(self.XW)
        self.p_fun = Function(self.MomentumTrace).assign(1)  # dummy
        rhs = inner(self.p_fun, dv)('+') * dS(CURVE_TAG)
        h3_form = trihelmholtz(v, dv, self.parameters.alpha)
        lvp = LinearVariationalProblem(h3_form, rhs, self.u, bcs=self.velocity_bcs)
        self.lvs = LinearVariationalSolver(lvp, solver_parameters=self.parameters.velocity_solver_parameters)

    def smoothen_shape(self, shape_function: Function):
        v, dv = TrialFunction(self.ShapeSpaceMovingMesh), TestFunction(self.ShapeSpaceMovingMesh)
        a = (inner(v, dv) + self.kappa * inner(grad(v), grad(dv))) * dx
        rhs = inner(shape_function, dv) * dx
        smooth_function = Function(self.ShapeSpaceMovingMesh)
        solve(a == rhs, smooth_function, bcs=DirichletBC(self.ShapeSpaceMovingMesh, 0., "on_boundary"))
        return smooth_function

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

    def velocity_solve(self):
        p = dot(
            transpose(inv(grad(self.diffeo_xw))),
            self.momentum * utils.shape_normal(self.mesh, self.VDGT)
        )
        utils.trace_project(self.p_fun.function_space(), p, CURVE_TAG, self.p_fun)
        self.lvs.solve()

    def momentum_function(self):
        """ Used in the inverse problem solver. """
        return Function(self.MomentumSpace)

    def dump_parameters(self):
        self._logger.info(f"{self.parameters}")
        self._logger.info(f"kappa = {self.kappa}")

    def update_mesh(self, coords):
        self.mesh.coordinates.assign(coords)
        self.mesh.clear_spatial_index()

    def smooth_shape_function_initial_mesh(self):
        # evaluate the smooth moved indicator on the original mesh
        self.shape_moved_original_mesh.dat.data[:] = (
            self.smooth_shape_function.at(
                self.initial_mesh.coordinates.dat.data_ro,
                tolerance=1e-03,
                dont_raise=True,
            )
        )
        self.shape_moved_original_mesh.dat.data[:] = nan_to_num(
            self.shape_moved_original_mesh.dat.data[:],
            nan=0.0,
        )
        #utils.my_heaviside(self.shape_moved_original_mesh)
        return self.shape_moved_original_mesh


class AxisAlignedDirichletBC(DirichletBC):
    axis_aligned = True


def trihelmholtz(u, dv, alpha):
    m = 3
    choose_m = lambda j: scipy.special.binom(m, j)

    def Diff(f, j):
        if j == 0:
            return f
        elif j % 2 == 0:
            return div(Diff(f, j - 1))
        else:
            return grad(Diff(f, j - 1))
    form = inner(u, dv)
    ux, uy = u
    vx, vy = dv
    for ui, vi in zip([ux, uy], [vx, vy]):
        for j in range(0, m + 1):
            Djui = Diff(ui, j)
            Djvi = Diff(vi, j)
            form += alpha**j * choose_m(j) * inner(Djui, Djvi)
    return form * dx
