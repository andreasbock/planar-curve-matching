import matplotlib.pyplot as plt
import firedrake
from firedrake import *
import logging as _logging
from datetime import datetime
import pickle
import os
from pathlib import Path
import sys
import numpy as np


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def project_root() -> Path:
    return Path(__file__).parent.parent


def date_string():
    return datetime.now().strftime("%Y-%m-%d|%H.%M.%S")


def create_dir_from_path_if_not_exists(path):
    path = os.path.dirname(path)
    if not os.path.exists(path) and path != '':
        os.makedirs(path)


class Logger:

    def __init__(self, logger_path: Path, communicator=None):
        self.log_path = logger_path
        self.logger_dir: Path = self.log_path.parent
        self.logger_dir.mkdir(parents=True, exist_ok=True)
        self._logger = basic_logger(logger_path)
        self._communicator = communicator

    def info(self, msg):
        if self._communicator is not None:
            if self._communicator.Get_rank() == 0:
                self._logger.info(msg)
        else:
            self._logger.info(msg)

    def debug(self, msg):
        if self._communicator is not None:
            if self._communicator.Get_rank() == 0:
                self._logger.debug(msg)
        else:
            self._logger.debug(msg)

    def critical(self, msg):
        if self._communicator is not None:
            if self._communicator.Get_rank() == 0:
                self._logger.critical(msg)
        else:
            self._logger.critical(msg)

    def log(self, msg, level):
        if self._communicator is not None:
            if self._communicator.Get_rank() == 0:
                self._logger.log(msg, level)
        else:
            self._logger.log(msg, level)


def basic_logger(logger_path: Path) -> _logging.Logger:
    if not logger_path.parent.exists():
        logger_path.parent.mkdir()

    logger = _logging.getLogger(str(logger_path))

    logger.setLevel(logging.INFO)
    format_string = "%(asctime)s [%(levelname)s]: %(message)s"
    log_format = _logging.Formatter(format_string, "%Y-%m-%d %H:%M:%S")

    # Creating and adding the console handler
    console_handler = _logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Creating and adding the file handler
    file_handler = _logging.FileHandler(str(logger_path))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


def uniform_parameterisation(n):
    return np.linspace(0, 2 * np.pi, n + 1)[:-1]


def pdump(f, name):
    path = Path(name)
    if not path.parent.exists():
        path.parent.mkdir()

    po = open(f"{name}", "wb")
    pickle.dump(f, po)
    po.close()


def pload(name):
    po = open(f"{name}", "rb")
    f = pickle.load(po)
    po.close()
    return f


def plot_curves(u, path: Path, levels: int = 50):
    fig, axes = plt.subplots()
    levels = np.linspace(0, 1, levels + 1)
    contours = tricontourf(u, levels=levels, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(path)


def shape_function(mesh: firedrake.Mesh, mesh_tag: int, dim=2):
    if dim == 2:
        meas = dx
        function_space = FunctionSpace(mesh, "DG", 0)
        v, dv = TrialFunction(function_space), TestFunction(function_space)
    else:
        meas = dS
        function_space = FunctionSpace(mesh, "CG", 1)
        v, dv = TrialFunction(function_space), TestFunction(function_space)

    shape = Function(function_space, name="shape_function")
    solve(v*dv*dx == dv('+')*meas(mesh_tag), shape)
    return shape


def check_points(domain_min: float, domain_max: float, points: np.array):
    """ Sanity check whether the points given by `points` lies within the
    spatial domain. """
    within_bounds = (points > domain_min).all() and (points < domain_max).all()
    if not within_bounds:
        raise Exception(f"Template points moved outside domain: {points}.")


def shape_normal(mesh: firedrake.Mesh, vector_function_space: firedrake.VectorFunctionSpace):
    mesh_tag = 10
    n = FacetNormal(mesh)
    N, dN = TrialFunction(vector_function_space), TestFunction(vector_function_space)
    sn = Function(vector_function_space, name="shape_normal")
    solve(dot(N, dN)('-')*dS + dot(N, dN)*ds == dot(n, dN)('-')*dS(mesh_tag), sn)
    return sn


def trace_interpolate(function_space: FunctionSpace, f: Function = None, mesh_tag: int = None):
    if f is None:
        f = as_vector((1, 1))
    u, v = TrialFunction(function_space), TestFunction(function_space)
    w = Function(function_space)
    a = inner(u, v)('+')*dS + inner(u, v)*ds
    L = inner(f, v)('+')*dS(mesh_tag) + inner(f, v)*ds
    solve(a == L, w)
    return w


def compute_facet_area(mesh):
    fs = FunctionSpace(mesh, "DGT", 0)
    h = Function(fs)
    h_trial, h_test = TrialFunction(fs), TestFunction(fs)
    h_lhs = inner(h_trial, h_test)('+') * dS + inner(h_trial, h_test) * ds
    h_rhs = inner(FacetArea(mesh), h_test)('+') * dS + inner(FacetArea(mesh), h_test) * ds
    facetarea_problem = LinearVariationalProblem(h_lhs, h_rhs, h, constant_jacobian=False)
    facetarea_solver = LinearVariationalSolver(facetarea_problem)
    facetarea_solver.solve()
    return h


def csr_localisation(mesh: Mesh, tolerance: float):
    f = Function(VectorFunctionSpace(mesh, "DG", 0, dim=2)).interpolate(SpatialCoordinate(mesh))
    fd = f.dat.data
    rows, cols = [], []
    within = lambda x, y: np.linalg.norm(x - y) <= tolerance

    for i in range(len(fd)):
        for j in range(len(fd)):
            if within(fd[i], fd[j]):
                rows.append(i)
                cols.append(j)

    return rows, cols
