import firedrake
from firedrake import *
import logging as _logging
from datetime import datetime
import pickle
import os
from pathlib import Path
import sys
import numpy as np


def project_root() -> Path:
    return Path(__file__).parent.parent


def date_string():
    return datetime.now().strftime("%Y-%m-%d|%H.%M.%S")


def create_dir_from_path_if_not_exists(path):
    path = os.path.dirname(path)
    if not os.path.exists(path) and path != '':
        os.makedirs(path)


class Logger:

    def __init__(self, logger_path):
        self.logger_path = logger_path
        self.logger_dir = self.logger_path.parent
        self._logger = _basic_logger(logger_path)

    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def critical(self, msg):
        self._logger.critical(msg)

    def log(self, msg, level):
        self._logger.log(msg, level)


def _basic_logger(logger_path: Path) -> _logging.Logger:
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
    directory = os.path.dirname(name)
    os.makedirs(directory, exist_ok=True)
    po = open("{}.pickle".format(name), "wb")
    pickle.dump(f, po)
    po.close()


def pload(name):
    po = open("{}.pickle".format(name), "rb")
    f = pickle.load(po)
    po.close()
    return f


def shape_function(mesh: firedrake.Mesh, mesh_tag: int):
    function_space = FunctionSpace(mesh, "CG", 1)
    v, dv = TrialFunction(function_space), TestFunction(function_space)
    shape = Function(function_space, name="shape_function")
    solve(v*dv*dx == dv('+')*dS(mesh_tag), shape)
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


def soft_eval(diffeo, fs, points):
    soft_diffeo = Function(fs).project(diffeo)
    return np.array(soft_diffeo.at(points))
