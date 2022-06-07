from firedrake import *
import logging as _logging
from firedrake.petsc import PETSc
from datetime import datetime
import pickle
import os
from pathlib import Path
import sys

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


def delta(u):
    return div(grad(u))


def bihelmholtz(v, dv, alp0=1, alp1=1, alp2=1):
    return (alp0*inner(v, dv)
          + alp1*inner(grad(v), grad(dv))
          + alp2*inner(delta(v), delta(dv))) * dx


def trihelmholtz(v, dv, alpha):
    alp0 = Constant(alpha ** 0)
    alp1 = Constant(alpha ** 2)
    alp2 = Constant(alpha ** 4)
    alp3 = Constant(alpha ** 6)

    vx, vy = v
    dvx, dvy = dv
    return (alp0*inner(v, dv)
          + alp1*inner(grad(vx), grad(dvx))
          + alp1*inner(grad(vy), grad(dvy))

          + alp2*inner(delta(vx), delta(dvx))
          + alp2*inner(delta(vy), delta(dvy))

          + alp3*inner(grad(delta(vx)), grad(delta(dvx)))
          + alp3*inner(grad(delta(vy)), grad(delta(dvy)))) * dx


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


def shape_function(DG, mesh_tag):
    v, dv = TrialFunction(DG), TestFunction(DG)
    shape = Function(DG, name="shape_function")
    solve(v*dv*dx == dv*dx(mesh_tag), shape)
    return shape


def shape_normal(mesh, VDGT):
    mesh_tag = 10
    n = FacetNormal(mesh)
    N, dN = TrialFunction(VDGT), TestFunction(VDGT)
    sn = Function(VDGT, name="shape_normal")
    w = Constant(10)
    solve(dot(N,dN)('-')*dS + dot(N,dN)*ds == w*dot(n,dN)('-')*dS(mesh_tag), sn)
    x, y = SpatialCoordinate(mesh)
    return as_vector((sign(x)*abs(sn[0]), sign(y)*abs(sn[1])))


def pprint(s):
    PETSc.Sys.Print(s)


def trace_interpolate(V, f):
    u, v = TrialFunction(V), TestFunction(V)
    w = Function(V)
    a = inner(u, v)('+')*dS + inner(u, v)*ds
    L = dot(f, v)('+')*dS + dot(f, v)*ds
    solve(a == L, w)
    return w


