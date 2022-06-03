from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
import time
import csv

"""
Biharmonic convergence test for WuXu element.
"""
n = 2  # two dimensions
m = n + 1

params = {"snes_type": "newtonls",
          "snes_linesearch_type": "basic",
          "snes_lag_jacobian": -2,
          "snes_lag_preconditioner": -2,
          "ksp_type": "preonly",
          "snes_max_it": 3,
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "snes_rtol": 1e-16,
          "snes_atol": 1e-25}

def delta(u):
    return div(grad(u))

def interp(expr, V):
    w = Function(V)
    u, v = TrialFunction(V), TestFunction(V)
    solve(u*v*dx == expr*v*dx, w, solver_parameters=params)
    return w

def dds(u, *n):
    for s in n:
        u = dot(grad(u), s)
    return u

def solve_ex1(h):
    mesh = UnitSquareMesh(h, h)
    x, y = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    PETSc.Sys.Print("\t setting up spaces...")
    V = FunctionSpace(mesh, "WuXu", 7)
    u, v = TrialFunction(V), TestFunction(V)

    n = FacetNormal(mesh)
    t = dot(as_tensor([[0, -1], [1, 0]]), n)
    h = CellSize(mesh)
    beta1 = Constant(40)
    beta2 = Constant(40)

    uex = Constant(100) * (x*(1-x)*y*(1-y))**2
    f = delta(delta(uex))

    nu = Constant(0.225)
    nu = Constant(0)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = (inner(delta(u), delta(v))*dx -
         (1-nu)*(2*inner(u.dx(0).dx(0), v.dx(1).dx(1)) +
                 2*inner(u.dx(1).dx(1), v.dx(0).dx(0)) -
                 4*inner(u.dx(0).dx(1), v.dx(0).dx(1)))*dx +
         inner(dds(delta(u), n) - 2*(1-nu) * dds(u, t, t, n), v)*ds +
         inner(u, dds(delta(v), n) - 2*(1-nu) * dds(v, t, t, n))*ds +
         beta1/h**2 * inner(u, v)*ds -
         inner(delta(u) - 2*(1-nu)*dds(u, t, t), dds(v, n))*ds -
         inner(dds(u, n), delta(v) - 2*(1-nu)*dds(v, t, t))*ds +
         beta2/h * inner(dds(u, n), dds(v, n))*ds)

    L = inner(f, v)*dx
    uh = Function(V)
    # solve
    start_time = time.time()
    solve(a == L, uh, solver_parameters=params)
    exec_time = time.time() - start_time

    return sqrt(assemble((uh-uex)**2*dx)), \
           sqrt(assemble(inner(grad(uh-uex), grad(uh-uex))*dx)), \
           sqrt(assemble(inner(div(grad(uh-uex)), div(grad(uh-uex)))*dx)), \
           exec_time

l2s, h1s, h2s = [], [], []
exec_times = []
res = [2**i for i in range(3, 9)]

# warm up cache
solve_ex1(4)

for h in res:
    PETSc.Sys.Print("h = {}".format(h))
    l2, h1, h2, exec_time = solve_ex1(h)
    l2s.append(l2)
    h1s.append(h1)
    h2s.append(h2)
    exec_times.append(exec_time)

def convergence_rate(errors):
    return np.array([np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
            for i in range(len(res)-1)])

l2rates = ["-"] + [format(x, '.2f') for x in convergence_rate(l2s)]
h1rates = ["-"] + [format(x, '.2f') for x in convergence_rate(h1s)]
h2rates = ["-"] + [format(x, '.2f') for x in convergence_rate(h2s)]

l2s = [format(x, '.4e') for x in l2s]
h1s = [format(x, '.4e') for x in h1s]
h2s = [format(x, '.4e') for x in h2s]
exec_times = [format(x, '.3') for x in exec_times]


titles = "hmesh,ltwoerr,ltwoord,honeerr,honeord,htwoerr,htwoord,exec".split(",")
data = [res, l2s, l2rates, h1s, h1rates, h2s, h2rates, exec_times]

path = "../../tex/"
with open(path + 'wuxu_biharmonic.csv', mode='w+') as cfile:
    cwriter = csv.writer(cfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    cwriter.writerow(titles)
    for row in zip(*data):
        cwriter.writerow(row)

print("L2 errors: {}".format(l2s))
print("H1 errors: {}".format(h1s))
print("H2 errors: {}".format(h2s))

print("L2 convergence rates: {}\n".format(l2rates))
print("H1 convergence rates: {}\n".format(h1rates))
print("H2 convergence rates: {}\n".format(h2rates))

print("Execution times: {}\n".format(exec_times))
