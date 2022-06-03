from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
import time
import csv

"""
Convergence test for WuXu element
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

def solve_ex3(h):
    mesh = UnitSquareMesh(h, h)
    x, y = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    PETSc.Sys.Print("\t setting up spaces...")
    V = FunctionSpace(mesh, "WuXu", 7)
    u, v = TrialFunction(V), TestFunction(V)

    # analytical paper
    uex = Constant(100) * (x**4*(1-x)**4*y**4*(1-y)**4)
    f = - delta(delta(delta(uex)))

    a = inner(grad(delta(u)), grad(delta(v)))*dx
    L = inner(f, v)*dx

    uh = Function(V)
    bcs = AxisAlignedDirichletBC(V, Function(V), "on_boundary")
    start_time = time.time()
    solve(a == L, uh, solver_parameters=params, bcs=bcs)
    exec_time = time.time() - start_time

    return sqrt(assemble((uex-uh)**2*dx)), \
           sqrt(assemble(inner(grad(uex-uh), grad(uex-uh))*dx)), \
           sqrt(assemble(inner(div(grad(uex-uh)), div(grad(uex-uh)))*dx)), \
           sqrt(assemble(inner(grad(div(grad(uex-uh))), grad(div(grad(uex-uh))))*dx)), \
           exec_time

l2s, h1s, h2s, h3s = [], [], [], []
res = [2**i for i in range(2, 8)]
exec_times = []

# warm up
solve_ex3(4)

for h in res:
    PETSc.Sys.Print("h = {}".format(h))
    l2, h1, h2, h3, exec_time = solve_ex3(h)
    l2s.append(l2)
    h1s.append(h1)
    h2s.append(h2)
    h3s.append(h3)
    exec_times.append(exec_time)

def convergence_rate(errors):
    return np.array([np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
            for i in range(len(res)-1)])

l2rates = ["-"] + [format(x, '.2f') for x in convergence_rate(l2s)]
h1rates = ["-"] + [format(x, '.2f') for x in convergence_rate(h1s)]
h2rates = ["-"] + [format(x, '.2f') for x in convergence_rate(h2s)]
h3rates = ["-"] + [format(x, '.2f') for x in convergence_rate(h3s)]

l2s = [format(x, '.4e') for x in l2s]
h1s = [format(x, '.4e') for x in h1s]
h2s = [format(x, '.4e') for x in h2s]
h3s = [format(x, '.4e') for x in h3s]
exec_times = [format(x, '.4e') for x in exec_times]

titles = "hmesh,ltwoerr,ltwoord,honeerr,honeord,htwoerr,htwoord,hthreeerr,hthreeord,exec".split(",")
data = [res, l2s, l2rates, h1s, h1rates, h2s, h2rates, h3s, h3rates, exec_times]
path = "../../tex/"
with open(path + 'wuxu3.csv', mode='w+') as cfile:
    cwriter = csv.writer(cfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    cwriter.writerow(titles)
    for row in zip(*data):
        cwriter.writerow(row)

print("L2 errors: {}".format(l2s))
print("H1 errors: {}".format(h1s))
print("H2 errors: {}".format(h2s))
print("H3 errors: {}".format(h3s))

print("L2 convergence rates: {}\n".format(l2rates))
print("H1 convergence rates: {}\n".format(h1rates))
print("H2 convergence rates: {}\n".format(h2rates))
print("H3 convergence rates: {}\n".format(h3rates))

print("Execution times: {}\n".format(exec_times))
