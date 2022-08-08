from firedrake import *

process_per_ensemble_member = 2
my_ensemble = Ensemble(COMM_WORLD, process_per_ensemble_member)
ensemble_size = COMM_WORLD.size

mesh = UnitSquareMesh(20, 20, comm=my_ensemble.comm)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)

q = Constant(1)
u = Function(V).interpolate(q)
print(f"rank {my_ensemble.ensemble_comm.rank}, u = {norm(u)}")

u_sum = Function(V)
my_ensemble.allreduce(u, u_sum)

if my_ensemble.ensemble_comm.rank == 0:
    print(f"sum = {norm(u_sum)}")
    print(f"expected = {ensemble_size / process_per_ensemble_member}")
