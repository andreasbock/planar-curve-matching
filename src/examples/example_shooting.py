from pathlib import Path
from firedrake import *
from src.shoot import GeodesicShooter
import src.utils as utils

timesteps = 5
mesh_file = "../meshes/mesh0.msh"
base = Path("RESULTS_SHOOTING")
logger = utils.Logger(base / "examples.log")

# Simple translation
name = 'A'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), logger)

x, _ = SpatialCoordinate(gs.mesh)
p0 = Constant(7)*cos(2*pi*x/5)
_, shape_function = gs.shoot(p0, timesteps)
File(logger.logger_dir / f"shape_{name}.pvd").write(shape_function)

# Triangle "C" shape
name = 'B'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), logger)
x, y = SpatialCoordinate(gs.mesh)
p0 = conditional(y < 0, -7*sign(y), 9*exp(-x**2/5))
_, shape_function = gs.shoot(p0, timesteps)
File(logger.logger_dir / f"shape_{name}.pvd").write(shape_function)

# Weird "C" shape
name = 'C'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), logger)
x, y = SpatialCoordinate(gs.mesh)
#p_vec = as_vector((conditional(x<-1, 2*10**2*exp(-(y**2/5)), 0), -40*sin(x/5)*abs(1*y)))
p0 = conditional(x < -1, 10*exp(-(y**2/5)), 0)
_, shape_function = gs.shoot(p0, timesteps)
File(logger.logger_dir / f"shape_{name}.pvd").write(shape_function)
