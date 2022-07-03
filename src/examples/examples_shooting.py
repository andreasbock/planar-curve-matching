from pathlib import Path
from firedrake import *
from src.shoot_pullback import GeodesicShooter
import src.utils as utils

EXAMPLES_SHOOTING_PATH = Path("RESULTS_EXAMPLES_SHOOTING")
logger = utils.Logger(EXAMPLES_SHOOTING_PATH / "examples.log")

timesteps = 5
mesh_file = "../meshes/mesh0.msh"

# Simple translation
name = 'A'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), logger)

x, _ = SpatialCoordinate(gs.mesh)
p0 = lambda x, y: Constant(7)*cos(2*pi*x/5)
gs.shoot(p0)
File(logger.logger_dir / f"shape_{name}.pvd").write(gs.shape_function)

# Triangle "C" shape
name = 'B'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), logger)
p0 = lambda x, y: conditional(y < 0, -7*sign(y), 9*exp(-x**2/5))
gs.shoot(p0)
File(logger.logger_dir / f"shape_{name}.pvd").write(gs.shape_function)

# Weird "C" shape
name = 'C'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), logger)
#p_vec = as_vector((conditional(x<-1, 2*10**2*exp(-(y**2/5)), 0), -40*sin(x/5)*abs(1*y)))
p0 = lambda x, y: conditional(x < -1, 10*exp(-(y**2/5)), 0)
gs.shoot(p0)
File(logger.logger_dir / f"shape_{name}.pvd").write(gs.shape_function)
