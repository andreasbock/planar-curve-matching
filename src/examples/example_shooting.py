from firedrake import *
from ..shoot import GeodesicShooter
import numpy as np
from pathlib import Path

t = np.linspace(0, 2*np.pi, 5)[:-1]  # doesn't matter here really

timesteps = 5
mesh_file = "../meshes/mesh0.msh"
base = Path("RESULTS_SHOOTING")

# Simple translation
name = 'A'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), base / f"example_{name}")
x, _ = SpatialCoordinate(gs.mesh)
p0 = Constant(7)*cos(2*pi*x/5)
gs.shoot(p0, timesteps)

# Triangle "C" shape
name = 'B'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), base / f"example_{name}")
x, y = SpatialCoordinate(gs.mesh)
p = conditional(y < 0, -7*sign(y), 9*exp(-x**2/5))
gs.shoot(p, timesteps)

# Weird "C" shape
name = 'C'
print(f"Test: {name}")
gs = GeodesicShooter(Mesh(mesh_file), base / f"example_{name}")
x, y = SpatialCoordinate(gs.mesh)
#p_vec = as_vector((conditional(x<-1, 2*10**2*exp(-(y**2/5)), 0), -40*sin(x/5)*abs(1*y)))
p = conditional(x<-1, 10*exp(-(y**2/5)), 0)#as_vector((, -40*sin(x/5)*abs(1*y)))
#p = Function(gs.VDGT).interpolate(p_vec)
gs.shoot(p, timesteps)
