from firedrake import *
from pathlib import Path
import subprocess

import src.utils as utils
from src.meshes.geometry import *
from meshes.curves import CURVES


_GMSH_BINARY = "gmsh"
_SHELL = '/bin/zsh'
_MESH_RESOLUTIONS = [1/(2*h) for h in range(1, 3)]


def geo_to_msh(geometry_file: Path) -> Path:
    msh_file = geometry_file.parent / f"{geometry_file.stem}.msh"
    gmsh_command = f"{_GMSH_BINARY} {geometry_file} -2 -o {msh_file}"
    print(f"Running the command: '{gmsh_command}'.")
    return_code = subprocess.call(gmsh_command, shell=True, executable=_SHELL)
    if return_code != 0:
        raise Exception("Gmsh command failed.")
    print(f"Ran gmsh on {geometry_file}, generated {msh_file}.")
    return msh_file


def msh_to_pvd(msh_file: Path, inside_tag: int = None) -> Path:
    mesh = Mesh(str(msh_file))
    function_space = FunctionSpace(mesh, "CG", 1)
    if inside_tag:
        indicator = utils.shape_function(function_space, mesh_tag=inside_tag)
    else:
        indicator = Function(function_space)

    pvd_file = msh_file.parent / f"{msh_file.stem}.pvd"
    File(pvd_file).write(indicator)
    print(f"Wrote {pvd_file}.")
    return pvd_file


def generate_mesh(
        params: MeshGenerationParameters,
        curve: Curve,
        base_path: Path,
) -> Path:
    path = base_path / f"h={params.mesh_size}"
    geo_file = path / f"{curve.name}.geo"

    write_geo_file(params, curve, geo_file)
    msh_file = geo_to_msh(geo_file)
    pvd_file = msh_to_pvd(msh_file, params.curve_tag)

    return pvd_file


if __name__ == "__main__":
    base_path = Path("meshes")

    for mesh_size in _MESH_RESOLUTIONS:
        for curve in CURVES:
            mesh_params = MeshGenerationParameters(mesh_size=mesh_size)
            generate_mesh(mesh_params, curve, base_path)