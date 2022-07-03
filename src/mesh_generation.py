import subprocess
from dataclasses import dataclass
from pathlib import Path

import firedrake
from firedrake import *

import src.utils as utils
from src.meshes.geometry import *
from src.curves import Curve, CURVES


_GMSH_BINARY = "gmsh"
_SHELL = '/bin/zsh'
_MESH_RESOLUTIONS = [1/(2*h) for h in range(1, 3)]


@dataclass
class MeshGenerationParameters:
    mesh_size: float
    min_xy: int = -10
    max_xy: int = 10
    curve_tag: int = 10
    inner_tag: int = 6
    outer_tag: int = 7


def write_geo_file(
    params: MeshGenerationParameters,
    curve: Curve,
    path_to_geo: Path,
):
    n = len(curve.points)
    points = ""
    for i, point in enumerate(curve.points):
        x, y = point
        points += f"Point({i + _OFFSET}) = {{{x}, {y}, 0, h}};\n"

    lines = ""
    for i in range(n):
        j = i + _OFFSET
        lines += f"Line({j}) = {{ {j}, {(i + 1) % n + _OFFSET} }};\n"

    loop_array = "{" + ",".join([str(i + _OFFSET) for i in range(n)]) + "}"

    txt = _RAW_GEOMETRY.format(
        min_xy=params.min_xy,
        max_xy=params.max_xy,
        mesh_size=params.mesh_size,
        CURVE_TAG=params.curve_tag,
        INNER_TAG=params.inner_tag,
        OUTER_TAG=params.outer_tag,
        POINTS=points,
        LINES=lines,
        LOOP_ARRAY=loop_array,
        NUMBER_OF_POINTS=n,
    )

    path_to_geo.parent.mkdir(exist_ok=True, parents=True)
    path_to_geo.write_text(txt)


def geo_to_msh(geometry_file: Path) -> Path:
    msh_file = geometry_file.parent / f"{geometry_file.stem}.msh"
    gmsh_command = f"{_GMSH_BINARY} {geometry_file} -2 -o {msh_file}"
    print(f"Running the command: '{gmsh_command}'.")
    return_code = subprocess.call(gmsh_command, shell=True, executable=_SHELL)
    if return_code != 0:
        raise Exception("Gmsh command failed.")
    print(f"Ran gmsh on {geometry_file}, generated {msh_file}.")
    return msh_file


def msh_to_pvd(msh_file: Path, inside_tag: int = None) -> None:
    mesh = Mesh(str(msh_file))
    function_space = FunctionSpace(mesh, "CG", 1)
    if inside_tag:
        indicator = utils.shape_function(function_space, mesh_tag=inside_tag)
    else:
        indicator = Function(function_space)

    pvd_file = msh_file.parent / f"{msh_file.stem}.pvd"
    File(pvd_file).write(indicator)
    print(f"Wrote {pvd_file}.")


def generate_mesh(
    params: MeshGenerationParameters,
    curve: Curve,
    base_path: Path,
) -> firedrake.Mesh:
    path = base_path / f"h={params.mesh_size}"
    geo_file = path / f"{curve.name}.geo"

    write_geo_file(params, curve, geo_file)
    msh_file = geo_to_msh(geo_file)
    msh_to_pvd(msh_file, params.curve_tag)

    return Mesh(str(msh_file))


if __name__ == "__main__":
    base_path = Path("meshes")

    for mesh_size in _MESH_RESOLUTIONS:
        for curve in CURVES:
            mesh_params = MeshGenerationParameters(mesh_size=mesh_size)
            generate_mesh(mesh_params, curve, base_path)
