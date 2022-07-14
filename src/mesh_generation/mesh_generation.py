from dataclasses import dataclass
from pathlib import Path
import subprocess

from firedrake import *

from src.curves import Curve
import src.utils as utils

from src.mesh_generation.geometry import *


_GMSH_BINARY = "gmsh"
_SHELL = '/bin/zsh'

_CURVE_TAG = 10
_INNER_TAG = 6
_OUTER_TAG = 7


@dataclass
class MeshGenerationParameters:
    mesh_size: float
    min_xy: int = -10
    max_xy: int = 10


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
        CURVE_TAG=_CURVE_TAG,
        INNER_TAG=_INNER_TAG,
        OUTER_TAG=_OUTER_TAG,
        POINTS=points,
        LINES=lines,
        LOOP_ARRAY=loop_array,
        NUMBER_OF_POINTS=n,
    )

    path_to_geo.parent.mkdir(exist_ok=True, parents=True)
    path_to_geo.write_text(txt)


def geo_to_msh(geometry_file: Path, overwrite: bool = False) -> Path:
    msh_file = geometry_file.parent / f"{geometry_file.stem}.msh"
    if msh_file.exists() or overwrite:
        return msh_file

    gmsh_command = f"{_GMSH_BINARY} {geometry_file} -2 -o {msh_file}"
    print(f"Running the command: '{gmsh_command}'.")
    return_code = subprocess.call(gmsh_command, shell=True, executable=_SHELL)
    if return_code != 0:
        raise Exception("Gmsh command failed.")
    print(f"Ran gmsh on {geometry_file}, generated {msh_file}.")
    return msh_file


def msh_to_pvd(msh_file: Path, overwrite: bool = False) -> None:
    pvd_file = msh_file.parent / f"{msh_file.stem}.pvd"
    if pvd_file.exists() or overwrite:
        return
    mesh = Mesh(str(msh_file))
    indicator = utils.shape_function(mesh, mesh_tag=_CURVE_TAG)

    File(pvd_file).write(indicator)
    print(f"Wrote {pvd_file}.")


def generate_mesh(
    params: MeshGenerationParameters,
    curve: Curve,
    base_path: Path,
) -> Path:
    path = base_path / f"h={params.mesh_size}"
    geo_file = path / f"{curve.name}.geo"

    write_geo_file(params, curve, geo_file)
    msh_file = geo_to_msh(geo_file)
    msh_to_pvd(msh_file)

    return msh_file
