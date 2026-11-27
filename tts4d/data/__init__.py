"""Conversion of the benchmarks' released data into scene .npz files (see ``common`` for the format) with the input /
target split of the paper's evaluation protocol.  One module per dataset; ``scripts/prepare_benchmark.py`` drives them."""
from .common import assemble, interp_split, save_scene, scene_meta, uniform  # noqa: F401
