#!/usr/bin/env python3
"""Install instance-raster export support into a Delineate-Anything checkout.

This is a tolerant fallback for environments where the commit patch cannot be
applied because the embedded Delineate-Anything checkout has local edits or a
different base commit.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SAVE_INSTANCE_RASTER_METHOD = """\
    def save_instance_raster(self, output_path, geotransform):
        t0 = time.time()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        driver = gdal.GetDriverByName("GTiff")
        options = [
            "BIGTIFF=IF_SAFER",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "TILED=YES",
            "NUM_THREADS=ALL_CPUS",
        ]
        height, width = self.instances_map.shape
        dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Int32, options)
        if dataset is None:
            raise RuntimeError(f"Could not create instance raster: {output_path}")

        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(self.srs_wkt)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.SetDescription("instance_id")
        band.WriteArray(self.instances_map)
        band.FlushCache()
        dataset.FlushCache()
        dataset = None

        logger.info(f"Instance raster saved to {output_path} in {time.time() - t0:.2f} s.")

"""


GET_INSTANCE_RASTER_PATH_FUNC = """\
def get_instance_raster_path(config, gpkg_path, region_counter, current_region, num_regions):
    instance_config = config.get("instance_raster_args", {})
    if not instance_config.get("save", False):
        return None

    tile_name = os.path.splitext(os.path.basename(gpkg_path))[0]
    output_root = instance_config.get("output_root")
    if output_root is None:
        output_root = os.path.join(os.path.dirname(gpkg_path), "instance_rasters")

    tile_output_dir = os.path.join(output_root, tile_name)
    os.makedirs(tile_output_dir, exist_ok=True)

    always_region_suffix = instance_config.get("always_region_suffix", False)
    if num_regions == 1 and not always_region_suffix:
        filename = f"{tile_name}.instances.tif"
    else:
        filename = (
            f"{tile_name}.region_{region_counter:04d}"
            f"_x{current_region[0]}_y{current_region[1]}.instances.tif"
        )
    return os.path.join(tile_output_dir, filename)


"""


INSTANCE_RASTER_CALL = """\
            instance_raster_path = get_instance_raster_path(
                full_config,
                layer_info[0],
                region_counter,
                planner.current_region,
                num_regions,
            )
            if instance_raster_path is not None:
                postproc_handler.save_instance_raster(instance_raster_path, planner.get_geotransform())

"""


INSTANCE_RASTER_CONFIG = """\
instance_raster_args:
  # Save the postprocessed instance-ID raster that is used for polygonization.
  # Positive values are field instance IDs, negative values are background IDs, and 0 is nodata/background.
  save: false
  output_root: null
  always_region_suffix: false

"""


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def write_if_changed(path: Path, old: str, new: str) -> bool:
    if old == new:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def patch_postproc(path: Path) -> bool:
    text = read(path)
    new = text

    if not re.search(r"^import os$", new, flags=re.MULTILINE):
        new = re.sub(r"^import numpy as np$", "import os\nimport numpy as np", new, count=1, flags=re.MULTILINE)

    if "from osgeo import gdal" not in new:
        new = re.sub(r"^from osgeo import ogr$", "from osgeo import gdal, ogr", new, count=1, flags=re.MULTILINE)

    if "def save_instance_raster(" not in new:
        marker = "\n    def dispose(self):"
        if marker not in new:
            raise RuntimeError(f"Could not find dispose() insertion point in {path}")
        new = new.replace(marker, "\n" + SAVE_INSTANCE_RASTER_METHOD + marker, 1)

    return write_if_changed(path, text, new)


def patch_inference(path: Path) -> bool:
    text = read(path)
    new = text

    if "def get_instance_raster_path(" not in new:
        marker = "\ndef execute_delineation("
        if marker not in new:
            raise RuntimeError(f"Could not find execute_delineation() insertion point in {path}")
        new = new.replace(marker, "\n" + GET_INSTANCE_RASTER_PATH_FUNC + marker, 1)

    if "postproc_handler.save_instance_raster(" not in new:
        pattern = (
            r"(?P<apply>^(?P<indent>[ \t]+)postproc_handler\.apply_background\(background\)\n)"
            r"(?P<blank>[ \t]*\n)?"
            r"(?P<poly>[ \t]+postproc_handler\.polygonize\(planner\.get_geotransform\(\), layer_info\))"
        )
        match = re.search(pattern, new, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Could not find apply_background()/polygonize() block in {path}")
        replacement = match.group("apply") + "\n" + INSTANCE_RASTER_CALL + match.group("poly")
        new = new[: match.start()] + replacement + new[match.end() :]

    if "num_regions = planner.get_num_regions()" not in new:
        new = new.replace(
            "    dataloader = None\n    with tqdm(total=planner.get_num_regions(),",
            "    dataloader = None\n    num_regions = planner.get_num_regions()\n    with tqdm(total=num_regions,",
            1,
        )

    return write_if_changed(path, text, new)


def patch_conf(path: Path) -> bool:
    text = read(path)
    new = text
    if "instance_raster_args:" not in new:
        marker = "\nfiltering_args:"
        if marker not in new:
            raise RuntimeError(f"Could not find filtering_args insertion point in {path}")
        new = new.replace(marker, "\n" + INSTANCE_RASTER_CONFIG + marker.lstrip("\n"), 1)
    return write_if_changed(path, text, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "delineate_root",
        nargs="?",
        default="Delineate-Anything",
        type=Path,
        help="Path to the Delineate-Anything checkout. Defaults to ./Delineate-Anything.",
    )
    args = parser.parse_args()

    root = args.delineate_root.resolve()
    if (root / ".git" / "rebase-apply").exists():
        raise SystemExit(f"git am is still active in {root}. Run: cd {root} && git am --abort")

    targets = [
        root / "methods" / "main" / "PostprocHandler.py",
        root / "methods" / "main" / "inference.py",
        root / "conf_sample.yaml",
    ]
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(target)

    changed = {
        "PostprocHandler.py": patch_postproc(targets[0]),
        "inference.py": patch_inference(targets[1]),
        "conf_sample.yaml": patch_conf(targets[2]),
    }

    for name, did_change in changed.items():
        print(f"{name}: {'updated' if did_change else 'already patched'}")
    print("Instance raster export support is installed.")


if __name__ == "__main__":
    main()
