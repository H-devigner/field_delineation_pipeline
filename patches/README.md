# Patches

`delineate-anything-instance-rasters.patch` contains the Delineate-Anything changes needed by the
`instance-raster-export` branch.

Use it when the embedded `Delineate-Anything` checkout is not already on a commit that includes
instance raster export support:

```bash
cd /path/to/field_delineation_pipeline/Delineate-Anything
git am --whitespace=nowarn ../patches/delineate-anything-instance-rasters.patch
```

The patch adds `instance_raster_args` support and saves postprocessed instance-ID GeoTIFFs before
polygonization when the pipeline passes `--save-instance-rasters`.

If `git am` fails because your local `Delineate-Anything` checkout has edits or a different base
commit, abort the patch and use the tolerant installer:

```bash
cd /path/to/field_delineation_pipeline/Delineate-Anything
git am --abort
cd ..
python patches/apply_delineate_instance_rasters.py Delineate-Anything
```
