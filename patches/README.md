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
