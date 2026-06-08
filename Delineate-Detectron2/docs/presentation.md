# Automated Agricultural Field Boundary Delineation — Presentation

> Light, professional style. Image placeholders have descriptive titles for generation.

---

## Slide 1 — Title

**Automated Agricultural Field Boundary Delineation from Satellite Imagery**

A complete remote sensing pipeline: from data acquisition to precise field polygon maps.

> [PLACEHOLDER IMAGE: Aerial satellite view of agricultural landscape with semi-transparent vector field boundaries overlaid in green, clean professional style on white background]

---

## Slide 2 — End-to-End Pipeline

**Processing Pipeline Overview**

> [PLACEHOLDER DIAGRAM: End-to-end remote sensing processing pipeline, horizontal flow with 6 connected boxes on white background:]
>
> Satellite Fetcher (STAC API) → Mosaicing (NDVI prioritization) → Super Resolution (sen2sr, 10m → 2.5m) → LCLU Masking (Google Earth Engine) → AI Segmentation (YOLO / Mask R-CNN) → Post-Processing & Delivery

Notes:
Our pipeline consists of six stages, each solving a specific problem in the chain from raw satellite data to production-ready field boundaries. Data is retrieved through our internal Satellite Fetcher tool via the STAC API, then processed into clean cloud-free mosaics. Super resolution sharpens boundaries beyond native resolution. Land cover masks constrain processing to cropland. The AI model performs instance segmentation, and post-processing produces final vector polygons.

---

## Slide 3 — Data Acquisition

**Satellite Fetcher — STAC API Integration**

- Internal tool for retrieving Sentinel-2 imagery
- Queries via the STAC (SpatioTemporal Asset Catalog) API
- Supports region-of-interest and temporal filtering
- Downloads raw spectral bands (RGB, NIR, SWIR)

> [PLACEHOLDER IMAGE: Simplified diagram showing a satellite icon connected via an arrow labeled "STAC API" to a database icon labeled "Satellite Fetcher", outputting multiple Sentinel-2 scene tiles]

Notes:
Our Satellite Fetcher is an internal tool that interfaces with the STAC API to retrieve Sentinel-2 scenes. It handles spatial and temporal queries, downloads the required spectral bands, and prepares them for the mosaicing stage. This decouples data acquisition from processing, allowing us to update imagery sources without changing downstream code.

---

## Slide 4 — Cloudless Mosaics

**NDVI-Based Pixel Prioritization**

- Stack all available scenes for a growing season
- For each pixel, prioritize observations with high NDVI values
- Produces a cloud-free composite that favors vegetation-rich observations
- Result: clean, consistent imagery with maximum crop signal

> [PLACEHOLDER IMAGE: Before/after comparison — left panel shows a cloudy, patchy satellite image; right panel shows the same region as a pristine cloud-free mosaic, both over agricultural terrain, clean professional style]

Notes:
Cloud contamination is the primary challenge with optical satellite data. Our mosaicing pipeline stacks an entire season's scenes and applies a pixel prioritization strategy based on high NDVI values. For each pixel, we select the observation with the strongest vegetation signal. This removes clouds, haze, and shadows while preserving the peak-vegetation state of fields — maximizing contrast between adjacent parcels.

---

## Slide 5 — Super Resolution

**sen2sr — 10m to 2.5m Enhancement**

| Parameter | Value |
|---|---|
| Input bands | RGB + NIR |
| Scale factor | x4 |
| Input resolution | 10 m |
| Output resolution | 2.5 m |
| Library | sen2sr |

> [PLACEHOLDER IMAGE: Side-by-side comparison of the same agricultural area — left shows pixelated 10m resolution with blurry field edges, right shows sharp 2.5m resolution with clear boundary lines, labeled accordingly]

Notes:
At 10 meters, a single pixel covers 100 square meters. Field boundaries are often just one or two pixels wide, creating mixed pixels where adjacent fields blend together. We use the sen2sr library, which takes RGB and NIR bands and performs x4 upscaling, producing 2.5-meter resolution imagery. This gives the segmentation model crisp boundaries to trace — particularly important for smallholder plots that may be only 20-30 meters across.

---

## Slide 6 — LCLU Data

**Land Cover / Land Use from Google Earth Engine**

Downloaded from GEE — two datasets generated per year:

| Dataset | Alignment | Purpose |
|---|---|---|
| Seasonal LCLU | Natural seasons (spring, summer, autumn, winter) | Natural vegetation cycles |
| Agricultural LCLU | Human agricultural calendar (irrigation-driven) | Crop-specific phenology |

**9 classes used for masking:**

| Role | Classes |
|---|---|
| Target (cropland) | Crops (4) |
| Clip (hard edges) | Water (0), Built area (6), Snow/Ice (8) |
| Background fill | Trees (1), Grass (2), Flooded veg. (3), Shrub (5), Bare ground (7) |

> [PLACEHOLDER IMAGE: Map showing an LCLU classification layer over agricultural terrain, with distinct colors for each land cover class, legend on the side, professional cartographic style]

Notes:
Our LCLU data comes from Google Earth Engine. We generate two datasets per year — one aligned with natural seasonality and one aligned with the agricultural calendar, which is mainly driven by irrigation cycles. This dual approach captures both natural vegetation dynamics and human-managed crop patterns. The 9-class map serves three roles in our pipeline: cropland identifies target areas, water/built/snow create hard boundary clip masks, and other vegetation classes fill background between fields.

---

## Slide 7 — Model Architectures

**Two architectures, two approaches**

### YOLO — Single-Stage Detection (Ultralytics)
- Processes the entire image in a single forward pass
- Fast inference, optimized for speed
- Predicts bounding boxes + class probabilities directly
- Well-suited for rapid detection tasks

### Mask R-CNN — Two-Stage Segmentation (Detectron2)
- Stage 1: Region Proposal Network generates candidate regions
- Stage 2: Classification head + mask head per proposal
- Produces both bounding boxes and per-instance segmentation masks
- Better suited for precise boundary delineation

> [PLACEHOLDER DIAGRAM: Side-by-side conceptual architecture comparison — left shows YOLO single-pass flow (image → grid → detections), right shows Mask R-CNN two-stage flow (image → proposals → classification + masks), clean schematic style]

Notes:
We used two model architectures at different stages of the project. YOLO, implemented through Ultralytics, is a single-stage detector — it processes the entire image in one pass, making it fast but limited to bounding boxes. Mask R-CNN, implemented with Detectron2, is a two-stage architecture: first it proposes candidate regions, then it classifies each and generates a pixel-level segmentation mask. This second approach is what gives us precise field boundary delineation rather than just rectangular boxes around fields.

---

## Slide 8 — Phase 1: Object Detection with YOLO

**Initial approach using YOLO (Ultralytics)**

| Parameter | Value |
|---|---|
| Architecture | YOLO (Ultralytics) |
| Task | Object detection (bounding boxes) |
| Training dataset | FBIS-22M (22M+ field instances) |
| Output | Bounding boxes around fields |

- Fast inference, suitable for initial field localization
- Large-scale pre-training on diverse global agriculture
- Limitation: bounding boxes do not capture precise field shapes

> [PLACEHOLDER IMAGE: Satellite image of agricultural fields with rectangular bounding boxes drawn around each field, showing detection output of YOLO model]

Notes:
Our first approach used YOLO through the Ultralytics framework. We leveraged the FBIS-22M dataset — over 22 million field boundary instances from diverse agricultural regions worldwide. YOLO provides fast detection with bounding boxes, which is effective for field localization. However, bounding boxes are fundamentally rectangular, while agricultural fields have irregular shapes. This motivated our transition to instance segmentation.

---

## Slide 9 — Phase 2: Instance Segmentation with Mask R-CNN

**Fine-tuned model using Mask R-CNN (Detectron2)**

| Parameter | Value |
|---|---|
| Architecture | Mask R-CNN (ResNet-101 + FPN) |
| Framework | Detectron2 |
| Task | Instance segmentation (per-field masks) |
| Training dataset | Custom dataset (Roboflow) |
| Pre-training | COCO ImageNet weights |
| Experiment tracking | MLflow |

- Produces pixel-level segmentation masks per field
- Fine-tuned on our own annotated agricultural data
- Full control over model behavior and training parameters

> [PLACEHOLDER IMAGE: Satellite image of agricultural fields with precise polygon segmentation masks colored differently for each field instance, showing Mask R-CNN output]

Notes:
To achieve precise boundary delineation, we fine-tuned a Mask R-CNN model using Detectron2. The backbone is ResNet-101 with a Feature Pyramid Network for multi-scale detection. We prepared our own training dataset using Roboflow, with pixel-level segmentation annotations. Starting from COCO-pretrained weights, we fine-tuned specifically on agricultural field data. MLflow tracks all training experiments — loss curves, validation mAP, and model artifacts — allowing us to compare runs and select the best model.

---

## Slide 10 — GPU-Accelerated Inference

**Parallel processing on 8x NVIDIA H100**

| Optimization | Technique |
|---|---|
| Parallelism | All 8 GPUs process tiles simultaneously |
| Precision | BF16 (approximately 2x throughput) |
| Data loading | Asynchronous prefetch with GDAL caching |
| Batch size | 64 tiles per batch |

> [PLACEHOLDER DIAGRAM: Simplified diagram showing a large satellite image being split into a grid of tiles, with arrows dispatching tiles to 8 GPU chips in parallel, outputs merging into a unified result]

Notes:
For production inference, we deploy on 8 NVIDIA H100 GPUs with 80 GB each. The satellite image is split into tiles, which are dispatched across all GPUs in parallel. BF16 precision nearly doubles throughput on H100 hardware without measurable quality loss. Asynchronous data prefetching ensures GPUs are never waiting for data. This setup processes large Sentinel-2 scenes efficiently, with post-processing and polygonization running on parallel CPU workers.

---

## Slide 11 — Post-Processing

**From raw predictions to clean vector boundaries**

| Stage | Purpose |
|---|---|
| Cross-tile merging | Stitch predictions at tile boundaries |
| Overlap dissolve | Merge overlapping polygons |
| Area filtering | Remove false positives below threshold |
| Hole removal | Fill small interior gaps |
| LCLU masking | Tag non-cropland polygons |
| Simplification | Smooth jagged polygon edges |
| Statistics | Compute area, perimeter, compactness |

> [PLACEHOLDER IMAGE: Before/after comparison — left shows noisy, overlapping, jagged raw prediction polygons; right shows clean, smooth, non-overlapping field boundaries after post-processing]

Notes:
Raw model output requires significant cleanup. Tile boundaries create artificial polygon edges. Small false positives appear as noise. Adjacent predictions may overlap. Our post-processing pipeline addresses each issue systematically: merging across tile boundaries, dissolving overlaps, filtering by area, removing small holes, applying LCLU masks, smoothing geometry with Douglas-Peucker simplification, and computing accurate statistics in an equal-area projection.

---

## Slide 12 — Real Results

**Detection and segmentation outputs on agricultural parcels**

> [PLACEHOLDER IMAGE: Grid of 4 real-world examples showing the pipeline results:
> Top-left: Large-scale industrial agriculture with clean boundary detection
> Top-right: Medium-sized mixed farming with varied field shapes
> Bottom-left: Super-resolution before/after comparison (10m vs 2.5m)
> Bottom-right: LCLU mask overlay showing crop vs non-crop classification]

| Example | Region type | Result quality |
|---|---|---|
| Industrial fields | Large rectangular (>10 ha) | High precision boundaries |
| Mixed farming | Medium varied shapes | Good detection, minor edge noise |
| Smallholder | Small irregular (<0.5 ha) | Challenging, SR helps significantly |

Notes:
Here are representative results from our pipeline. Industrial agricultural regions with large, well-defined fields achieve high-precision boundaries. Medium-scale mixed farming shows good detection with minor edge refinement needed. The super resolution comparison demonstrates the significant improvement from 10m to 2.5m — boundaries that were blurred and ambiguous become sharp and traceable. The LCLU overlay shows how masking effectively constrains detection to agricultural areas.

---

## Slide 13 — Limitations and Scale Challenge

**Resolution constraints at different agricultural scales**

| | Industrial Agriculture | Smallholder Plots |
|---|---|---|
| Typical size | 50+ hectares | < 0.5 hectares |
| Field width | > 500 meters | 20-30 meters |
| At 10m resolution | 50+ pixels across | 2-3 pixels across |
| Detection quality | High precision | Challenging |

**Mitigation strategies:**
- Super resolution (10m to 2.5m) via sen2sr
- Region-specific fine-tuning on local annotated data
- Higher-resolution imagery sources (Planet 3m, drone)

> [PLACEHOLDER IMAGE: Split comparison — left shows large industrial fields in a temperate region with precise green boundary lines; right shows dense smallholder plots in a tropical region with pixelated, uncertain boundaries at 10m resolution]

Notes:
The fundamental limitation is pixel resolution. At 10 meters, industrial fields spanning hundreds of meters are straightforward — 50 or more pixels across, clear boundaries. But smallholder plots common in sub-Saharan Africa and Southeast Asia may be only 20-30 meters wide — just 2-3 pixels. At this scale, mixed pixels blur boundaries and adjacent plots are indistinguishable. Super resolution helps significantly, improving to 2.5 meters gives 8-12 pixels across the same field. Region-specific fine-tuning and higher-resolution imagery are complementary solutions.

---

## Slide 14 — Next Steps

**Planned improvements and extensions**

| Timeframe | Priority |
|---|---|
| Short-term | Expand training data across regions and seasons |
| Short-term | Integrate sen2sr into the automated pipeline |
| Short-term | Benchmark against published results |
| Mid-term | Temporal field tracking across multiple seasons |
| Mid-term | Country-scale deployment |
| Long-term | Streaming pipeline for new satellite acquisitions |
| Long-term | Web-based boundary visualization interface |

> [PLACEHOLDER DIAGRAM: Roadmap timeline showing the progression from current state through short-term, mid-term, and long-term milestones, clean horizontal layout]

Notes:
Our immediate priorities are expanding the training dataset to cover more agricultural regions and seasons, integrating super resolution directly into the pipeline, and benchmarking our mAP scores against published results. In the medium term, we plan temporal analysis — tracking how field boundaries change across seasons — and scaling to country-level deployment. Longer-term, we envision a streaming pipeline that automatically processes new Sentinel-2 acquisitions as they become available, with a web interface for visualization and validation.
