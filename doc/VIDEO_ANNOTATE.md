
# SAM3 Video Annotation & Rendering

This module provides a structured pipeline for:

1. Extracting object masks from videos using SAM3
2. Saving masks in COCO format
3. Rendering annotated videos
4. Re-rendering videos from saved COCO files with object filtering

All the code is in the **./annotate** folder.

---

## 1. Recommended Workflow

* Annotate **one** video for experiment:
Set episode you want to experiment with in PROFILES first.

```bash
python ./annotate/video_annotate.py s1c2
```

* Render **one** video for experiment:
This is based on the result(s) of video_annotate.py. Make sure you've completed the first step.
The default render tag is "render"

```bash
python ./annotate/video_read_coco.py s1c2
python ./annotate/video_read_coco.py s1c2 --render_tag no_hand
```

* Iteratively annotate/render all videos
For a certain data set in PROFILE (e.g. s1c2), modify PY_SCRIPT as video_annotate.py first, and run the following command.

```bash
./run_all_video.sh
```

Modify **PY_SCRIPT** as video_read_coco.py, and re-run the above command to render the video with custom requirements.

## 2. Basic Introduction

The workflow is divided into three files:

### 2.1 `video_annotate.py`

* Runs SAM3 on a selected scenario/camera profile
* Extracts masks per object
* Saves results as COCO JSON
* renders annotated video with all masks (for verification)

### 2.2 `video_read_coco.py`

* Reads existing COCO JSON
* Selectively renders chosen objects
* Outputs rendered video to a custom folder

### 2.3 `coco_io.py`

* Converts `outputs_merged` (frame-indexed masks, boxes, object IDs) into a minimal COCO JSON file
* Encodes masks as RLE and converts normalized boxes to COCO pixel format
* Preserves tracking IDs via a custom `track_id` field
* Supports loading COCO JSON back into `outputs_merged` format for re-rendering or post-processing

---

## 3. Config Setting

### 3.1 Scenario Camera Profile Setting

Each run is defined by a **profile key**:

``` bash
s{scenario}c{cam}
```

Examples:

``` bash
s1c2
s1c3
s2c2
s2c3
```

Profiles automatically determine:

* video path
* output directory
* object set per scenario
* default prompts
* default colors

See the global configuration section at the top of `video_annotate.py` for details.

### 3.2 Object Configuration

Objects are centrally defined:

```python
OBJ = {
    "hand":   {"id": 0, "prompt": "...", "color_bgr": (...)},
    "bottle": {"id": 1, "prompt": "...", "color_bgr": (...)},
    "pad":    {"id": 2, "prompt": "...", "color_bgr": (...)},
    "box":    {"id": 3, "prompt": "...", "color_bgr": (...)},
}
```

Derived mappings:

* `OBJ_ID`
* `OBJ_NAME`
* `DEFAULT_PROMPT_BY_ID`
* `DEFAULT_COLOR_BY_ID`

### 3.3 Scenario Object Sets

```python
SCENARIO_OBJECTS = {
    1: ("hand", "bottle", "pad"),
    2: ("hand", "bottle", "box"),
}
```

Each scenario automatically uses its corresponding object set.

---

## 4. Parameter Tunning

### 4.1 Annotation Parameter Tuning

Change initialization frame for bottle:

```bash
python ./annotate/video_annotate.py s1c2 --frame_idx_bottle 60
```

Change object prompt:

```bash
python ./annotate/video_annotate.py s1c2 --hand_prompt "robotic hand with thumb"
```

Prompts, the initial frame and color of all objects can be override.

---

### 4.2 Selected Object Rendering

When you render the object from the COCO dataset, you can select which object to render (Example: exclude hand).

Inside video_read_coco.py script:

```python
KEEP_NAMES = ("bottle", "pad")
```

### 4.3 Batch Processing with Shell Script

For large-scale processing across multiple date directories, a unified shell script is provided.

This script:

* Automatically reads default configuration from `video_annotate.py`
* Scans all date directories (e.g., `2025.12.08`, `2026.01.05`, etc.)
* Iterates through all `.mp4` files
* Runs annotation and rendering per video
* Preserves date structure in the output directory

---
