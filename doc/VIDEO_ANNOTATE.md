
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
Set episode you want to experiment with in PROFILES in profile.py first.

```bash
python -m annotate.video_annotate s1c2
```

* Render **one** video for experiment:
This is based on the result of video_annotate.py. The following command automatically uses the *coco.json file generated above.

```bash
python -m annotate.video_render s1c2
```

or run the following commands together

```bash
python -m annotate.video_annotate s3c3 && python -m annotate.video_render s3c3
```

* Iteratively annotate/render all videos
Set a certain data set (e.g. s1c2) run the following command.

```bash
./run_all_video.sh
```

## 2. Config Setting

### 2.1 Scenario Camera Profile Setting

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

### 2.2 Object Configuration

Objects are centrally defined:

```python
OBJ = {
    "0":   {"name": "hand", "prompt": "...", "color_bgr": (...)},
    "1": {"name": "bottle", "prompt": "...", "color_bgr": (...)},
    "..."
}
```

OBJ_ID is derived from OBJ, a reverse mapping from name to id.

Specifically, prompts can be based on camera or fall back to default:

```python
"prompt": "white-and-black robotic hand",  # fallback
"prompt_by_cam": {
    2: "white-and-black robotic hand",
    3: "robotic hand with thumb",
},
```

### 2.3 Scenario Object Sets

```python
SCENARIO_OBJECTS = {
    1: ("hand", "bottle", "pad"),
    2: ("hand", "bottle", "box"),
    "..."
}
```

Each scenario automatically uses its corresponding object set.

### 2.4 Defualt Test Setting

 For one experiment with video_annotate.py/video_render.py, we use the default setting in PROFILES in profile.py.

 ```python
 PROFILES: Dict[str, ProfileSpec] = {
    "s1c2": ProfileSpec(
        scenario=1,
        cam=2,
        episode=3,
        date_dir="2026.02.25",
    ),
    "..."
}
 ```

---

## 3. Object Render Tuning

When you render the object from the COCO dataset, you can select which object to render (Example: exclude hand).

Inside video_read_coco.py script:

```python
KEEP_NAMES = ("bottle", "pad")
```

Note: you can also modify the render style in **./annotate/video_render.py** files, which can influence the VLA performance.

---
