# SAM3 Optimized Point Prompt Video Annotation

This module describes the **optimized point prompt annotation workflow**
for video segmentation using **SAM3**.

Compared to the original interactive workflow, this method:

- **removes per-instance waiting time**
- allows **batch point annotation**
- runs SAM3 **offline automatically**

------------------------------------------------------------------------

## 🔥 Key Idea

Instead of:

    select point → wait for SAM3 → refine → repeat

we use:

    collect all point prompts → save to JSON → run SAM3 once

This significantly improves **annotation efficiency**.

------------------------------------------------------------------------

## Workflow Overview

The optimized pipeline consists of:

1. Select videos that need refinement\
2. Collect **point prompts for all objects (offline)**\
3. Run SAM3 to generate segmentation \
4. Render the refined video\
5. Replace original results if needed

------------------------------------------------------------------------

## Step 1 --- Select Episodes for Refinement

First, check the rendered videos from the previous annotation.
Identify episodes where: - segmentation is incorrect - objects are
missing - masks are inaccurate.

------------------------------------------------------------------------

## Step 2 --- Configure `profile.py`

Modify:

    ./annotate/config/profile.py

### 2.1 Update `scenario_objects`

Keep this **consistent with the original workflow**.

------------------------------------------------------------------------

### 2.2 Update `PROFILES`

Add a new field **refine_episode** to the PROFILES dict:

```python
PROFILES: Dict[str, ProfileSpec] = {
    "s3c2": ProfileSpec(
        scenario=3,
        cam=2,
        episode=1,
        date_dir="2025.12.03",
        refine_episode=(45,80),
    ),
}
```

This defines which episodes will be refined.

------------------------------------------------------------------------

## Step 3 --- Collect Point Prompts (Fast Annotation Mode)

Run:

``` bash
python -m video_annotate_point_fast s3c2
```

- Only the **first frame** is shown (Which can be modified in profile.py)
- No SAM3 inference is executed
- You click points for each object

### Instructions

- Follow the caption to select the correct object
- Use mouse to click:
  - positive points (object region)
  - negative points (optional)

After completing all objects:

👉 A `.json` file is generated containing all point prompts

------------------------------------------------------------------------

## Step 4 --- Run SAM3 Batch Inference

Run:

```bash
python -m video_annotate_point_fast_inference s3c2
```

For each episode: 1. Load saved point prompts 2. Apply: - text prompt -
point prompts 3. Run SAM3 propagation over the full video 4. Generate
**COCO-format annotations**

------------------------------------------------------------------------

## Step 5 --- Render Refined Video

Run:

``` bash
python -m video_render_refine s3c2
```

This will generate the **refined annotated video**.

------------------------------------------------------------------------

## Step 6 --- Validate and Replace

- Check the rendered results
- If correct:

👉 Manually replace the original annotation results

------------------------------------------------------------------------

## Recommended Usage

Use this optimized workflow when: - refining existing annotations -
labeling large datasets - improving segmentation quality efficiently
