# LeRobot Dataset Augmentation Guide

This document describes the complete procedure for performing **video augmentation** and **text augmentation** for the LeRobot dataset. After completing these steps, the augmented dataset can be transferred to the training server.

---

## Environment Setup

Run the following command to install the required packages

'''bash
pip install openai pyarrow
'''

---

## Dataset Structure

The LeRobot dataset contains four metadata files:

* `episodes.json`
* `tasks.json`
* `info.json`
* `stats.json`

These files must be updated consistently after augmentation.

---

## Dataset Augmentation

Run:

```bash
./dataset_mod.sh
```

## Transfer Dataset to Training Server

After augmentation, transfer the dataset using `rsync`.

Example command:

```bash
rsync -av --progress \
/home/grail/training_data/real_data/augment/data/scenario_1_cam2_cam3 \
zoyi@100.121.248.2:/home/zoyi/Projects/manipulation/training_data/lerobot/T170_augment/
```

## Dataset Configuration Parameters Explanation

### SRC_DIR

`SRC_DIR="/home/grail/training_data/real_data/scenario_1"`

This is the directory containing the original dataset that will be used as the source for augmentation and processing.

---

### DEST_DIR

`DEST_DIR="./data/scenario_1_cam2_cam3_mod"`

This is the directory where the newly generated augmented dataset and modified files will be saved.

---

### TASK_TEXT

`TASK_TEXT="Place the bottle on the pad."`

This is the base task instruction text that describes the robot action and will be used for annotation and text augmentation.

---

### TEXT_VARIANT_NUM

`TEXT_VARIANT_NUM=9`

This specifies the number of alternative text variants to generate from the base task instruction for language augmentation.

---

### CAM2_SRC & CAM3_SRC

These are the directories containing the annotated data recorded from Camera 2 for multi-view dataset integration.

---
