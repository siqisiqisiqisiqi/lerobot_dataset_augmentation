# Lerobot Dataset Augmentation

## Getting Started

### Python Virtual Environment

1. Create a new Conda environment:

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    ```

2. Install dependencies:

    ```bash
    uv pip install -r requirements.txt
    ```

3. Install the SAM3 package:

    ```bash
    uv pip install -e .
    ```

### HuggingFace Login

You must be authenticated with a Hugging Face account to interact with the Hub. [Create an
account](https://huggingface.co/join) if you don’t already have one, and then sign in to get your [User Access
Token](https://huggingface.co/docs/hub/security-tokens) from your [Settings
page](https://huggingface.co/settings/tokens). The User Access Token is used to authenticate your identity to the Hub.

1. Obtain the permission to access SAM3 model on HuggingFace.

2. Authenticate your computer:

    ```bash
    hf auth login
    ```

## Usage

### Annotate Videos

#### Annotate One Video

> Check the detailed steps listed [here](./doc/1-VIDEO_ANNOTATE.md).

Modify parameters in the file `./annotate/config/profile.py` followed by running the below command to annotate a
specific case.

```bash
python -m annotate.video_annotate s1c2
```

Once the annotation was saved, the following script can be executed to render objects to the video.

```bash
python -m annotate.video_render s1c2
```

#### Annotate Video Batch

Modify the parameter `PROFILE` in `run_all_video.sh` and run the following command:

```bash
./run_all_video.sh
```

### Merge Datasets

Annotated datasets should be augmented to the original dataset. A script is helpful to do that.

> Check the detailed steps listed [here](./doc/2-LEROBOT_DATASET_AUG.md).

Modify parameters `DATA_ROOT` and `SCENARIO` in `dataset_mod.sh` and run the following command:

```bash
./dataset_mod.sh
```
