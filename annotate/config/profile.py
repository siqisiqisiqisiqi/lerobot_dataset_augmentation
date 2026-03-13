from pathlib import Path
from typing import Dict
from typing import Dict, Tuple
from typing import Optional
from dataclasses import dataclass, field

DATA_ROOT = Path("/home/grail/training_data/real_data")

OBJ = {
    0: {
        "name": "hand",
        "prompt": "white-and-black robotic hand",  # fallback
        "prompt_by_cam": {
            2: "white-and-black robotic hand",
            3: "robotic hand with thumb",
        },
        "color_bgr": (255, 0, 0),
    },
    1: {
        "name": "bottle",
        "prompt": "bottle",
        "color_bgr": (0, 255, 0),
    },
    2: {
        "name": "pad",
        "prompt": "blue or grey circle",
        "color_bgr": (0, 0, 255),
    },
    3: {
        "name": "box",
        "prompt": "grey rectangular tray",
        "color_bgr": (255, 255, 0),
    },
    4: {
        "name": "tissue",
        "prompt": "tissue in green bag",
        "prompt_by_cam": {
            2: "tissue in green bag",
            3: "tissue in green and yellow bag",
        },
        "color_bgr": (0, 255, 0),
    },
    5: {
        "name": "tissue2",
        "prompt": "blue object with spots",
        "color_bgr": (255, 0, 255),
    },
    6: {
        "name": "orange",
        "prompt": "orange",
        "color_bgr": (0, 255, 255),
    },
    7: {
        "name": "tissue3",
        "prompt": "green tissue roll",
        "color_bgr": (0, 255, 255),
    },

}
OBJ_ID = {cfg["name"]: obj_id for obj_id, cfg in OBJ.items()}

SCENARIO_OBJECTS = {
    1: ("hand", "bottle", "pad"),
    2: ("hand", "bottle", "box"),
    3: ("hand", "tissue", "box"),
    # 3: ("hand", "tissue2", "box"),
    # 3: ("hand", "orange", "box"),
    # 3: ("tissue3",), ## 单元素 tuple 一定要带逗号
}

VIDEO_CHUNK_SIZE = 2000

@dataclass(frozen=True)
class ProfileSpec:
    scenario: int
    cam: int
    episode: int
    date_dir: Optional[str] = None
    chunk: int = 0

    prompts: Dict[int, str] = field(default_factory=dict)      # obj_id -> prompt
    init_frame: Dict[int, int] = field(default_factory=dict)   # obj_id -> frame index
    colors_bgr: Dict[int, Tuple[int,int,int]] = field(default_factory=dict)  # overrides
    
    @property
    def key(self) -> str:
        return f"s{self.scenario}c{self.cam}"
    
    @property
    def objects(self) -> Tuple[str, ...]:
        return SCENARIO_OBJECTS[self.scenario]
    
    def prompt(self, obj_id: int) -> str:
        if obj_id in self.prompts:
            return self.prompts[obj_id]
        cfg = OBJ[obj_id]
        return cfg.get("prompt_by_cam", {}).get(self.cam, cfg["prompt"])
    
    def frame(self, obj_id: int) -> int:
        return self.init_frame.get(obj_id, 0)

    def color(self, obj_id: int) -> Tuple[int,int,int]:
        return self.colors_bgr.get(obj_id, OBJ[obj_id]["color_bgr"])

PROFILES: Dict[str, ProfileSpec] = {
    "s1c2": ProfileSpec(
        scenario=1,
        cam=2,
        episode=3,
        date_dir="2026.02.25",
    ),
    "s1c3": ProfileSpec(
        scenario=1,
        cam=3,
        episode=9,
        date_dir="2026.02.25",
    ),
    "s2c2": ProfileSpec(
        scenario=2,
        cam=2,
        episode=9,
        date_dir="2025.12.02",
    ),
    "s2c3": ProfileSpec(
        scenario=2,
        cam=3,
        episode=9,
        date_dir="2025.12.02",
    ),
    "s3c2": ProfileSpec(
        scenario=3,
        cam=2,
        episode=0,
        date_dir="2025.12.03",
    ),
    "s3c3": ProfileSpec(
        scenario=3,
        cam=3,
        episode=14,
        date_dir="2025.12.03",
    ),
}