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
        "prompt": "small blue circular coaster on the table",
        "color_bgr": (0, 0, 255),
    },
    3: {
        "name": "box",
        "prompt": "grey object on the table",
        "prompt_by_cam": {
            2: "grey rectangular tray",
            3: "light gray basket on the table",
        },
        "color_bgr": (255, 255, 0),
    },
    4: {
        "name": "glass bottle",
        "prompt": "transparent bottle",
        "color_bgr": (0, 255, 0),
    },
    5: {
        "name": "cup",
        "prompt": "cup",
        "color_bgr": (0, 255, 255),
    },
    6: {
        "name": "blue cup",
        "prompt": "blue cylinder",
        "color_bgr": (0, 255, 0),
    },
    7: {
        "name": "red cup",
        "prompt": "red cylinder",
        "color_bgr": (255, 0, 0),
    },
    8: {
        "name": "cup holder",
        "prompt": "black holder with red and blue twigs",
        "color_bgr": (255, 255, 0),
    },

}
OBJ_ID = {cfg["name"]: obj_id for obj_id, cfg in OBJ.items()}

SCENARIO_OBJECTS = {
    1: ("hand", "bottle", "pad"),
    2: ("hand", "bottle", "box"),
    3: ("hand", "tissue", "box"),
    4: ("cup","glass bottle"),
    7: ("bottle", "pad", "box"),
    8: ("bottle", "pad", "box"),
    9: ("bottle", "pad"),
    10: ("bottle", "box"),
    20: ("blue cup", "cup holder"),
    21: ("red cup", "cup holder"),
    22: ("blue cup", "red cup", "cup holder"),
    23: ("blue cup", "red cup", "cup holder"),
}

VIDEO_CHUNK_SIZE = 1000

@dataclass(frozen=True)
class ProfileSpec:
    scenario: int
    cam: int
    episode: int
    date_dir: Optional[str] = None
    refine_episode: tuple = ()
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
        episode=4,
        date_dir="2025.12.01",
    ),
    "s1c3": ProfileSpec(
        scenario=1,
        cam=3,
        episode=9,
        date_dir="2025.12.01",
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
        episode=1,
        date_dir="2025.12.03",
        refine_episode=(45),
    ),
    "s3c3": ProfileSpec(
        scenario=3,
        cam=3,
        episode=41,
        date_dir="2026.04.17",
    ),
    "s4c2": ProfileSpec(
        scenario=4,
        cam=2,
        episode=20,
        date_dir="2026.01.30",
    ),
    "s7c2": ProfileSpec(
        scenario=7,
        cam=2,
        episode=12,
        date_dir="2026.07.07",
    ),
    "s7c3": ProfileSpec(
        scenario=7,
        cam=3,
        episode=12,
        date_dir="2026.07.07",
    ),
    "s8c2": ProfileSpec(
        scenario=8,
        cam=2,
        episode=12,
        date_dir="2026.07.08",
    ),
    "s8c3": ProfileSpec(
        scenario=8,
        cam=3,
        episode=12,
        date_dir="2026.07.08",
    ),
    "s9c2": ProfileSpec(
        scenario=9,
        cam=2,
        episode=12,
        date_dir="2026.07.07",
    ),
    "s9c3": ProfileSpec(
        scenario=9,
        cam=3,
        episode=12,
        date_dir="2026.07.07",
    ),
    "s10c2": ProfileSpec(
        scenario=10,
        cam=2,
        episode=12,
        date_dir="2026.07.09",
    ),
    "s10c3": ProfileSpec(
        scenario=10,
        cam=3,
        episode=12,
        date_dir="2026.07.09",
    ),
    "s20c2": ProfileSpec(
        scenario=20,
        cam=2,
        episode=19,
        date_dir="2026.07.17",
    ),
    "s21c2": ProfileSpec(
        scenario=21,
        cam=2,
        episode=19,
        date_dir="2026.07.17",
    ),
    "s22c2": ProfileSpec(
        scenario=22,
        cam=2,
        episode=25,
        date_dir="2026.07.16",
    ),
    "s23c2": ProfileSpec(
        scenario=23,
        cam=2,
        episode=25,
        date_dir="2026.07.16",
    ),
}