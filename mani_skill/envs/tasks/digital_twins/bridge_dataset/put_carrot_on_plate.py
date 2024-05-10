import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.envs.tasks.digital_twins.bridge_dataset.base_env import BaseBridgeEnv
from mani_skill.utils.registration import register_env


@register_env("PutCarrotOnPlateInScene-v0", max_episode_steps=60)
class PutCarrotOnPlateInScene(BaseBridgeEnv):
    rgb_overlay_path = str(
        ASSET_DIR / "tasks/bridge_dataset/real_inpainting/bridge_real_eval_1.png"
    )

    def __init__(self, **kwargs):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.887529),
                                np.append(grid_pos_2, 0.869532),
                            ]
                        )
                    )
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        quat_configs = torch.tensor(
            [
                np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
            ]
        )
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put carrot on plate"
