from typing import Dict, List

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.widowx.widowx import WidowX250S
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


class WidowX250SBridgeDataset(WidowX250S):
    uid = "widowx250s_bridgedataset"
    # class WidowXBridgeDatasetCameraSetupConfig(WidowXDefaultConfig):
    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used in the Bridge dataset
                pose=sapien.Pose(
                    [0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]
                ),
                width=640,
                height=480,
                entity_uid="base_link",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),  # logitech C920
            ),
        ]


class BaseBridgeEnv(BaseDigitalTwinEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    SUPPORTED_REWARD_MODES = ["none"]
    rgb_overlay_cameras = ["3rd_view_camera"]
    scene_table_height: float = 0.87
    objs: Dict[str, Actor] = dict()

    def __init__(
        self,
        obj_names: List[str],
        xyz_configs: torch.Tensor,
        quat_configs: torch.Tensor,
        **kwargs
    ):
        self.obj_names = obj_names
        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs
        super().__init__(
            robot_uids=WidowX250SBridgeDataset,
            **kwargs,
        )

    @property
    def _default_human_render_camera_configs(self):
        sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose(
                [0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]
            ),
            width=512,
            height=512,
            intrinsic=np.array(
                [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
            ),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["base_link"],
        )

    def _build_actor_helper(
        self,
        model_id: str,
        scale: float = 1,
        physical_material: PhysxMaterial = None,
        density: float = 1000,
    ):
        return super()._build_actor_helper(
            model_id,
            scale,
            physical_material,
            density,
            root_dir=ASSET_DIR / "tasks/bridge_dataset/custom",
        )

    def _load_scene(self, options: dict):
        # load background
        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_file = str(
            ASSET_DIR / "tasks/bridge_dataset/stages/bridge_table_1_v1.glb"
        )
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)
        scene_offset = np.array(
            [-2.0634, -2.8313, 0.0]
        )  # corresponds to the default offset of bridge_table_1_v1.glb
        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        for name in self.obj_names:
            self.objs[name] = self._build_actor_helper(name)

        self.xyz_configs = common.to_tensor(self.xyz_configs)
        self.quat_configs = common.to_tensor(self.quat_configs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            pos_episode_ids = torch.randint(0, len(self.xyz_configs), size=(b,))
            quat_episode_ids = torch.randint(0, len(self.quat_configs), size=(b,))
            # if self.robot_uid in ['widowx', 'widowx_bridge_dataset_camera_setup']:
            # measured values for bridge dataset
            qpos = np.array(
                [
                    -0.01840777,
                    0.0398835,
                    0.22242722,
                    -0.00460194,
                    1.36524296,
                    0.00153398,
                    0.037,
                    0.037,
                ]
            )
            self.agent.robot.set_qpos(qpos)
            # self.agent.robot.set_pose(sapien.Pose(robot_init_xyz, robot_init_rot_quat))
            # elif self.robot_uid == 'widowx_sink_camera_setup':
            #     qpos = np.array([-0.2600599, -0.12875618, 0.04461369, -0.00652761, 1.7033415, -0.26983038, 0.037,
            #                         0.037])
            robot_init_height = 0.870
            self.agent.robot.set_pose(
                sapien.Pose([0.147, 0.028, robot_init_height], q=[0, 0, 0, 1])
            )

            for i, actor in enumerate(self.objs.values()):
                xyz = self.xyz_configs[pos_episode_ids, i]
                actor.set_pose(
                    Pose.create_from_pq(p=xyz, q=self.quat_configs[quat_episode_ids, i])
                )
