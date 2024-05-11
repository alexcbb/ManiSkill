from .sequential_task import SequentialTaskEnv
from .planner import (
    TaskPlan,
    Subtask,
    PickSubtask,
    SubtaskConfig,
    PickSubtaskConfig,
)

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
import sapien
import sapien.physx as physx

import torch
import torch.random
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from tqdm import tqdm
from functools import cached_property
import itertools
from typing import Any, Dict, List, Tuple


PICK_OBS_EXTRA_KEYS = [
    "tcp_pose_wrt_base",
    "obj_pose_wrt_base",
    "is_grasped",
]


@register_env("PickSequentialTask-v0", max_episode_steps=200)
class PickSequentialTaskEnv(SequentialTaskEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    # TODO (arth): add locomotion, open fridge, close fridge
    pick_cfg = PickSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
    )
    place_cfg = None

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        # spawn randomization
        randomize_arm=False,
        randomize_base=False,
        randomize_loc=False,
        # additional spawn randomization, shouldn't need to change
        spawn_loc_radius=2,
        # colliison tracking
        robot_force_mult=0,
        robot_force_penalty_min=0,
        robot_cumulative_force_limit=torch.inf,
        # additional randomization
        obj_randomization=False,
        **kwargs,
    ):

        # NOTE (arth): task plan length and order checking left to SequentialTaskEnv
        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], PickSubtask
        ), "Task plans for Pick training must be one PickSubtask long"

        # randomization vals
        self.randomize_arm = randomize_arm
        self.randomize_base = randomize_base
        self.randomize_loc = randomize_loc
        self.spawn_loc_radius = spawn_loc_radius

        # force reward hparams
        self.robot_force_mult = robot_force_mult
        self.robot_force_penalty_min = robot_force_penalty_min
        self.robot_cumulative_force_limit = robot_cumulative_force_limit

        # additional randomization
        self.obj_randomization = obj_randomization

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # COLLISION TRACKING
    # -------------------------------------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
        return super().reset(*args, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT RANDOMIZATION
    # -------------------------------------------------------------------------------------------------
    # TODO (arth): maybe check that obj won't fall when noise is added
    # -------------------------------------------------------------------------------------------------

    def _after_reconfigure(self, options):
        force_rew_ignore_links = [
            self.agent.finger1_link,
            self.agent.finger2_link,
            self.agent.tcp,
        ]
        self.force_articulation_link_ids = [
            link.name
            for link in self.agent.robot.get_links()
            if link not in force_rew_ignore_links
        ]

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            original_env_idx = env_idx.clone()

            super()._initialize_episode(env_idx, options)
            b = len(env_idx)

            if self.obj_randomization:
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
                xyz += self.subtask_objs[0].pose.p
                xyz[..., 2] += 0.005

                qs = quaternion_raw_multiply(
                    randomization.random_quaternions(
                        b, lock_x=True, lock_y=True, lock_z=False
                    ),
                    self.subtask_objs[0].pose.q,
                )
                self.subtask_objs[0].set_pose(Pose.create_from_pq(xyz, qs))

            robot_init_p, robot_init_q, robot_init_qpos = (
                self.agent.robot.pose.p.clone(),
                self.agent.robot.pose.q.clone(),
                self.agent.robot.qpos.clone(),
            )
            # keep going until no collisions
            while True:

                centers = self.subtask_objs[0].pose.p[env_idx, :2]
                navigable_positions = []
                for env_num, center in zip(env_idx, centers):
                    positions = torch.tensor(
                        self.scene_builder.navigable_positions[env_num]
                    )
                    navigable_positions.append(
                        positions[
                            torch.norm(positions - center, dim=1)
                            <= self.spawn_loc_radius
                        ]
                    )
                num_navigable_positions = torch.tensor(
                    [len(positions) for positions in navigable_positions]
                ).int()
                navigable_positions = pad_sequence(
                    navigable_positions, batch_first=True, padding_value=0
                ).float()

                positions_wrt_centers = navigable_positions - centers.unsqueeze(1)
                dists = torch.norm(positions_wrt_centers, dim=-1)

                rots = (
                    torch.sign(positions_wrt_centers[..., 1])
                    * torch.arccos(positions_wrt_centers[..., 0] / dists)
                    + torch.pi
                )
                rots %= 2 * torch.pi

                self.resting_qpos = torch.tensor(self.agent.RESTING_QPOS[3:-2])

                # NOTE (arth): it is assumed that scene builder spawns agent with some qpos
                qpos = self.agent.robot.get_qpos()

                if self.randomize_loc:
                    low = torch.zeros_like(num_navigable_positions)
                    high = num_navigable_positions
                    size = env_idx.size()
                    idxs: List[int] = (
                        (torch.randint(2**63 - 1, size=size) % (high - low) + low)
                        .int()
                        .tolist()
                    )
                    locs = torch.stack(
                        [
                            positions[i]
                            for positions, i in zip(navigable_positions, idxs)
                        ],
                        dim=0,
                    )
                    rots = torch.stack(
                        [rot[i] for rot, i in zip(rots, idxs)],
                        dim=0,
                    )
                else:
                    raise NotImplementedError()
                robot_pos = self.agent.robot.pose.p
                robot_pos[env_idx, :2] = locs
                self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

                qpos[env_idx, 2] = rots
                if self.randomize_base:
                    # base pos
                    robot_pos = self.agent.robot.pose.p
                    robot_pos[env_idx, :2] += torch.clamp(
                        torch.normal(0, 0.1, robot_pos[env_idx, :2].shape), -0.2, 0.2
                    ).to(self.device)
                    self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))
                    # base rot
                    qpos[env_idx, 2:3] += torch.clamp(
                        torch.normal(0, 0.25, qpos[env_idx, 2:3].shape), -0.5, 0.5
                    ).to(self.device)
                if self.randomize_arm:
                    qpos[env_idx, 5:6] += torch.clamp(
                        torch.normal(0, 0.05, qpos[env_idx, 5:6].shape), -0.1, 0.1
                    ).to(self.device)
                    qpos[env_idx, 7:-2] += torch.clamp(
                        torch.normal(0, 0.05, qpos[env_idx, 7:-2].shape), -0.1, 0.1
                    ).to(self.device)
                self.agent.reset(qpos)

                robot_init_p[env_idx] = self.agent.robot.pose.p[env_idx].clone()
                robot_init_q[env_idx] = self.agent.robot.pose.q[env_idx].clone()
                robot_init_qpos[env_idx] = self.agent.robot.qpos[env_idx].clone()

                if physx.is_gpu_enabled():
                    self._scene._gpu_apply_all()
                    self._scene.px.gpu_update_articulation_kinematics()
                    self._scene._gpu_fetch_all()
                self._scene.step()

                robot_force = (
                    self.agent.robot.get_net_contact_forces(
                        self.force_articulation_link_ids
                    )
                    .norm(dim=-1)
                    .sum(dim=-1)
                )

                self.scene_builder.initialize(original_env_idx, self.init_config_idxs)
                self.agent.reset(robot_init_qpos)
                self.agent.robot.set_pose(
                    Pose.create_from_pq(p=robot_init_p, q=robot_init_q)
                )

                if torch.all((robot_force < 1e-3)[env_idx]):
                    break

                env_idx = torch.where(robot_force >= 1e-3)[0]

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------
    # Remove irrelevant obs for pick task from state dict
    # -------------------------------------------------------------------------------------------------

    def _get_obs_state_dict(self, info: Dict):
        state_dict = super()._get_obs_state_dict(info)

        extra_state_dict_keys = list(state_dict["extra"].keys())
        for key in extra_state_dict_keys:
            if key not in PICK_OBS_EXTRA_KEYS:
                state_dict["extra"].pop(key, None)

        return state_dict

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): evaluate() function here to support continuous task wrapper on cpu sim
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        with torch.device(self.device):
            infos = super().evaluate()

            # set to zero in case we use continuous task wrapper in cpu sim
            #   this way, if the termination signal is ignored, env will
            #   still reevaluate success each step
            self.subtask_pointer = torch.zeros_like(self.subtask_pointer)

            robot_force = (
                self.agent.robot.get_net_contact_forces(
                    self.force_articulation_link_ids
                )
                .norm(dim=-1)
                .sum(dim=-1)
            )
            self.robot_cumulative_force += robot_force

            infos.update(
                robot_force=robot_force,
                robot_cumulative_force=self.robot_cumulative_force,
            )

            return infos

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj_pos = self.subtask_objs[0].pose.p
            rest_pos = self.ee_rest_world_pose.p
            tcp_pos = self.agent.tcp_pose.p

            # NOTE (arth): reward "steps" are as follows:
            #       - reaching_reward
            #       - if not grasped
            #           - not_grasped_reward
            #       - is_grasped_reward
            #       - if grasped
            #           - grasped_rewards
            #       - if grasped and ee_at_rest
            #           - static_reward
            #       - success_reward
            # ---------------------------------------------------
            # CONDITION CHECKERS
            # ---------------------------------------------------

            not_grasped = ~info["is_grasped"]
            not_grasped_reward = torch.zeros_like(reward[not_grasped])

            is_grasped = info["is_grasped"]
            is_grasped_reward = torch.zeros_like(reward[is_grasped])

            ee_rest = is_grasped & (
                torch.norm(tcp_pos - rest_pos, dim=1) <= self.pick_cfg.ee_rest_thresh
            )
            ee_rest_reward = torch.zeros_like(reward[ee_rest])

            # ---------------------------------------------------

            # reaching reward
            tcp_to_obj_dist = torch.norm(obj_pos - tcp_pos, dim=1)
            reaching_rew = 3 * (1 - torch.tanh(5 * tcp_to_obj_dist))
            reward += reaching_rew

            # penalty for ee moving too much when not grasping
            ee_vel = self.agent.tcp.linear_velocity
            ee_still_rew = 1 - torch.tanh(torch.norm(ee_vel, dim=1) / 5)
            reward += ee_still_rew

            # pick reward
            grasp_rew = 2 * info["is_grasped"]
            reward += grasp_rew

            # success reward
            success_rew = 3 * info["success"]
            reward += success_rew

            # encourage arm and torso in "resting" orientation
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )
            arm_resting_orientation_rew = 1 - torch.tanh(arm_to_resting_diff / 5)
            reward += arm_resting_orientation_rew

            # ---------------------------------------------------------------
            # colliisions
            step_no_col_rew = 1 - torch.tanh(
                3
                * (
                    torch.clamp(
                        self.robot_force_mult * info["robot_force"],
                        min=self.robot_force_penalty_min,
                    )
                    - self.robot_force_penalty_min
                )
            )
            reward += step_no_col_rew

            # cumulative collision penalty
            cum_col_under_thresh_rew = (
                info["robot_cumulative_force"] < self.robot_cumulative_force_limit
            ).float()
            reward += cum_col_under_thresh_rew
            # ---------------------------------------------------------------

            if torch.any(not_grasped):
                # penalty for torso moving up and down too much
                tqvel_z = self.agent.robot.qvel[..., 3][not_grasped]
                torso_not_moving_rew = 1 - torch.tanh(5 * torch.abs(tqvel_z))
                not_grasped_reward += torso_not_moving_rew

                # penalty for ee not over obj
                ee_over_obj_rew = 1 - torch.tanh(
                    5
                    * torch.norm(
                        obj_pos[..., :2][not_grasped] - tcp_pos[..., :2][not_grasped],
                        dim=1,
                    )
                )
                not_grasped_reward += ee_over_obj_rew

            if torch.any(is_grasped):
                # not_grasped reward has max of +2
                # so, we add +2 to grasped reward so reward only increases as task proceeds
                is_grasped_reward += 2

                # place reward
                ee_to_rest_dist = torch.norm(
                    tcp_pos[is_grasped] - rest_pos[is_grasped], dim=1
                )
                place_rew = 5 * (1 - torch.tanh(3 * ee_to_rest_dist))
                is_grasped_reward += place_rew

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][is_grasped]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                is_grasped_reward += base_still_rew

            if torch.any(ee_rest):
                qvel = self.agent.robot.qvel[..., :-2][ee_rest]
                static_rew = 1 - torch.tanh(torch.norm(qvel, dim=1))
                ee_rest_reward += static_rew

            # add rewards to specific envs
            reward[not_grasped] += not_grasped_reward
            reward[is_grasped] += is_grasped_reward
            reward[ee_rest] += ee_rest_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 21.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
