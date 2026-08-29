# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Keyboard controller for SE(2) control."""

import weakref
from collections.abc import Callable

import carb
import omni
import torch
from isaaclab.devices.device_base import DeviceBase
from isaaclab.utils import math as math_utils

from legged_lab.envs.base.base_env import BaseEnv


class Keyboard(DeviceBase):
    def __init__(
        self,
        env: BaseEnv,
        enable_velocity_control: bool = False,
        linear_step: float = 0.1,
        angular_step: float = 0.1,
        command_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        enable_push_control: bool = False,
        push_step: float = 0.2,
        random_push_max: float = 1.0,
    ):
        """Initialize the keyboard layer."""
        self.env = env
        self.enable_velocity_control = enable_velocity_control
        self.linear_step = linear_step
        self.angular_step = angular_step
        self.command_limits = command_limits or ((-0.6, 1.0), (-0.5, 0.5), (-1.0, 1.0))
        self.enable_push_control = enable_push_control
        self.push_step = push_step
        self.random_push_max = random_push_max
        self._command = [0.0, 0.0, 0.0]
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for ManagerBasedRLEnv: {self.__class__.__name__}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        self._command = [0.0, 0.0, 0.0]
        self._apply_velocity_command()

    def add_callback(self, key: str, func: Callable):
        pass

    def advance(self):
        if self.enable_velocity_control:
            # Re-apply the selected command because the environment's command
            # generator can otherwise overwrite it on an episode reset.
            self._apply_velocity_command()

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING:
                if event.input.name == "R":
                    self.env.episode_length_buf = torch.ones_like(self.env.episode_length_buf) * 1e6
                elif event.input.name == "N":
                    # Next robot: increment lookat_id
                    if hasattr(self.env, 'lookat_id'):
                        self.env.lookat_id = (self.env.lookat_id + 1) % self.env.num_envs
                        print(f"[Keyboard] Switched to robot {self.env.lookat_id}")
                elif event.input.name == "P":
                    # Previous robot: decrement lookat_id
                    if hasattr(self.env, 'lookat_id'):
                        self.env.lookat_id = (self.env.lookat_id - 1) % self.env.num_envs
                        print(f"[Keyboard] Switched to robot {self.env.lookat_id}")
                elif self.enable_push_control and event.input.name in {"I", "K", "J", "L", "M"}:
                    self._apply_push(event.input.name)
                elif self.enable_velocity_control:
                    self._update_velocity_command(event.input.name)

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            "R": "reset envs",
            "N": "next robot (lookat_id + 1)",
            "P": "previous robot (lookat_id - 1)",
        }
        if self.enable_velocity_control:
            self._INPUT_KEY_MAPPING.update(
                {
                    "W": "increase forward velocity",
                    "S": "decrease forward velocity",
                    "A": "increase leftward velocity",
                    "D": "increase rightward velocity",
                    "Q": "increase left yaw velocity",
                    "E": "increase right yaw velocity",
                    "SPACE": "zero all velocity commands",
                }
            )
        if self.enable_push_control:
            self._INPUT_KEY_MAPPING.update(
                {
                    "I": "forward velocity impulse",
                    "K": "backward velocity impulse",
                    "J": "leftward velocity impulse",
                    "L": "rightward velocity impulse",
                    "M": "random velocity impulse",
                }
            )

    def _update_velocity_command(self, key: str):
        """Change the target velocity by exactly one step for one key press."""
        if key == "W":
            self._command[0] += self.linear_step
        elif key == "S":
            self._command[0] -= self.linear_step
        elif key == "A":
            self._command[1] += self.linear_step
        elif key == "D":
            self._command[1] -= self.linear_step
        elif key == "Q":
            self._command[2] += self.angular_step
        elif key == "E":
            self._command[2] -= self.angular_step
        elif key == "SPACE":
            self._command = [0.0, 0.0, 0.0]

        self._command = [
            min(max(value, limits[0]), limits[1])
            for value, limits in zip(self._command, self.command_limits)
        ]
        self._apply_velocity_command()
        print(
            "[Keyboard] command: "
            f"vx={self._command[0]:+.2f} m/s, "
            f"vy={self._command[1]:+.2f} m/s, "
            f"yaw={self._command[2]:+.2f} rad/s"
        )

    def _apply_velocity_command(self):
        if not self.enable_velocity_control:
            return
        command = torch.tensor(self._command, device=self.env.device, dtype=self.env.command_generator.command.dtype)
        self.env.command_generator.command[:] = command

    def _apply_push(self, key: str):
        """Apply one directional heading-frame or random world-frame velocity jump."""
        robot_id = int(getattr(self.env, "lookat_id", 0))
        env_ids = torch.tensor([robot_id], device=self.env.device, dtype=torch.long)
        delta_heading = torch.zeros(1, 3, device=self.env.device)
        if key == "I":
            delta_heading[:, 0] = self.push_step
        elif key == "K":
            delta_heading[:, 0] = -self.push_step
        elif key == "J":
            delta_heading[:, 1] = self.push_step
        elif key == "L":
            delta_heading[:, 1] = -self.push_step
        elif key == "M":
            delta_heading[:, :2].uniform_(-self.random_push_max, self.random_push_max)

        if key == "M":
            # Match the original training event: independent world-frame x/y samples.
            delta_world = delta_heading
            delta_label = "delta_v_world"
        else:
            heading_quat = math_utils.yaw_quat(self.env.robot.data.root_link_quat_w[env_ids])
            delta_world = math_utils.quat_apply(heading_quat, delta_heading)
            delta_label = "delta_v_heading"
        root_velocity = self.env.robot.data.root_vel_w[env_ids].clone()
        root_velocity[:, :2] += delta_world[:, :2]
        self.env.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        print(
            "[Keyboard] push: "
            f"robot={robot_id}, "
            f"{delta_label}=({delta_heading[0, 0]:+.2f}, {delta_heading[0, 1]:+.2f}) m/s"
        )
