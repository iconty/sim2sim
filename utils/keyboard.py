# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) control."""

import numpy as np
import weakref
from collections.abc import Callable
import torch
import carb
import omni
from legged_lab.envs.base.base_env import BaseEnv
from isaaclab.devices.device_base import DeviceBase


class Keyboard(DeviceBase):

    def __init__(self, env: BaseEnv):
        """Initialize the keyboard layer."""
        self.env = env
        self.device = self.env.device
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
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)
        self.look_at_id = 0
        self.fixed_cam = False
        self.key = ""

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
        pass

    def add_callback(self, key: str, func: Callable):
        pass

    def advance(self):
        pass

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
                self.key = event.input.name
                if event.input.name == "R":
                    self.env.episode_length_buf = torch.ones_like(self.env.episode_length_buf) * 1e6
                if event.input.name == "LEFT_BRACKET":
                    self.look_at_id  = (self.look_at_id-1) % self.env.num_envs
                    self.look_at()
                if event.input.name == "RIGHT_BRACKET":
                    self.look_at_id  = (self.look_at_id+1) % self.env.num_envs
                    self.look_at()
                if event.input.name == "SLASH":
                    self.look_at()
                if event.input.name == "PERIOD":
                    self.fixed_cam = not self.fixed_cam
        # since no error, we are fine :)
        return True
    
    def look_at(self):
        # 获取目标位置（假设返回的是PyTorch张量）
        look_at_pos = self.env.robot.data.body_com_pos_w[self.look_at_id, 0, :3].clone()
        # 计算相机位置（确保self.lookat_vec是张量）
        cam_pos = look_at_pos + self.lookat_vec
        # 转换为Python原生float列表
        cam_pos = [float(x) for x in cam_pos.detach().cpu().numpy().squeeze()]
        look_at_pos = [float(x) for x in look_at_pos.detach().cpu().numpy().squeeze()]
        # 直接传递列表参数（不需要元组）
        self.env.sim.set_camera_view(cam_pos, look_at_pos)


    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # forward command
            "R": "reset envs",
            "LEFT_BRACKET": "prev_id,[",
            "RIGHT_BRACKET": "next_id,]",
            "SLASH": "lookat,/",
            "PERIOD": "fixed_cam,."
        }
