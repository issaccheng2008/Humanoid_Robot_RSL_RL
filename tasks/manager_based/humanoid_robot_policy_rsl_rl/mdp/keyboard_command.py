# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard velocity command for policy playback."""

from collections.abc import Sequence
import weakref

import carb
import omni

from isaaclab.envs.mdp import UniformVelocityCommand


class KeyboardVelocityCommand(UniformVelocityCommand):
    """Generate velocity commands from all currently held keys."""

    FORWARD_SPEED = 0.7
    TURN_FORWARD_SPEED = 0.20
    TURN_SPEED = 0.7

    FORWARD_KEYS = {"W", "UP"}
    LEFT_KEYS = {"A", "LEFT"}
    RIGHT_KEYS = {"D", "RIGHT"}

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # Store all keys that are currently being held.
        self._held_keys: set[str] = set()

        # Connect directly to the Isaac Sim keyboard interface.
        self._app_window = omni.appwindow.get_default_app_window()
        self._input_interface = carb.input.acquire_input_interface()
        self._keyboard = self._app_window.get_keyboard()

        self._keyboard_subscription = (
            self._input_interface.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self):
                    obj._on_keyboard_event(event, *args),
            )
        )

        print(
            "[INFO] Keyboard control enabled:\n"
            "  W / Up          : move forward\n"
            "  A / Left        : turn left\n"
            "  D / Right       : turn right\n"
            "  Up + Left/Right : move forward while turning\n"
            "  S / Down        : no effect"
        )

    def _on_keyboard_event(self, event, *args):
        """Record each key's pressed/released state."""
        key = event.input.name

        controlled_keys = (
            self.FORWARD_KEYS
            | self.LEFT_KEYS
            | self.RIGHT_KEYS
        )

        if key not in controlled_keys:
            return True

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._held_keys.add(key)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held_keys.discard(key)

        return True

    def _is_any_key_held(self, keys: set[str]) -> bool:
        """Return whether at least one key in a group is held."""
        return not self._held_keys.isdisjoint(keys)

    def _resample_command(self, env_ids: Sequence[int]):
        """Disable random command sampling during playback."""
        self.vel_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        """Construct a command from all simultaneously held keys."""
        forward_pressed = self._is_any_key_held(self.FORWARD_KEYS)
        left_pressed = self._is_any_key_held(self.LEFT_KEYS)
        right_pressed = self._is_any_key_held(self.RIGHT_KEYS)

        # If both turning directions are held, cancel the turn.
        turn_left = left_pressed and not right_pressed
        turn_right = right_pressed and not left_pressed
        turning = turn_left or turn_right

        # Forward speed:
        # - Up/W held: 0.7 m/s, including while turning.
        # - Turning alone: 0.20 m/s, to help the robot turn properly.
        # - Otherwise: zero.
        if forward_pressed:
            forward_velocity = self.FORWARD_SPEED
        elif turning:
            forward_velocity = self.TURN_FORWARD_SPEED
        else:
            forward_velocity = 0.0

        # Yaw rate.
        if turn_left:
            yaw_velocity = self.TURN_SPEED
        elif turn_right:
            yaw_velocity = -self.TURN_SPEED
        else:
            yaw_velocity = 0.0

        # Broadcast the command to all playback environments.
        self.vel_command_b[:, 0] = forward_velocity
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = yaw_velocity

    def __del__(self):
        """Release the keyboard subscription."""
        if (
            hasattr(self, "_input_interface")
            and hasattr(self, "_keyboard")
            and hasattr(self, "_keyboard_subscription")
            and self._keyboard_subscription is not None
        ):
            self._input_interface.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._keyboard_subscription,
            )
            self._keyboard_subscription = None

        # Release the command term's visualization callback.
        try:
            super().__del__()
        except AttributeError:
            pass
