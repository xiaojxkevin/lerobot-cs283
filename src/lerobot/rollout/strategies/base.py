# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import os
import sys
import time
from threading import Event

from lerobot.common.control_utils import is_headless
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.

    Multi-episode flow (``--num_rollouts=N``)::

        Return to start_position  →  wait for Space  →  run episode
          ↑                                                    │
          └──────────── Right Arrow ends episode ──────────────┘

    Press **Esc** at any time to abort the entire session.
    """

    def __init__(self, config):
        super().__init__(config)
        self._listener = None
        self._episode_end_event = Event()
        self._episode_start_event = Event()

    # ------------------------------------------------------------------
    # Keyboard listener
    # ------------------------------------------------------------------

    @staticmethod
    def _start_keyboard_listener(
        episode_start_event: Event,
        episode_end_event: Event,
        shutdown_event: Event,
    ):
        """Start a pynput listener.

        ==============  ===========================================
        Key              Action
        ==============  ===========================================
        **Space**        Start the next episode (when waiting).
        **Right Arrow**  End the current episode early.
        **Esc**          Full shutdown.
        ==============  ===========================================
        """
        if not PYNPUT_AVAILABLE or is_headless():
            logger.warning("Headless environment or pynput unavailable — keyboard controls disabled")
            return None

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    episode_start_event.set()
                elif key == keyboard.Key.right:
                    logger.info("Right Arrow pressed — ending current episode")
                    episode_end_event.set()
                elif key == keyboard.Key.esc:
                    logger.info("Esc pressed — requesting full shutdown")
                    shutdown_event.set()
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        logger.info("Keyboard listener started (Space = start, Right Arrow = end, Esc = stop)")
        return listener

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine and keyboard listener."""
        self._init_engine(ctx)
        self._listener = self._start_keyboard_listener(
            self._episode_start_event, self._episode_end_event, ctx.runtime.shutdown_event
        )
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run autonomous rollout episodes until shutdown or ``num_rollouts``
        is exhausted.
        """
        cfg = ctx.runtime.cfg
        num_episodes = max(cfg.num_rollouts, 1)
        episode = 0

        while episode < num_episodes and not ctx.runtime.shutdown_event.is_set():
            episode += 1
            self._episode_end_event.clear()

            # Return to start position before every episode (including the
            # first — the robot is already there after ``build_rollout_context``,
            # so this is a no-op for episode 1 unless the robot was moved).
            if ctx.hardware.initial_position:
                if num_episodes > 1:
                    logger.info("Returning to start position (episode %d / %d)...", episode, num_episodes)
                try:
                    self._return_to_initial_position(
                        ctx.hardware, duration_s=cfg.start_position_duration
                    )
                except Exception as e:
                    logger.warning("Failed to return to start position: %s", e)

            if ctx.runtime.shutdown_event.is_set():
                break

            # Wait for Space before starting.
            logger.info("=== Episode %d / %d — press Space to start ===", episode, num_episodes)
            self._wait_for_space(ctx.runtime.shutdown_event)
            if ctx.runtime.shutdown_event.is_set():
                break

            # Reset and run.
            self._engine.reset()
            self._interpolator.reset()
            self._engine.resume()

            self._run_single_episode(ctx)

            self._engine.pause()

            if self._episode_end_event.is_set():
                logger.info("Episode %d ended by user", episode)

        logger.info("All %d episode(s) complete", num_episodes)

    def _wait_for_space(self, shutdown_event: Event) -> None:
        """Block until Space (start episode) or Esc (shutdown) is pressed."""
        self._episode_start_event.clear()
        # Drain any stale start signal from before the wait began.
        while not self._episode_start_event.is_set() and not shutdown_event.is_set():
            time.sleep(0.1)

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")

    # ------------------------------------------------------------------
    # Single-episode control loop
    # ------------------------------------------------------------------

    def _run_single_episode(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop for one episode.

        Exits when *duration* expires, *shutdown_event* is set, or
        *episode_end_event* is set (Right Arrow).
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        logger.info("Control loop started")

        while not ctx.runtime.shutdown_event.is_set() and not self._episode_end_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Control loop is running slower ({1 / dt:.1f} Hz) than "
                    f"the target FPS ({cfg.fps} Hz). Robot control may be "
                    f"unstable. Common causes: 1) Camera FPS not keeping up "
                    f"2) Policy inference taking too long 3) CPU starvation"
                )
