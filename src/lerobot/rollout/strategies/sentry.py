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

"""Sentry rollout strategy: continuous autonomous recording with auto-upload."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock

from lerobot.common.control_utils import is_headless
from lerobot.datasets import VideoEncodingManager
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import SentryStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, estimate_max_episode_seconds, safe_push_to_hub, send_next_action

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


class SentryStrategy(RolloutStrategy):
    """Continuous autonomous rollout with always-on recording.

    Episode duration is derived from camera resolution, FPS, and
    ``DEFAULT_VIDEO_FILE_SIZE_IN_MB`` so that each saved episode
    produces a video file that has crossed the chunk-size boundary.
    This keeps ``push_to_hub`` efficient — it uploads complete video
    files rather than re-uploading a still-growing one.

    The dataset is pushed to the Hub via a bounded single-worker executor
    so no push is ever silently dropped and exactly one push runs at a
    time.

    Policy state (hidden state, RTC queue) is reset between ``--num_rollouts``
    episodes so each episode starts from a consistent state.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    to prevent disk I/O from blocking the control loop.

    Multi-episode flow (``--num_rollouts=N``)::

        Return to start_position  →  wait for Space  →  record episode (auto-rotate)
          ↑                                                                    │
          └──────────────── Right Arrow ends episode ─────────────────────────┘

    Press **Esc** at any time to abort the entire session.
    """

    config: SentryStrategyConfig

    def __init__(self, config: SentryStrategyConfig):
        super().__init__(config)
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()
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
        """Initialise the inference engine, push executor, and keyboard listener."""
        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentry-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )
        self._listener = self._start_keyboard_listener(
            self._episode_start_event, self._episode_end_event, ctx.runtime.shutdown_event
        )
        logger.info(
            "Sentry strategy ready (episode_duration=%.0fs, upload_every=%d eps)",
            self._episode_duration_s,
            self.config.upload_every_n_episodes,
        )

    def run(self, ctx: RolloutContext) -> None:
        """Run autonomous recording episodes until shutdown or ``num_rollouts``
        is exhausted.
        """
        cfg = ctx.runtime.cfg
        num_episodes = max(cfg.num_rollouts, 1)
        episode = 0

        while episode < num_episodes and not ctx.runtime.shutdown_event.is_set():
            episode += 1
            self._episode_end_event.clear()

            # Return to start position before every episode.
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
        while not self._episode_start_event.is_set() and not shutdown_event.is_set():
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Single-episode recording loop
    # ------------------------------------------------------------------

    def _run_single_episode(self, ctx: RolloutContext) -> None:
        """Run the continuous recording loop for one episode with automatic
        episode rotation based on video file size.

        Exits when *duration* expires, *shutdown_event* is set, or
        *episode_end_event* is set (Right Arrow).
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)

        play_sounds = cfg.play_sounds
        episode_duration_s = self._episode_duration_s

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        logger.info("Sentry recording started (episode_duration=%.0fs)", episode_duration_s)

        with VideoEncodingManager(dataset):
            try:
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

                    if action_dict is not None:
                        self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        frame = {**obs_frame, **action_frame, "task": task_str}
                        # ``add_frame`` writes to the in-progress episode buffer; the
                        # background pusher only ever touches *finalised* episode
                        # artifacts on disk.  The two operate on disjoint state, so
                        # ``add_frame`` does not need ``_episode_lock``.
                        dataset.add_frame(frame)

                    # Episode rotation derived from video file-size target.
                    # The duration is a conservative estimate so the actual
                    # video has crossed DEFAULT_VIDEO_FILE_SIZE_IN_MB by now,
                    # keeping push_to_hub efficient (uploads complete files).
                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= episode_duration_s:
                        # ``save_episode`` finalises the in-progress episode and
                        # flushes it to disk; ``_episode_lock`` serialises this with
                        # ``push_to_hub`` (run in the background executor) so the
                        # pusher never reads a half-written episode.
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info(
                            "Episode saved (total: %d, elapsed: %.1fs)",
                            dataset.num_episodes,
                            elapsed,
                        )
                        log_say(f"Episode {dataset.num_episodes} saved", play_sounds)

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()

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

            finally:
                logger.info("Sentry recording paused — saving final episode")
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()

    def teardown(self, ctx: RolloutContext) -> None:
        """Flush pending pushes, finalise the dataset, and disconnect hardware."""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping sentry recording")
        log_say("Stopping sentry recording", play_sounds)

        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()

        # Flush any queued/running push cleanly.
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if self._needs_push.is_set() and ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                logger.info("Pushing final dataset to hub...")
                if safe_push_to_hub(
                    ctx.data.dataset,
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                ):
                    logger.info("Dataset uploaded to hub")
                    log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Sentry strategy teardown complete")

    def _background_push(self, dataset, cfg) -> None:
        """Queue a Hub push on the single-worker executor.

        The executor's max_workers=1 guarantees at most one push runs at
        a time; submitted tasks are queued rather than dropped.
        """
        if self._push_executor is None:
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    if safe_push_to_hub(
                        dataset,
                        tags=cfg.dataset.tags if cfg.dataset else None,
                        private=cfg.dataset.private if cfg.dataset else False,
                    ):
                        self._needs_push.clear()
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        logger.info("Background push task submitted")
