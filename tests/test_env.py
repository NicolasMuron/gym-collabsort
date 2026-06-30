"""
Unit tests for environment.
"""

import gymnasium as gym
import pygame
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock, patch

import gym_collabsort
from gym_collabsort.config import Action, Config, RenderMode
from gym_collabsort.envs.env import CollabSortEnv
from gym_collabsort.envs.robot import Robot


def test_version() -> None:
    """Test environment version"""

    # Check that version string is not empty
    assert gym_collabsort.__version__


def test_check_env() -> None:
    """Test compatibility with Gymnasium API"""

    check_env(CollabSortEnv())


def test_api() -> None:
    """Test environment API"""

    env = CollabSortEnv()
    _, info = env.reset()

    assert info["n_collisions"] == 0
    assert info["n_placed_objects"] == 0

    # TODO test observation format


def test_render_rgb() -> None:
    """Test RGB rendering"""

    env = CollabSortEnv(config=Config(render_mode=RenderMode.RGB_ARRAY))
    env.reset()

    env.step(action=env.action_space.sample())

    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[0] == env.config.window_dimensions[1]
    assert frame.shape[1] == env.config.board_width


def test_random_agent() -> None:
    """Test an agent choosing random actions"""

    env = gym.make("CollabSort-v0")
    env.reset()

    for _ in range(60):
        _, _, _, _, _ = env.step(action=env.action_space.sample())

    env.close()


def test_reward_noise_is_applied() -> None:
    """Test that configured reward noise affects the returned reward"""

    config = Config(
        reward_noise_std=1.0,
        render_mode=RenderMode.RGB_ARRAY,
        robot_enabled=False,
    )
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    rewards = [env.step(action=Action.NONE.value)[1] for _ in range(100)]

    assert np.std(rewards) > 0.5


def test_reward_noise_not_applied_when_zero() -> None:
    """Test that no reward noise is applied when std is 0"""

    config = Config(
        reward_noise_std=0.0,
        render_mode=RenderMode.RGB_ARRAY,
        robot_enabled=False,
    )
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    rewards = [env.step(action=Action.NONE.value)[1] for _ in range(100)]

    assert np.std(rewards) == 0.0


def test_reward_change_step_switches_matrices() -> None:
    """Test that the agent reward matrix changes after the configured step threshold."""

    agent_rewards_after = np.array(
        [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    )

    config = Config(
        render_mode=RenderMode.RGB_ARRAY,
        robot_enabled=False,
        enable_reward_change=True,
        reward_change_step=5, 
        agent_rewards_after=agent_rewards_after,
    )
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    assert np.array_equal(env.current_agent_rewards, config.agent_rewards)

    for _ in range(5):
        env.step(action=Action.NONE.value)

    assert env.total_steps == 5
    assert np.array_equal(env.current_agent_rewards, agent_rewards_after)

    env.step(action=Action.NONE.value)
    assert env.total_steps == 6
    assert np.array_equal(env.current_agent_rewards, agent_rewards_after)


def test_robotic_agent(pause_at_end: bool = False) -> None:
    """Test an agent using the same behavior as the robot, but with specific rewards"""

    config = Config(render_mode=RenderMode.HUMAN, n_objects=10)

    env = CollabSortEnv(config=config)
    env.reset()

    # Use robot policy with agent rewards
    robotic_agent = Robot(
        board=env.board,
        arm=env.board.agent_arm,
        rewards=config.agent_rewards,
    )

    ep_over: bool = False
    while not ep_over:
        _, _, terminated, truncated, _ = env.step(
            action=robotic_agent.choose_action().value
        )
        ep_over = terminated or truncated

    if pause_at_end:
        # Wait for any user input to exit environment
        pygame.event.clear()
        _ = pygame.event.wait()

    env.close()


def test_disabled_robot_env() -> None:
    """Test environment with robot arm disabled"""

    config = Config(robot_enabled=False, render_mode=RenderMode.RGB_ARRAY)
    env = CollabSortEnv(config=config)
    obs, info = env.reset()

    assert info["n_collisions"] == 0
    assert info["n_placed_objects"] == 0
    assert env.robot is None

    # Check observation format
    assert "self" in obs
    assert "robot" in obs
    assert "moving_objects" in obs

    # Robot coordinates should still be returned, corresponding to robot retracted base location [1, 4]
    assert (obs["robot"] == [1, 4]).all()

    # Step the environment
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(
            action=env.action_space.sample()
        )
        # Robot position must remain retracted
        assert (obs["robot"] == [1, 4]).all()
        # No collisions should ever occur because the robot arm is not active
        assert info["n_collisions"] == 0

    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3

    env.close()


def test_configurable_treadmills() -> None:
    """Test environment with various treadmill configurations"""

    treadmill_configs = [
        ("upper",),
        ("middle",),
        ("lower",),
        ("upper", "middle"),
        ("upper", "lower"),
        ("middle", "lower"),
        ("upper", "middle", "lower"),
    ]

    for active in treadmill_configs:
        config = Config(
            active_treadmills=active,
            render_mode=RenderMode.RGB_ARRAY,
            n_objects=20,
        )
        env = CollabSortEnv(config=config)
        env.reset()

        # Run enough steps to spawn several objects
        for _ in range(100):
            env.step(action=env.action_space.sample())

        # All spawned objects should be on an active treadmill row
        active_rows = set(config.treadmill_rows)
        for obj in env.board.objects:
            assert obj.coords.row in active_rows, (
                f"Object at row {obj.coords.row} not in active rows {active_rows} "
                f"for config active_treadmills={active}"
            )

        # Rendering must work
        frame = env.render()
        assert frame is not None
        assert frame.ndim == 3

        env.close()


def test_empty_treadmills_raises() -> None:
    """Test that an empty treadmill configuration raises an error"""

    with pytest.raises((ValueError, AssertionError)):
        config = Config(active_treadmills=())
        env = CollabSortEnv(config=config)
        env.reset()


def test_collision_drops_held_objects_and_increments_removed() -> None:
    """Test that a collision forces both arms to drop their objects and increments n_removed_objects."""

    config = Config(render_mode=RenderMode.RGB_ARRAY, robot_enabled=True)
    env = CollabSortEnv(config=config)
    env.reset(seed=42)

    env.board.robot_arm.act = MagicMock(return_value=(True, None, None))
    env.board.agent_arm.act = MagicMock(return_value=(True, None, None))

    env.board.robot_arm._picked_object = MagicMock()
    env.board.agent_arm._picked_object = MagicMock()

    with (
        patch.object(
            type(env.board.robot_arm),
            "picked_object",
            new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(
            type(env.board.agent_arm),
            "picked_object",
            new_callable=PropertyMock,
            return_value=True,
        ),
    ):
        env.step(action=Action.NONE.value)
        
    assert env.n_collisions == 1

    env.board.robot_arm._picked_object.empty.assert_called_once()
    env.board.agent_arm._picked_object.empty.assert_called_once()

    assert env.n_removed_objects == 2

    env.close()


if __name__ == "__main__":
    # Standalone execution with pause at end
    test_robotic_agent(pause_at_end=True)
