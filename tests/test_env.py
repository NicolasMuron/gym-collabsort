"""
Unit tests for environment.
"""

import gymnasium as gym
import pygame
from gymnasium.utils.env_checker import check_env

import gym_collabsort
from gym_collabsort.config import Config, RenderMode
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


if __name__ == "__main__":
    # Standalone execution with pause at end
    test_robotic_agent(pause_at_end=True)
