"""
Unit tests for robot.
"""

import numpy as np

from gym_collabsort.board.object import Color, Shape
from gym_collabsort.config import Action, Config, RenderMode, RobotStrategy
from gym_collabsort.envs.env import CollabSortEnv
from gym_collabsort.envs.robot import Robot
from unittest.mock import MagicMock


def test_robot_supports_multiple_strategies() -> None:
    """Test that the robot accepts multiple target-selection strategies"""

    config = Config(render_mode=RenderMode.RGB_ARRAY, n_objects=10)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    strategies = [
        RobotStrategy.BEST_OBJECT,
        RobotStrategy.RANDOM_OBJECT,
        RobotStrategy.CLOSEST_OBJECT,
        RobotStrategy.RANDOM_ACTION,
    ]

    for strategy in strategies:
        robot = Robot(
            board=env.board,
            arm=env.board.robot_arm,
            rewards=config.robot_rewards,
            strategy=strategy,
        )
        action = robot.choose_action()

        assert action in set(Action)

    env.close()


def test_best_object_strategy_prefers_highest_reward() -> None:
    """The robot should prefer the reachable object with the highest reward."""

    config = Config(render_mode=RenderMode.RGB_ARRAY, n_objects=0)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    from pygame.math import Vector2

    from gym_collabsort.board.object import Object

    reward_matrix = np.array([[0, 0, 0], [0, 0, 0], [10, 10, 10]])

    high_reward_obj = Object(
        location=Vector2(x=300, y=100),
        config=config,
        color=Color.RED,
        shape=Shape.SQUARE,
    )
    low_reward_obj = Object(
        location=Vector2(x=250, y=100),
        config=config,
        color=Color.BLUE,
        shape=Shape.SQUARE,
    )

    env.board.objects.add(high_reward_obj)
    env.board.objects.add(low_reward_obj)

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=reward_matrix,
        strategy=RobotStrategy.BEST_OBJECT,
    )

    target = robot._select_target_object()

    assert target is high_reward_obj

    env.close()


def test_closest_object_strategy_selects_nearest_reachable_object() -> None:
    """The robot should select the closest reachable object."""

    config = Config(render_mode=RenderMode.RGB_ARRAY, n_objects=0)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    from pygame.math import Vector2

    from gym_collabsort.board.object import Object

    env.board.objects.empty()

    close_obj = Object(
        location=Vector2(x=220, y=100),
        config=config,
        color=Color.RED,
        shape=Shape.SQUARE,
    )
    far_obj = Object(
        location=Vector2(x=350, y=100),
        config=config,
        color=Color.BLUE,
        shape=Shape.SQUARE,
    )

    env.board.objects.add(close_obj)
    env.board.objects.add(far_obj)

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=config.robot_rewards,
        strategy=RobotStrategy.CLOSEST_OBJECT,
    )

    target = robot._select_target_object()

    assert target is close_obj

    env.close()


def test_random_action_strategy_returns_valid_action() -> None:
    """The random-action strategy should return one of the allowed robot actions."""

    config = Config(render_mode=RenderMode.RGB_ARRAY, n_objects=0)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=config.robot_rewards,
        strategy=RobotStrategy.RANDOM_ACTION,
    )

    action = robot.choose_action()

    assert action in set(Action)

    env.close()


def test_random_object_strategy_selects_a_reachable_object() -> None:
    """The random-object strategy should select one of the reachable objects."""

    config = Config(render_mode=RenderMode.RGB_ARRAY, n_objects=0)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    from pygame.math import Vector2
    from gym_collabsort.board.object import Object

    env.board.objects.empty()

    reachable_obj = Object(
        location=Vector2(x=220, y=100),
        config=config,
        color=Color.RED,
        shape=Shape.SQUARE,
    )
    env.board.objects.add(reachable_obj)

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=config.robot_rewards,
        strategy=RobotStrategy.RANDOM_OBJECT,
    )

    target = robot._select_target_object()

    assert target is reachable_obj

    env.close()


def test_agent_target_strategy_uses_agent_position_when_no_explicit_target() -> None:
    """If no explicit agent target is available, the robot should target the object nearest the agent."""

    config = Config(render_mode=RenderMode.RGB_ARRAY, n_objects=0)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    from pygame.math import Vector2
    from gym_collabsort.board.object import Object

    env.board.objects.empty()

    near_agent_obj = Object(
        location=Vector2(x=220, y=100),
        config=config,
        color=Color.RED,
        shape=Shape.SQUARE,
    )
    far_from_agent_obj = Object(
        location=Vector2(x=350, y=100),
        config=config,
        color=Color.BLUE,
        shape=Shape.SQUARE,
    )

    env.board.objects.add(near_agent_obj)
    env.board.objects.add(far_from_agent_obj)

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=config.robot_rewards,
        strategy=RobotStrategy.AGENT_TARGET,
        agent_arm=env.board.agent_arm,
    )

    target = robot._select_target_object()

    assert target is near_agent_obj

    env.close()


def test_robot_slow_mode_returns_none_on_odd_steps() -> None:
    """When slow_mode is enabled, the robot returns Action.NONE on odd steps."""
    config = Config(render_mode=RenderMode.RGB_ARRAY)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=config.robot_rewards,
        strategy=RobotStrategy.BEST_OBJECT,
        slow_mode=True,
    )

    action_1 = robot.choose_action()
    assert action_1 == Action.NONE
    assert robot._step_count == 1

    robot.choose_action()
    assert robot._step_count == 2

    env.close()


def test_get_agent_target_object_returns_picked_object() -> None:
    """_get_agent_target_object returns the object currently held by the agent."""
    config = Config(render_mode=RenderMode.RGB_ARRAY)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    mock_agent_arm = MagicMock()
    mock_object = MagicMock()
    mock_agent_arm.picked_object = mock_object

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=config.robot_rewards,
        agent_arm=mock_agent_arm,
    )

    target = robot._get_agent_target_object()

    assert target is mock_object

    env.close()


def test_get_agent_target_object_returns_none_when_agent_arm_is_none() -> None:
    """_get_agent_target_object returns None immediately if agent_arm is None."""
    config = Config(render_mode=RenderMode.RGB_ARRAY)
    env = CollabSortEnv(config=config)
    env.reset(seed=0)

    robot = Robot(
        board=env.board,
        arm=env.board.robot_arm,
        rewards=config.robot_rewards,
        agent_arm=None,
    )

    target = robot._get_agent_target_object()
    assert target is None

    env.close()
