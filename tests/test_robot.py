"""
Unit tests for robot.
"""

import numpy as np

from gym_collabsort.board.object import Color, Shape
from gym_collabsort.config import Action, Config, RenderMode, RobotStrategy
from gym_collabsort.envs.env import CollabSortEnv
from gym_collabsort.envs.robot import Robot


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

    # On crée un objet volontairement proche pour qu'il soit détecté comme "reachable"
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

    # Cette fois, reachable_objects contient un élément,
    # donc le bloc interne de RANDOM_OBJECT va s'exécuter !
    target = robot._select_target_object()

    assert target is reachable_obj

    env.close()
