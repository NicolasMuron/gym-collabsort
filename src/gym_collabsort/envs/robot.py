"""
Implementation of robot policy.
"""

import numpy as np

from ..board.arm import Arm
from ..board.board import Board
from ..board.object import Object
from ..config import Action, RobotStrategy


class Robot:
    def __init__(
        self,
        board: Board,
        arm: Arm,
        rewards: np.ndarray,
        strategy: RobotStrategy | None = None,
    ) -> None:
        self.board = board
        self.arm = arm
        self.rewards = rewards
        self.strategy = strategy or RobotStrategy.BEST_OBJECT

    def choose_action(self) -> Action:
        """Return the chosen action"""

        if self.arm.moving_back:
            return Action.NONE

        if self.strategy == RobotStrategy.RANDOM_ACTION:
            return self._choose_random_action()

        target_object = self._select_target_object()
        if target_object is None:
            return Action.NONE

        if target_object.coords == self.arm.gripper.coords:
            return Action.PICK

        target_col_delta = target_object.coords.col - self.arm.base.coords.col
        gripper_row = self.arm.gripper.coords.row
        target_row = target_object.coords.row

        if target_col_delta == target_row - gripper_row:
            return Action.DOWN

        if target_col_delta == gripper_row - target_row:
            return Action.UP

        return Action.NONE

    def _select_target_object(self) -> Object | None:
        """Return the target object according to the configured strategy."""

        reachable_objects = self._get_reachable_objects()
        if not reachable_objects:
            return None

        if self.strategy == RobotStrategy.RANDOM_OBJECT:
            # Fix: Use integers() to get a random index safely
            idx = self.board.rng.integers(len(reachable_objects))
            return reachable_objects[idx]

        if self.strategy == RobotStrategy.CLOSEST_OBJECT:
            return min(
                reachable_objects,
                key=lambda obj: (
                    abs(obj.coords.col - self.arm.gripper.coords.col)
                    + abs(obj.coords.row - self.arm.gripper.coords.row)
                ),
            )

        # Default behavior: choose the most rewarding reachable object
        reachable_objects.sort(
            key=lambda obj: obj.get_reward(rewards=self.rewards), reverse=True
        )
        return reachable_objects[0]

    def _choose_random_action(self) -> Action:
        """Return a random action among the possible robot actions."""

        return self.board.rng.choice([Action.NONE, Action.UP, Action.DOWN, Action.PICK])

    def _get_reachable_objects(self) -> list[Object]:
        """Return the objects that can potentially be picked in the future."""

        return [
            obj
            for obj in self.board.objects
            if (obj.coords.col - self.arm.gripper.coords.col)
            >= abs(obj.coords.row - self.arm.gripper.coords.row)
        ]
