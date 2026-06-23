"""
Gym environment for a collaborative sorting task.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import EnvSpec

from ..board.board import Board
from ..board.object import Color, Shape
from ..config import Action, Config, RenderMode
from .robot import Robot


class CollabSortEnv(gym.Env):
    """Gym environment implementing a collaborative sorting task"""

    metadata = {
        # Supported rendering modes
        "render_modes": [rm.value for rm in RenderMode],
        # Default FPS for human rendering
        "render_fps": Config().render_fps,
    }

    def __init__(
        self,
        render_mode: RenderMode = RenderMode.NONE,
        config: Config | None = None,
    ) -> None:
        """Initialize the environment"""

        if config is None:
            # Use default configuration values
            config = Config()
            # Use rendering mode provided as constructor arg
            rm = render_mode
        else:
            # Use rendering mode provided by configuration
            rm = config.render_mode

        self.config = config
        self.render_mode = rm

        if self.spec is None:
            # Duplicate the info provided by register() in root __init__.py file.
            # Setting env as non-deterministic is mandatory for all check_env() tests to pass
            self.spec = EnvSpec(
                id="CollabSort-v0",
                entry_point="gym_collabsort.envs.env:CollabSortEnv",
                nondeterministic=True,
            )

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window: pygame.Surface | None = None
        self.clock = None

        # Create board
        self.board = Board(rng=self.np_random, config=self.config)

        # Create robot
        if self.config.robot_enabled:
            self.robot = Robot(
                board=self.board,
                arm=self.board.robot_arm,
                rewards=config.robot_rewards,
            )
        else:
            self.robot = None

        # Number of removed objects: placed by any arm or fallen from any treadmill.
        # Used to assess the end of episode
        self.n_removed_objects: int = 0

        # Number of collisions
        self.n_collisions: int = 0

        # Total rewards for the agent and robot
        self.agent_episode_reward: float = 0
        self.robot_episode_reward: float = 0

        # Define action format
        self.action_space = gym.spaces.Discrete(len(Action))

        # Define observation format. See _get_obs() method for details
        self.observation_space = gym.spaces.Dict(
            {
                "self": self._get_agent_space(),
                "moving_objects": gym.spaces.Sequence(self._get_object_space()),
                "robot": self._get_coords_space(),
            }
        )

    def _get_agent_space(self) -> gym.spaces.Space:
        """Helper method to create a Dict space (coordinates and presence of a picked object) for the agent"""

        return gym.spaces.Dict(
            {
                "coords": self._get_coords_space(),
                "picked_object": gym.spaces.Discrete(1),
            }
        )

    def _get_coords_space(self) -> gym.spaces.Space:
        """Helper method to create a Box space for the 2D coordinates (row, col) of a board element"""

        return gym.spaces.Box(
            # Coordonates are 1-based
            low=np.array([1, 1]),
            # Maximum values are bounded by board dimensions
            high=np.array(
                [
                    self.config.n_rows,
                    self.config.n_cols,
                ]
            ),
            dtype=np.int64,
        )

    def _get_object_space(self) -> gym.spaces.Space:
        """Helper method to create a Dict space for the properties of a board object"""

        return gym.spaces.Dict(
            {
                "coords": self._get_coords_space(),
                "color": gym.spaces.Discrete(n=len(Color)),
                "shape": gym.spaces.Discrete(n=len(Shape)),
            }
        )

    @property
    def collision_penalty(self) -> bool:
        """Return penalty mode status: are arms in penalty mode after a collision?"""

        return (
            self.board.agent_arm.collision_penalty
            or self.board.robot_arm.collision_penalty
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        # Init the RNG
        super().reset(seed=seed, options=options)

        # Reset episode metrics
        self.n_removed_objects = 0
        self.n_collisions = 0
        self.agent_episode_reward = 0
        self.robot_episode_reward = 0

        self.board.reset()

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return (self._get_obs(), self._get_info())

    def _get_obs(self) -> dict:
        """
        Return an observation given to the agent.

        An observation is a dictionary containing:
        - the properties of the agent (coordinates of arm gripper and presence of a picked object)
        - the properties of all moving objects
        - the coordinates of robot arm gripper
        """

        # Get list of moving objects
        objects = tuple(obj.get_props() for obj in self.board.moving_objects)

        return {
            "self": {
                "coords": self.board.agent_arm.gripper.coords.as_vector(),
                "picked_object": int(self.board.agent_arm.picked_object is not None),
            },
            "moving_objects": objects,
            "robot": self.board.robot_arm.gripper.coords.as_vector(),
        }

    def _get_info(self) -> dict:
        """Return additional information given to the agent"""

        return {
            "n_collisions": self.n_collisions,
            "n_placed_objects": self.board.agent_arm.n_placed_objects,
        }

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        # Init step reward for agent and robot
        agent_reward: float = self.config.step_reward
        robot_reward: float = self.config.step_reward

        # Apply robot action.
        # Robot can move or pick only if it is not currently moving back to its base
        if self.config.robot_enabled and self.robot is not None:
            robot_action = (
                self.robot.choose_action()
                if not self.robot.arm.moving_back
                else Action.NONE
            )
            robot_collision, robot_placed_object, robot_picked_object = (
                self.board.robot_arm.act(
                    action=robot_action,
                    objects=self.board.objects,
                    other_arm=self.board.agent_arm,
                )
            )
        else:
            robot_action = Action.NONE
            robot_collision, robot_placed_object, robot_picked_object = (
                False,
                None,
                None,
            )

        # Apply agent action.
        # Agent can move or pick only if it is not currently moving back to its base
        agent_action = (
            Action(action) if not self.board.agent_arm.moving_back else Action.NONE
        )
        agent_collision, agent_placed_object, agent_picked_object = (
            self.board.agent_arm.act(
                action=agent_action,
                objects=self.board.objects,
                other_arm=self.board.robot_arm if self.config.robot_enabled else None,
            )
        )

        # Compute movement penalties
        if robot_action in (Action.UP, Action.DOWN):
            robot_reward += self.config.movement_penalty
        if agent_action in (Action.UP, Action.DOWN):
            agent_reward += self.config.movement_penalty

        # Handle collisions
        if robot_collision or agent_collision:
            self.n_collisions += 1

            # Drop any picked object in case of a collision.
            # Any dropped object is removed from the board
            if self.board.robot_arm.picked_object:
                self.board.robot_arm._picked_object.empty()
                self.n_removed_objects += 1
            if self.board.agent_arm.picked_object:
                self.board.agent_arm._picked_object.empty()
                self.n_removed_objects += 1

            # Compute negative rewards for the collision
            agent_reward += self.config.collision_penalty
            robot_reward += self.config.collision_penalty
        else:
            # Handle robot object
            if robot_placed_object is not None:
                # Robot arm has placed an object: move it to score bar
                self.board.robot_scorebar.add(placed_object=robot_placed_object)
                # Increment number of objects removed from the board
                self.n_removed_objects += 1
            elif robot_picked_object is not None:
                # Compute robot reward
                robot_reward += robot_picked_object.get_reward(
                    rewards=self.config.robot_rewards
                )

            # Handle agent object
            if agent_placed_object is not None:
                # Agent arm has placed an object: move it to score bar
                self.board.agent_scorebar.add(placed_object=agent_placed_object)
                # Increment number of objects removed from the board
                self.n_removed_objects += 1
            elif agent_picked_object is not None:
                # Compute agent reward
                agent_reward += agent_picked_object.get_reward(
                    rewards=self.config.agent_rewards
                )

        # Update world state
        self.n_removed_objects += self.board.animate()
        self.agent_episode_reward += agent_reward
        self.robot_episode_reward += robot_reward

        observation = self._get_obs()
        info = self._get_info()

        # Episode is terminated when all objects have either been placed by an arm or have fallen from a treadmill
        terminated = (
            self.n_removed_objects >= self.config.n_objects
            and self.board.agent_arm.picked_object is None
            and self.board.robot_arm.picked_object is None
        )

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return observation, agent_reward, terminated, False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == RenderMode.RGB_ARRAY:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        """Render the current state of the environment as a frame"""

        canvas = self.board.draw(
            agent_reward=self.agent_episode_reward,
            robot_reward=self.robot_episode_reward,
            collision_count=self.n_collisions,
            collision_penalty=self.collision_penalty,
        )

        if self.render_mode == RenderMode.HUMAN:
            if self.window is None:
                # Init pygame display
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    size=self.config.window_dimensions
                )
                pygame.display.set_caption(self.config.window_title)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            # The following line copies our drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.config.render_fps)

        else:  # rgb_array
            return self.board.get_frame()

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
            pygame.quit()
            pygame.quit()
