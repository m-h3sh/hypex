import itertools
import random

import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from gymnasium import spaces
from PIL import Image
from io import BytesIO


def goal_positions(option, num_cells):
    """Task distributions from exp1/exp2 notebooks (4 goals each)."""
    if option == 1:
        # top-right corner cluster
        pos = [(num_cells - 1, num_cells - 1), (num_cells - 1, num_cells - 2),
               (num_cells - 2, num_cells - 1), (num_cells - 2, num_cells - 2)]
    elif option == 2:
        # mid / edge spread
        pos = [(num_cells // 2, num_cells // 2), (num_cells - 1, num_cells // 2),
               (num_cells // 2, num_cells - 1), (num_cells - 1, num_cells - 1)]
    elif option == 3:
        # four corners
        pos = [(0, 0), (0, num_cells - 1), (num_cells - 1, 0), (num_cells - 1, num_cells - 1)]
    elif option == 4:
        # bottom-left corner cluster
        pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    elif option == 5:
        # top-left corner cluster
        pos = [(0, num_cells - 1), (0, num_cells - 2), (1, num_cells - 1), (1, num_cells - 2)]
    elif option == 6:
        # bottom-right corner cluster
        pos = [(num_cells - 1, 0), (num_cells - 1, 1), (num_cells - 2, 0), (num_cells - 2, 1)]
    else:
        raise ValueError(f"Unknown goal_positions option: {option}")
    return pos


class GridNavi(gym.Env):
    def __init__(self, num_cells=5, num_steps=15, starting_state=(0.0, 0.0), ring=False, hall_with_door_every=None,
                 new_state_r=None, stuck=False, distance_reward=False, show_goal_at_start=False, possible_goals=None,
                 task_dist=None):
        if starting_state is None:
            assert not ring and hall_with_door_every is None, "Random start location not supported for ring or hall"
        super(GridNavi, self).__init__()

        if task_dist is not None:
            if possible_goals is not None:
                raise ValueError("Pass only one of possible_goals or task_dist")
            possible_goals = goal_positions(int(task_dist), num_cells)

        self.num_cells = num_cells
        self.num_states = num_cells ** 2 # should be  num_cells * 2 if hallway, but this messes up visualization 
        self.new_state_r = new_state_r # reward for looking at a potential goal
        self.distance_reward = distance_reward
        self.show_goal_at_start = show_goal_at_start

        self._max_episode_steps = num_steps
        self.step_count = 0

        self.hall_with_door_every = hall_with_door_every # whether to train in a hallway with doors (potential goal) every x steps
        self.stuck = stuck # Whether the agent gets "stuck" behind the door they enter

        self.observation_space = spaces.Box(
            low=0, high=self.num_cells - 1, shape=(2,), dtype=np.float32
        ) if hall_with_door_every is None else spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.num_cells - 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.task_dim = 2

        # possible starting states
        self.starting_state = starting_state

        if starting_state is not None:
            sx, sy = starting_state
            sx, sy = int(sx), int(sy)
        # define possible goal locations
        if possible_goals is not None:
            self.possible_goals = [tuple(g) for g in possible_goals]
        elif ring: #goals on perimeter
            assert hall_with_door_every is None, "Cannot have ring and hallway"
            self.possible_goals = set()
            for offset in range(num_cells):
                self.possible_goals.add((0, offset)) #left side
                self.possible_goals.add((num_cells-1, offset)) #right side
                self.possible_goals.add((offset, num_cells-1)) #top
                self.possible_goals.add((offset, 0)) #bottom
            if (sx, sy) in self.possible_goals:
                self.possible_goals.remove((sx, sy))
            self.possible_goals = list(self.possible_goals)
        elif hall_with_door_every is not None:
            assert starting_state[1] == 0, "Must start in hall if hall_with_door_every is not None"
            self.possible_goals = []
            for offset in range(hall_with_door_every, num_cells, hall_with_door_every): # door every x steps, but not at start
                self.possible_goals.append((offset, 1))
        else:
            self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
            if starting_state is not None:
                # goals can be anywhere except on possible starting states and immediately around it
                for x_offset in [-1, 0, 1]:
                    for y_offset in [-1, 0, 1]:
                        pos = (sx+x_offset, sy+y_offset)
                        if pos in self.possible_goals:
                            self.possible_goals.remove(pos)

        # keep a list of places the agent has not checked yet
        self.goals_unchecked = set(self.possible_goals)

        # reset the environment state
        self._env_state = np.array(
            random.choice(self.possible_goals) if starting_state is None else self.starting_state,
            dtype=np.float32,
        )
        # reset the goal
        self._goal = self.reset_task()

    def reset_task(self, task=None):
        if task is None:
            self._goal = np.array(random.choice(self.possible_goals), dtype=np.float32)
        else:
            self._goal = np.array(task, dtype=np.float32)
        self.goals_unchecked = set(self.possible_goals)
        return self._goal

    def get_task(self):
        return self._goal.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._env_state = np.array(
            random.choice(self.possible_goals) if self.starting_state is None else self.starting_state,
            dtype=np.float32,
        )
        return self._env_state.copy(), {}

    def state_transition(self, action):
        """
        Moving the agent between states
        """
        if self.hall_with_door_every is not None and self._env_state[1] == 1:
            if self.stuck:
                return self._env_state # Cannot go backward once a door is entered!
            elif action != 3:
                return self._env_state # Can only go down once a door is entered!

        if action == 1:  # up
            # can only go up if not in hallway or there is door above agent
            if self.hall_with_door_every is None:
                self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
            elif (int(self._env_state[0]), int(self._env_state[1]+1)) in self.possible_goals:
                self._env_state[1] = self._env_state[1]+1
        elif action == 2:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 4:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])

        return self._env_state

    def step(self, action):

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        terminated = False
        truncated = False

        # perform state transition
        state = self.state_transition(action)

        # keep track of potential goals checked
        if self.new_state_r is not None:
            checked_potential_goal = False
            sx, sy = state
            sx, sy = int(sx), int(sy)
            if (sx, sy) in self.goals_unchecked:
                checked_potential_goal = True
                self.goals_unchecked.remove((sx, sy))

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            truncated = True

        # compute reward
        if self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]:
            reward = 1.0
        elif self.new_state_r is not None and checked_potential_goal:
            reward = self.new_state_r
        elif self.distance_reward:
            reward = (-abs(self._env_state[0] - self._goal[0]) - abs(self._env_state[1] - self._goal[1]))/self.num_cells
        else:
            reward = -0.1

        info = {'task': self.get_task()}

        if self.show_goal_at_start:
            assert self.starting_state is not None, "Start state must be specified to show goal there"
            if self._env_state[0] == self.starting_state[0] and self._env_state[1] == self.starting_state[1]:
                state = self._goal[:]  # show goal as obs

        return state.copy(), reward, terminated, truncated, info

    def render(self):
        num_cells_x = int(self.observation_space.high[0] + 1)
        num_cells_y = int(self.observation_space.high[1] + 1)

        state = self._env_state.astype(int)
        goal = self._goal.astype(int)

        plt.figure(figsize=(5, 5))
        plt.axis('off')

        # Draw grid
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                rec = Rectangle((i, j), 1, 1, facecolor='w', alpha=0.5, edgecolor='k')
                plt.gca().add_patch(rec)

        # Plot state and goal with larger markers and high z-order
        plt.plot(state[0] + .5, num_cells_y - state[1] - 1 + .5, 'bo', markersize=10, zorder=5)
        plt.plot(goal[0] + .5, num_cells_y - goal[1] - 1 + .5, 'gx', markersize=15, zorder=5)

        # Make it look nice
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, num_cells_x])
        plt.ylim([num_cells_y, 0])  # Invert the y-axis limits

        # Save plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()

        # Open the image from the buffer and convert to RGB
        buffer.seek(0)
        image = Image.open(buffer)
        image_rgb = np.array(image.convert('RGB'))

        return image_rgb

    @staticmethod
    def goal_entropy(possible_goals):
        """Shannon entropy H(p(g)) for uniform p over the goal support."""
        n = len(possible_goals)
        if n <= 1:
            return 0.0
        p = 1.0 / n
        return -n * p * np.log(p)