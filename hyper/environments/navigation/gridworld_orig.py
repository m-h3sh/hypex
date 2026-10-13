"""
Original GridNavi environment from the paper.
"""
from environments.navigation.gridworld import GridNavi, goal_positions


class GridNaviOrig(GridNavi):
    def __init__(
        self,
        num_cells=6,
        num_steps=20,
        starting_state=(0.0, 0.0),
        distance_reward=False,
        task_dist=None,
        possible_goals=None,
        **kwargs,
    ):
        if task_dist is not None and possible_goals is not None:
            raise ValueError("Pass only one of possible_goals or task_dist")

        if task_dist is not None:
            possible_goals = goal_positions(int(task_dist), num_cells)
            # Corner start can coincide with some TaskDist goals; drop exact start.
            if starting_state is not None:
                sx, sy = int(starting_state[0]), int(starting_state[1])
                possible_goals = [g for g in possible_goals if g != (sx, sy)]
            if not possible_goals:
                raise ValueError(
                    f"task_dist={task_dist} left no goals after removing start {starting_state}"
                )

        super().__init__(
            num_cells=num_cells,
            num_steps=num_steps,
            starting_state=starting_state,
            distance_reward=distance_reward,
            possible_goals=possible_goals,
            task_dist=None,
            **kwargs,
        )
