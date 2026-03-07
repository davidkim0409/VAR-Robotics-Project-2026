"""Drone class for swarm simulation."""

import numpy as np


class Drone:
    """
    Represents a single drone in the swarm with basic movement and communication.
    """

    def __init__(self, position: np.ndarray | list, index: int):
        self.position = np.array(position, dtype=np.float64)
        self.index = index
        self.target_position = np.array(position, dtype=np.float64)

    def update_position(
        self,
        neighbor_positions: list[np.ndarray],
        behavior_algorithms: list,
    ) -> None:
        """Update position by averaging results of all behavior algorithms."""
        new_positions = [
            algo.apply(self, neighbor_positions, self.position.copy())
            for algo in behavior_algorithms
        ]
        self.position = np.mean(new_positions, axis=0)

    def communicate(self) -> np.ndarray:
        """Return current position for neighbor communication."""
        return self.position.copy()

    def get_position(self) -> np.ndarray:
        """Return current position."""
        return self.position.copy()
