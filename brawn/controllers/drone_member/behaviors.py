"""Swarm behavior algorithms: consensus, collision avoidance, formation control."""

import numpy as np


class ConsensusAlgorithm:
    """Moves drone toward average neighbor position for cohesion."""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def apply(
        self,
        drone,
        neighbor_positions: list[np.ndarray],
        current_position: np.ndarray,
    ) -> np.ndarray:
        if len(neighbor_positions) == 0:
            return current_position.copy()
        mean_neighbor = np.mean(neighbor_positions, axis=0)
        return current_position + self.epsilon * (mean_neighbor - current_position)


class CollisionAvoidanceAlgorithm:
    """Pushes drone away from neighbors below collision threshold."""

    def __init__(self, collision_threshold: float):
        self.collision_threshold = collision_threshold

    def apply(
        self,
        drone,
        neighbor_positions: list[np.ndarray],
        current_position: np.ndarray,
    ) -> np.ndarray:
        pos = current_position.copy()
        for neighbor_pos in neighbor_positions:
            dist = np.linalg.norm(pos - neighbor_pos)
            if dist > 0 and dist < self.collision_threshold:
                direction = (pos - neighbor_pos) / dist
                pos += direction * (self.collision_threshold - dist)
        return pos


class CustomFormationAlgorithm:
    """Moves each drone toward its target position from brain coords (data/coordinates/*_coords.json)."""

    def __init__(
        self,
        target_positions: list[np.ndarray],
        step_size: float = 0.1,
    ):
        self.target_positions = [np.array(t, dtype=np.float64) for t in target_positions]
        self.step_size = step_size

    def apply(
        self,
        drone,
        neighbor_positions: list[np.ndarray],
        current_position: np.ndarray,
    ) -> np.ndarray:
        if drone.index >= len(self.target_positions):
            return current_position.copy()
        target = self.target_positions[drone.index]
        direction = target - current_position
        return current_position + self.step_size * direction
