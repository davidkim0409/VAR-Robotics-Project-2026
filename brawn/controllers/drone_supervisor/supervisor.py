"""
Supervisor: load brain coords from data/coordinates, n×n grid init,
run swarm (collision_avoidance, consensus, formation_control), optional Unity UDP.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

from ..drone_member import (
    Drone,
    ConsensusAlgorithm,
    CollisionAvoidanceAlgorithm,
    CustomFormationAlgorithm,
)

# Brain output is 2D; formation is on one plane. YZ plane: X fixed.
X_PLANE = 0.0
# Single color for all drones (brain has no layers).
DEFAULT_COLOR = (0.0, 1.0, 1.0)  # cyan, matches brain visualize_sampling


def _project_root() -> Path:
    """Project root: brawn/controllers/drone_supervisor -> three levels up."""
    return Path(__file__).resolve().parents[3]


def load_coords_brain(path: str | Path) -> tuple[list[np.ndarray], list[tuple[float, float, float]]]:
    """
    Load VAR brain coords from data/coordinates/*_coords.json.
    Returns (target_positions_3d, colors). Points on YZ plane (X=X_PLANE).
    Image (x, y) -> 3D (X_PLANE, x, -y); Z shifted so all Z > 0.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("points", [])
    target_positions = []
    colors = []
    for pt in points:
        x, y = float(pt["x"]), float(pt["y"])
        # YZ plane: (X, Y, Z) = (constant, x, -y)
        target_positions.append(np.array([X_PLANE, x, -y], dtype=np.float64))
        colors.append(DEFAULT_COLOR)
    # Shift Z so all coordinates are > 0
    z_min = min(p[2] for p in target_positions)
    z_offset = (1.0 - z_min) if z_min < 1.0 else 0.0
    for p in target_positions:
        p[2] += z_offset
    return target_positions, colors


def compute_grid_positions(
    n: int,
    spacing: float,
    grid_origin: tuple[float, float, float],
) -> list[np.ndarray]:
    """Return n×n grid positions on XY plane (Z constant) as list of (x, y, z) arrays."""
    positions = []
    for i in range(n * n):
        col = i % n
        row = i // n
        x = grid_origin[0] + col * spacing
        y = grid_origin[1] + row * spacing
        z = grid_origin[2]  # constant (XY plane)
        positions.append(np.array([x, y, z], dtype=np.float64))
    return positions


def run(args: argparse.Namespace) -> None:
    target_positions, colors = load_coords_brain(args.coords)
    if not target_positions:
        print("No positions to simulate.", file=sys.stderr)
        sys.exit(1)

    N = len(target_positions)
    colors_arr = np.array(colors)

    # Grid on XY plane (Z=0): n = ceil(sqrt(N)), origin in front of formation (negative X)
    min_y = min(p[1] for p in target_positions)
    n = math.ceil(math.sqrt(N))
    grid_origin = (-n * args.grid_spacing, min_y - n * args.grid_spacing, 0.0)
    grid_positions = compute_grid_positions(n, args.grid_spacing, grid_origin)[:N]

    # Drones
    drones = [Drone(grid_positions[i], i) for i in range(N)]
    for i, d in enumerate(drones):
        d.target_position = target_positions[i]

    # Algorithms
    behaviors = [
        ConsensusAlgorithm(args.epsilon),
        CollisionAvoidanceAlgorithm(args.collision_threshold),
        CustomFormationAlgorithm(target_positions, args.step_size),
    ]

    # UDP socket for Unity (optional)
    udp_socket = None
    unity_addr = None
    if args.unity_host:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unity_addr = (args.unity_host, args.unity_port)
        print(f"UDP send to Unity: {args.unity_host}:{args.unity_port}", file=sys.stderr)

    def send_positions_to_unity() -> None:
        if udp_socket is None or unity_addr is None:
            return
        positions = [d.get_position() for d in drones]
        payload = {
            "drones": [
                {
                    "id": i,
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "r": float(colors_arr[i, 0]),
                    "g": float(colors_arr[i, 1]),
                    "b": float(colors_arr[i, 2]),
                }
                for i, p in enumerate(positions)
            ]
        }
        try:
            udp_socket.sendto(
                json.dumps(payload).encode("utf-8"),
                unity_addr,
            )
        except OSError as e:
            print(f"UDP send error: {e}", file=sys.stderr)

    def simulation_step() -> None:
        for drone in drones:
            neighbor_positions = [
                other.communicate()
                for other in drones
                if other != drone
            ]
            drone.update_position(neighbor_positions, behaviors)

    if args.unity_only:
        if not args.unity_host:
            print("Error: --unity-only requires --unity-host", file=sys.stderr)
            sys.exit(1)
        interval_sec = args.interval / 1000.0
        frame_count = 0
        max_frames = args.frames if args.frames is not None else 300
        if args.frames is None:
            print(
                f"Running {max_frames} frames (default). Use --frames N to override.",
                file=sys.stderr,
            )
        try:
            while frame_count < max_frames:
                simulation_step()
                send_positions_to_unity()
                frame_count += 1
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            pass
        if udp_socket:
            udp_socket.close()
        return

    # Matplotlib 3D animation
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.view_init(elev=20, azim=-35)

    positions = np.array([d.get_position() for d in drones])
    scat = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=colors_arr,
        s=8,
        alpha=0.9,
    )

    margin = 30
    all_pos = np.vstack([np.array(grid_positions), np.array(target_positions)])
    min_pos = np.min(all_pos, axis=0)
    max_pos = np.max(all_pos, axis=0)
    center = (min_pos + max_pos) / 2
    range_ = max(np.max(max_pos - min_pos) / 2, margin)
    ax.set_xlim(center[0] - range_, center[0] + range_)
    ax.set_ylim(center[1] - range_, center[1] + range_)
    ax.set_zlim(center[2] - range_, center[2] + range_)

    def update_view() -> None:
        positions = np.array([d.get_position() for d in drones])
        min_p = np.min(positions, axis=0)
        max_p = np.max(positions, axis=0)
        c = (min_p + max_p) / 2
        r = max(np.max(max_p - min_p) / 2, margin)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)

    def animate(frame: int) -> tuple:
        simulation_step()
        positions = np.array([d.get_position() for d in drones])
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        update_view()
        send_positions_to_unity()
        return (scat,)

    fps = 1000 / args.interval
    save_count = args.frames
    if args.save_gif and save_count is None:
        save_count = 300
    anim = animation.FuncAnimation(
        fig,
        animate,
        interval=args.interval,
        blit=False,
        save_count=save_count,
        cache_frame_data=False,
    )

    try:
        if args.save_gif:
            print(f"Writing GIF to {args.save_gif}...")
            anim.save(args.save_gif, writer="pillow", fps=fps)
            print("Done.")
            plt.close(fig)
        else:
            plt.show()
    finally:
        if udp_socket is not None:
            udp_socket.close()


def main() -> None:
    root = _project_root()
    default_coords = root / "data" / "coordinates" / "apple_coords.json"

    parser = argparse.ArgumentParser(
        description="Brawn: load brain coords from data/coordinates, n×n grid -> formation (consensus, collision, formation)."
    )
    parser.add_argument(
        "coords",
        nargs="?",
        default=str(default_coords),
        help="Path to brain coords JSON (e.g. data/coordinates/apple_coords.json)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Simulation update interval (ms)",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=5.0,
        help="Initial n×n grid spacing (wider = more space between drones)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.08,
        help="Formation step size",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Consensus epsilon",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=1.5,
        help="Collision avoidance threshold",
    )
    parser.add_argument(
        "--save-gif",
        metavar="FILE",
        default=None,
        help="Save animation as GIF",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Max animation frames (default: 300 when saving GIF or --unity-only, infinite otherwise)",
    )
    parser.add_argument(
        "--unity-host",
        metavar="HOST",
        default=None,
        help="Send drone positions to Unity via UDP (e.g. 127.0.0.1). Omit to disable.",
    )
    parser.add_argument(
        "--unity-port",
        type=int,
        default=6060,
        help="UDP port for Unity (default: 6060)",
    )
    parser.add_argument(
        "--unity-only",
        action="store_true",
        help="Run simulation without matplotlib; only send to Unity via UDP",
    )
    args = parser.parse_args()

    path = Path(args.coords)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        print(f"Error: coords file not found: {path}", file=sys.stderr)
        sys.exit(1)
    args.coords = str(path)

    run(args)


if __name__ == "__main__":
    main()
