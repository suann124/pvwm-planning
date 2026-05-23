"""Terrain path planning with flood-level obstacles.

Loads the base terrain, then overlays one of the flood PLYs as additional
obstacles.  Flood Gaussians whose Z exceeds `FLOOD_Z_THRESHOLD` are projected
onto the occupancy grid and marked impassable before planning.

Usage:
    python demos/hillcounty_flood_planning.py           # default: medium flood
    python demos/hillcounty_flood_planning.py low
    python demos/hillcounty_flood_planning.py high
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from foci.visualisation.vis_utils import ViserVis
from foci.utils.ply import extract_splat_data, filter_floaters, load_flood_data
from foci.utils.terrain import build_height_map, query_height
from foci.planners.planner_terrain import TerrainPlanner

LOCAL = os.path.dirname(os.path.abspath(__file__))

FLOOD_LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'medium'
assert FLOOD_LEVEL in ('low', 'medium', 'high'), "level must be low / medium / high"

TERRAIN_PLY = os.path.join(LOCAL, 'data/hillcounty_sm_30000.ply')
FLOOD_PLY   = os.path.join(LOCAL, f'data/hillcountry_flood_{FLOOD_LEVEL}.ply')

# ── 1. Terrain (identical pipeline to hillcounty_terrain.py) ─────────────────
print("Loading terrain...")
means, covs, colors, opacities = extract_splat_data(TERRAIN_PLY)
# means, covs, colors, opacities = filter_floaters(
#     means, covs, colors, opacities,
#     opacity_threshold=0.3,
#     density_radius=0.05,
#     min_neighbors=5,
# )

ground_z = np.percentile(means[:, 2], 6)
print(f"Ground z (pre-scale): {ground_z:.4f}")
means[:, 2] -= ground_z

scale = 20
means = scale * means
covs  = scale**2 * covs

print("Scene bounds (ground=0, scaled):")
print(f"  X: {means[:,0].min():.1f} .. {means[:,0].max():.1f}")
print(f"  Y: {means[:,1].min():.1f} .. {means[:,1].max():.1f}")
print(f"  Z: {means[:,2].min():.1f} .. {means[:,2].max():.1f}")

colors = np.clip(colors, 0.0, 1.0)   # SH-derived colors can exceed [0,1]

start_xy = [11.0, -18.0]
end_xy   = [20.0,  10.0]

z_var = covs[:, 2, 2]
flat_threshold = np.percentile(z_var, 30)
flat_mask = z_var < flat_threshold
print(f"Flat Gaussians: {flat_mask.sum()} / {len(means)}")

# Build height map from flat Gaussians — ground surface Z per cell.
height_map, hm_x_min, hm_y_min, hm_cell = build_height_map(means[flat_mask], cell_size=0.5)
print(f"Height map: {height_map.shape}")

# Two separate height parameters:
#   obs_threshold — a Gaussian this far above the local terrain surface is an
#                   obstacle (trees, walls, flood).  Kept small so low vegetation
#                   is included.
#   clearance     — the robot's navigation height above terrain.  Must satisfy
#                   clearance > obs_threshold + voxel_size + margin so the robot
#                   flies above the top of the obstacle voxels and never starts
#                   inside one.  (voxel_size = 0.25 → need clearance > ~0.6)
obs_threshold = 0.3   # scaled units
clearance     = 1   # scaled units (navigates ~50 cm above terrain, ~25 cm above tallest obs voxel top)

start_z_terrain = query_height(height_map, hm_x_min, hm_y_min, hm_cell, start_xy[0], start_xy[1])
end_z_terrain   = query_height(height_map, hm_x_min, hm_y_min, hm_cell, end_xy[0],   end_xy[1])
nav_z = 0.5 * (start_z_terrain + end_z_terrain) + clearance
print(f"Terrain Z  — start: {start_z_terrain:.3f}, end: {end_z_terrain:.3f}")
print(f"obs_threshold: {obs_threshold}, clearance: {clearance}, nav_z: {nav_z:.3f}")

# ── 2. Load flood data and build flood obstacle mask ─────────────────────────
print(f"\nLoading flood PLY: {FLOOD_LEVEL}...")
flood_means, flood_covs, flood_colors, flood_opacities = load_flood_data(FLOOD_PLY, ground_z, scale)
print(f"Flood {FLOOD_LEVEL} (scaled): "
      f"X={flood_means[:,0].min():.1f}..{flood_means[:,0].max():.1f}  "
      f"Y={flood_means[:,1].min():.1f}..{flood_means[:,1].max():.1f}  "
      f"Z={flood_means[:,2].min():.1f}..{flood_means[:,2].max():.1f}  "
      f"n={len(flood_means)}")

# ── 2b. Build flood surface map + dense flood obstacle carpet ─────────────────
# Map each flood Gaussian into the terrain grid.
fxi = np.clip(((flood_means[:, 0] - hm_x_min) / hm_cell).astype(int), 0, height_map.shape[0] - 1)
fyi = np.clip(((flood_means[:, 1] - hm_y_min) / hm_cell).astype(int), 0, height_map.shape[1] - 1)

# flood_surface[i, j] = max flood Z seen in that cell
flood_surface = np.full(height_map.shape, -np.inf)
np.maximum.at(flood_surface, (fxi, fyi), flood_means[:, 2])
flood_surface[flood_surface == -np.inf] = np.nan

# Flooded cells = water sitting meaningfully above terrain
flood_depth = flood_surface - height_map
flood_mask = (~np.isnan(flood_surface)) & (flood_depth > 0.02)   # tune threshold

print(f"Flood cells in grid: {flood_mask.sum()}")

# Build one obstacle Gaussian per flooded grid cell.
flood_cells = np.argwhere(flood_mask)   # rows are [ix, iy]

flood_means_nav = np.zeros((len(flood_cells), 3), dtype=float)
flood_covs_nav = np.zeros((len(flood_cells), 3, 3), dtype=float)

for k, (ix, iy) in enumerate(flood_cells):
    # Cell center in world coordinates
    x = hm_x_min + (ix + 0.5) * hm_cell
    y = hm_y_min + (iy + 0.5) * hm_cell

    # Put flood obstacle at the ground robot navigation plane
    z = height_map[ix, iy] + clearance

    flood_means_nav[k] = [x, y, z]

    # Dense lateral footprint so neighboring cells overlap and form a carpet.
    # Tune these if needed.
    sigma_xy = 0.75 * hm_cell
    sigma_z  = 0.05

    flood_covs_nav[k] = np.diag([
        sigma_xy**2,
        sigma_xy**2,
        sigma_z**2,
    ])

print(f"Flood Gaussians kept as nav-obstacles: {len(flood_means_nav)}")

# ── 3. Build obstacle mask ────────────────────────────────────────────────────

# Terrain Gaussians: obstacle if above local ground + obs_threshold
xi_all = np.clip(((means[:, 0] - hm_x_min) / hm_cell).astype(int), 0, height_map.shape[0] - 1)
yi_all = np.clip(((means[:, 1] - hm_y_min) / hm_cell).astype(int), 0, height_map.shape[1] - 1)
local_ground = height_map[xi_all, yi_all]
obstacle_mask = means[:, 2] > local_ground + obs_threshold

print(f"Terrain obstacles: {obstacle_mask.sum()}, flood nav-obstacles: {len(flood_means_nav)}")

# Corridor crop
corridor = 20
x_lo = min(start_xy[0], end_xy[0]) - corridor
x_hi = max(start_xy[0], end_xy[0]) + corridor
y_lo = min(start_xy[1], end_xy[1]) - corridor
y_hi = max(start_xy[1], end_xy[1]) + corridor

in_corridor = (
    (means[:, 0] > x_lo) & (means[:, 0] < x_hi) &
    (means[:, 1] > y_lo) & (means[:, 1] < y_hi)
)
fl_in_corridor = (
    (flood_means_nav[:, 0] > x_lo) & (flood_means_nav[:, 0] < x_hi) &
    (flood_means_nav[:, 1] > y_lo) & (flood_means_nav[:, 1] < y_hi)
)

# No global z_range: A* follows terrain dynamically via the height_map band,
# and the post-processing step snaps the path Z to terrain + clearance.
z_range = None

obstacle_means = np.vstack([
    means[obstacle_mask & in_corridor],
    flood_means_nav[fl_in_corridor],
])

obstacle_covs = np.vstack([
    covs[obstacle_mask & in_corridor] * 0.01,
    flood_covs_nav[fl_in_corridor],
])

print(f"Corridor obstacles: {len(obstacle_means)}")

# ── 4. Height-map PNG with flood overlay ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
im = ax.imshow(height_map.T, origin='lower', cmap='terrain',
               extent=[hm_x_min, hm_x_min + height_map.shape[0] * hm_cell,
                       hm_y_min, hm_y_min + height_map.shape[1] * hm_cell])
plt.colorbar(im, ax=ax, label='Z height (scaled)')
ax.set_title(f'Terrain height map + flood ({FLOOD_LEVEL})')
ax.set_xlabel('X'); ax.set_ylabel('Y')

ax.scatter(flood_means[:, 0], flood_means[:, 1],
           c='cyan', s=1, alpha=0.4, label=f'flood {FLOOD_LEVEL}')
ax.scatter(*start_xy, c='red',  s=100, zorder=5, label='start')
ax.scatter(*end_xy,   c='blue', s=100, zorder=5, label='end')
ax.legend(markerscale=4)

ax2 = axes[1]
ax2.imshow(height_map.T, origin='lower', cmap='terrain',
           extent=[hm_x_min, hm_x_min + height_map.shape[0] * hm_cell,
                   hm_y_min, hm_y_min + height_map.shape[1] * hm_cell])
ax2.scatter(flood_means[:, 0], flood_means[:, 1],
            c='cyan', s=1, alpha=0.6)
ax2.set_title(f'Flood obstacles above terrain ({FLOOD_LEVEL})')
ax2.set_xlabel('X'); ax2.set_ylabel('Y')
ax2.scatter(*start_xy, c='red',  s=100, zorder=5)
ax2.scatter(*end_xy,   c='blue', s=100, zorder=5)

plt.tight_layout()
plt.savefig(f'flood_planning_{FLOOD_LEVEL}.png', dpi=150)
print(f"Saved flood_planning_{FLOOD_LEVEL}.png")
plt.close()

# ── 5. Plan ───────────────────────────────────────────────────────────────────
robot_cov = np.eye(3) * 0.0001
planner = TerrainPlanner(obstacle_means, obstacle_covs, robot_cov,
                         num_control_points=10, num_samples=50, z_range=z_range)

start = [start_xy[0], start_xy[1], start_z_terrain + clearance, np.pi/2]
end   = [end_xy[0],   end_xy[1],   end_z_terrain   + clearance, np.pi/2]


opt_curve, astar = planner.plan(
    start, end,
    height_map=height_map,
    hm_x_min=hm_x_min,
    hm_y_min=hm_y_min,
    hm_cell=hm_cell,
    clearance=clearance,
    z_band=0.15,   # robot navigates within 15 cm band above local terrain
)

print(f"A* path points: {len(astar)}")

for i in range(len(opt_curve)):
    x, y = opt_curve[i, 0], opt_curve[i, 1]
    opt_curve[i, 2] = query_height(height_map, hm_x_min, hm_y_min, hm_cell, x, y) + clearance

# ── 6. Visualise ──────────────────────────────────────────────────────────────
margin = 30
x_min_v, y_min_v = astar[:, :2].min(axis=0) - margin
x_max_v, y_max_v = astar[:, :2].max(axis=0) + margin
crop_mask = (
    (means[:, 0] > x_min_v) & (means[:, 0] < x_max_v) &
    (means[:, 1] > y_min_v) & (means[:, 1] < y_max_v)
)

vis = ViserVis()
vis.add_gaussians(means[crop_mask], covs[crop_mask],
                  color=colors[crop_mask], opacity=opacities[crop_mask])

# Show flood region as a cyan point cloud (all flood Gaussians, no opacity filter)
FLOOD_RGB = np.array([[0.0, 0.8, 1.0]] * len(flood_means))
vis.server.scene.add_point_cloud(
    f"flood_{FLOOD_LEVEL}",
    flood_means,
    colors=FLOOD_RGB,
    point_size=0.1,
)

vis.add_curve(astar[:, :3], color=[0, 1, 0], name="astar")
vis.add_gaussian_path(opt_curve, robot_cov, planner.kinematics,
                      color=[0, 1, 0], name="opt_curve")
vis.server.scene.add_label("start", text=f"S({start[0]:.1f},{start[1]:.1f},{start[2]:.1f})", position=astar[0, :3])
vis.server.scene.add_label("end",   text=f"E({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})", position=astar[-1, :3])
vis.show()
