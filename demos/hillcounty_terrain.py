import os
import numpy as np
import matplotlib.pyplot as plt

from foci.visualisation.vis_utils import ViserVis
from foci.utils.ply import extract_splat_data, filter_floaters
from foci.utils.terrain import build_height_map, query_height
from foci.planners.planner_terrain import TerrainPlanner

LOCAL = os.path.dirname(os.path.abspath(__file__))
ply_file = os.path.join(LOCAL, 'data/hillcounty_sm_30000.ply')

means, covs, colors, opacities = extract_splat_data(ply_file)
means, covs, colors, opacities = filter_floaters(
    means, covs, colors, opacities,
    opacity_threshold=0.3,   # more lenient than 0.5 since density also filters
    density_radius=0.05,     # in raw (pre-scale) units — tune this first
    min_neighbors=5
)

# Normalize Z and scale
ground_z = np.percentile(means[:, 2], 6)
print(f"Ground z cut off before normalization: {ground_z:.4f}")
means[:, 2] -= ground_z

scale = 20
means = scale * means
covs  = scale**2 * covs

print("Scene bounds (ground = 0, scaled):")
print(f"  X: {means[:,0].min():.3f} to {means[:,0].max():.3f}")
print(f"  Y: {means[:,1].min():.3f} to {means[:,1].max():.3f}")
print(f"  Z: {means[:,2].min():.3f} to {means[:,2].max():.3f}")

# Define start/end XY first (scaled coordinates)
start_xy = [-30.0, -18.0]
end_xy   = [-20.0,  10.0]

def local_ground_z(means, x, y, radius=5.0, percentile=30):
    """Estimate terrain height at (x,y) from nearby Gaussians."""
    dists = np.sqrt((means[:, 0] - x)**2 + (means[:, 1] - y)**2)
    nearby = means[dists < radius]
    if len(nearby) == 0:
        return 0.0
    return np.percentile(nearby[:, 2], percentile)

# --- Covariance-based ground separation ---
# Flat Gaussians (small cov[2,2]) = ground surface; tall/spherical = bushes/vegetation
z_var = covs[:, 2, 2]
flat_threshold = np.percentile(z_var, 30)  # keep the 30% flattest; tune if needed
flat_mask = z_var < flat_threshold
print(f"Flat Gaussians (ground candidates): {flat_mask.sum()} / {len(means)}")

h_start = local_ground_z(means[flat_mask], *start_xy)
h_end   = local_ground_z(means[flat_mask], *end_xy)
nav_z   = max(h_start, h_end) + 0.1
z_range = (nav_z - 0.05, nav_z + 0.10)
print(f"Terrain at start: {h_start:.2f}, end: {h_end:.2f} → nav_z: {nav_z:.2f}, z_range: {z_range}")

# =================================== Height map ======================================
# Only flat Gaussians below nav_z — excludes bush/vegetation Gaussians
# terrain_gaussians = means[(means[:, 2] < ground_threshold) & (means[:, 2] > -2.0)]
terrain_gaussians = means[flat_mask & (means[:, 2] < nav_z)]
height_map, hm_x_min, hm_y_min, hm_cell = build_height_map(terrain_gaussians, cell_size=0.5)
print(f"terrain_gaussians count: {len(terrain_gaussians)}")
print(f"Height map: {height_map.shape}, z range: {np.nanmin(height_map):.2f} to {np.nanmax(height_map):.2f}")

# Save height map image
plt.figure(figsize=(12, 8))
plt.imshow(height_map.T, origin='lower', cmap='terrain',
           extent=[hm_x_min, hm_x_min + height_map.shape[0] * hm_cell,
                   hm_y_min, hm_y_min + height_map.shape[1] * hm_cell])
plt.colorbar(label='Z height (scaled)')
plt.scatter([start_xy[0]], [start_xy[1]], c='red',  s=100, label='start')
plt.scatter([end_xy[0]],   [end_xy[1]],   c='blue', s=100, label='end')
plt.legend()
plt.title('Terrain Height Map')
plt.savefig('height_map.png')
print("Saved height_map.png")

# Side profile along the path (cross-section at x = start_xy[0])
xi_path  = np.clip(int((start_xy[0] - hm_x_min) / hm_cell), 0, height_map.shape[0] - 1)
y_vals   = hm_y_min + np.arange(height_map.shape[1]) * hm_cell
z_profile = height_map[xi_path, :]

plt.figure(figsize=(12, 4))
plt.plot(y_vals, z_profile, color='saddlebrown', linewidth=1.5, label='terrain')
plt.fill_between(y_vals, np.nanmin(z_profile) - 1, z_profile, alpha=0.3, color='saddlebrown')
plt.axhline(nav_z, color='red', linestyle='--', label=f'nav_z={nav_z}')
plt.axvline(start_xy[1], color='green', linestyle=':', label='start Y')
plt.axvline(end_xy[1],   color='blue',  linestyle=':', label='end Y')
plt.xlabel('Y (scaled)')
plt.ylabel('Z height (scaled)')
plt.title(f'Terrain side profile at X={start_xy[0]}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('terrain_profile.png')
print("Saved terrain_profile.png")

# =============================Defining Obstacles ======================================
# Obstacles: Gaussians above the navigation band
# obstacle_mask  = means[:, 2] > z_range[1]
# obstacle_mask = means[:,2] > ground_threshold
# Non-flat Gaussians near nav_z are bushes; flat ones above nav_z are elevated terrain
obstacle_mask = (means[:, 2] > z_range[1]) | (~flat_mask & (means[:, 2] > z_range[1]))

# Crop to corridor — the GPU can't handle 4M+ obstacles (Warp allocates num_samples×num_obstacles matrix)
corridor = 20
x_lo = min(start_xy[0], end_xy[0]) - corridor
x_hi = max(start_xy[0], end_xy[0]) + corridor
y_lo = min(start_xy[1], end_xy[1]) - corridor
y_hi = max(start_xy[1], end_xy[1]) + corridor
corridor_mask = obstacle_mask & (means[:,0] > x_lo) & (means[:,0] < x_hi) & (means[:,1] > y_lo) & (means[:,1] < y_hi)
obstacle_means = means[corridor_mask]
obstacle_covs  = covs[corridor_mask] * 0.001
print(f"Obstacle Gaussians: {obstacle_mask.sum()} total, {corridor_mask.sum()} in corridor")

robot_cov = np.eye(3) * 0.0001
planner = TerrainPlanner(obstacle_means, obstacle_covs, robot_cov,
                        num_control_points=10, num_samples=100, z_range=z_range)

start = [start_xy[0], start_xy[1], nav_z, np.pi/2]
end   = [end_xy[0],   end_xy[1],   nav_z, np.pi/2]
opt_curve, astar = planner.plan(start, end)

print(f"A* path points: {len(astar)}")
print(f"  X: {astar[:,0].min():.2f} to {astar[:,0].max():.2f}")
print(f"  Y: {astar[:,1].min():.2f} to {astar[:,1].max():.2f}")
print(f"  Z: {astar[:,2].min():.2f} to {astar[:,2].max():.2f}")

# =============================Pin path to terrain surface======================================
for i in range(len(opt_curve)):
    x, y = opt_curve[i, 0], opt_curve[i, 1]
    opt_curve[i, 2] = query_height(height_map, hm_x_min, hm_y_min, hm_cell, x, y) + 0.05

# Crop visualization to path region
margin = 10
x_min, y_min = astar[:, :2].min(axis=0) - margin
x_max, y_max = astar[:, :2].max(axis=0) + margin
crop_mask = (
    (means[:, 0] > x_min) & (means[:, 0] < x_max) &
    (means[:, 1] > y_min) & (means[:, 1] < y_max)
)
print(f"Vis Gaussians after crop: {crop_mask.sum()} / {len(means)}")

vis = ViserVis()
vis.add_gaussians(means[crop_mask], covs[crop_mask],
                  color=colors[crop_mask], opacity=opacities[crop_mask])
vis.add_curve(astar[:, :3], color=[0, 1, 0], name="astar")
vis.add_gaussian_path(opt_curve, robot_cov, planner.kinematics,
                      color=[0, 1, 0], name="opt_curve")
vis.server.scene.add_label("start", text=f"S({start[0]:.1f},{start[1]:.1f},{start[2]:.1f})",
                            position=astar[0, :3])
vis.server.scene.add_label("end",   text=f"E({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})",
                            position=astar[-1, :3])
vis.show()
