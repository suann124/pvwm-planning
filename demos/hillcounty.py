import os
import numpy as np

from foci.visualisation.vis_utils import ViserVis
from foci.utils.ply import extract_splat_data
from foci.utils.terrain import build_height_map, query_height
from foci.planners.planner_ground import GroundPlanner

LOCAL = os.path.dirname(os.path.abspath(__file__))
ply_file = os.path.join(LOCAL, 'data/hillcounty_sm_30000.ply')

means, covs, colors, opacities = extract_splat_data(ply_file)

# Normalize Z so ground = 0
ground_z = np.percentile(means[:, 2], 6)
print(f'Ground z cut off before normalization: {ground_z}')
means[:, 2] -= ground_z

# Scale up so robot covariance appears small relative to scene
scale = 20
means = scale * means
covs  = scale**2 * covs

print("Scene bounds (ground = 0, scaled):")
print(f"  X: {means[:,0].min():.3f} to {means[:,0].max():.3f}")
print(f"  Y: {means[:,1].min():.3f} to {means[:,1].max():.3f}")
print(f"  Z: {means[:,2].min():.3f} to {means[:,2].max():.3f}")

robot_cov = np.eye(3) * 0.001
z_range   = (1, 1.2)  # tight band — keeps A* 2D (< step_size 0.25)
nav_z = np.average(z_range)

# Terrain height lookup from ground-level Gaussians
ground_gaussians = means[(means[:, 2] > z_range[0] - 2) & (means[:, 2] < z_range[1])]

def terrain_height(x, y, radius=0.3 * scale):
    dists = np.sqrt((ground_gaussians[:, 0] - x)**2 + (ground_gaussians[:, 1] - y)**2)
    nearby = ground_gaussians[dists < radius]
    if len(nearby) == 0:
        return 0.0
    return nearby[:, 2].max()

# Only above-ground Gaussians are obstacles
obstacle_mask  = means[:, 2] > z_range[1]
obstacle_means = means[obstacle_mask]
obstacle_covs  = covs[obstacle_mask]
print(f"Obstacle Gaussians: {obstacle_mask.sum()} / {len(means)}")

planner = GroundPlanner(obstacle_means, obstacle_covs, robot_cov,
                        num_control_points=10, num_samples=40, z_range=z_range)

# Start/end in scaled coordinates (original * scale)
start = [11, -20, nav_z, 0.0]
end   = [11,  20, nav_z, 0.0]
opt_curve, astar = planner.plan(start, end)
print(f"A* path points: {len(astar)}")
print(f"  X: {astar[:,0].min():.2f} to {astar[:,0].max():.2f}")
print(f"  Y: {astar[:,1].min():.2f} to {astar[:,1].max():.2f}")
print(f"  Z: {astar[:,2].min():.2f} to {astar[:,2].max():.2f}")
print(astar[:, :3])

# Pin path to terrain surface
for i in range(len(opt_curve)):
    x, y = opt_curve[i, 0], opt_curve[i, 1]
    opt_curve[i, 2] = terrain_height(x, y) + 0.05

# Crop visualization to path region
margin = 100
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

vis.server.scene.add_label("start", text=f"S({astar[0,0]:.1f},{astar[0,1]:.1f},{astar[0,2]:.1f})", position=astar[0,:3])
vis.server.scene.add_label("end",   text=f"E({astar[-1,0]:.1f},{astar[-1,1]:.1f},{astar[-1,2]:.1f})", position=astar[-1,:3])
vis.show()