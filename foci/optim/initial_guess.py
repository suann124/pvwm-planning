import numpy as np
import casadi as cas
from astar import AStar
import open3d as o3d
from foci.splines.bsplines import spline_eval

def linear_interpolation(start_point, end_point, n_control_points):
    """Return the control points of a linear interpolation between start and end point
    @args:
        start_point: np.array
        end_point: np.array
        n_control_points: int
    """
    control_points = np.zeros((n_control_points, start_point.shape[0]))
    for i in range(n_control_points):
        control_points[i] = start_point + i/(n_control_points - 1) * (end_point - start_point)
    return control_points

class BasicAStar(AStar):
    def __init__(self, bounds, occ_map, step_size=1.0,
                 height_map=None, hm_x_min=0.0, hm_y_min=0.0, hm_cell=1.0,
                 clearance=0.0, z_band=0.2, impassable_mask=None):
        self.bounds = bounds
        self.occ_map = occ_map
        self.step_size = step_size
        # Terrain-following: when height_map is supplied the valid Z at every
        # (x, y) is the local band [terrain_z + clearance, terrain_z + clearance + z_band]
        # instead of the global bounds[2] interval.
        self.height_map = height_map
        self.hm_x_min = hm_x_min
        self.hm_y_min = hm_y_min
        self.hm_cell = hm_cell
        self.clearance = clearance
        self.z_band = z_band
        # Optional (nx, ny) bool grid (same frame as height_map) marking cells the
        # robot cannot enter — e.g. flood water deeper than the wading depth. The
        # search refuses to expand into these cells, so it routes around them.
        self.impassable_mask = impassable_mask

    def _is_impassable(self, x, y):
        """True if world (x, y) lands in a cell flagged impassable."""
        if self.impassable_mask is None:
            return False
        nx, ny = self.impassable_mask.shape
        ix = int(np.clip((x - self.hm_x_min) / self.hm_cell, 0, nx - 1))
        iy = int(np.clip((y - self.hm_y_min) / self.hm_cell, 0, ny - 1))
        return bool(self.impassable_mask[ix, iy])

    def _terrain_z(self, x, y):
        """Return local terrain height at world position (x, y), or None if unavailable."""
        nx, ny = self.height_map.shape
        xi = int(np.clip((x - self.hm_x_min) / self.hm_cell, 0, nx - 1))
        yi = int(np.clip((y - self.hm_y_min) / self.hm_cell, 0, ny - 1))
        h = self.height_map[xi, yi]
        return float(h) if not np.isnan(h) else None

    def neighbors(self, n):
        neighbors = []

        if self.height_map is not None:
            # 2.5D terrain-following: XY search, Z constrained to a band above
            # the local terrain surface [terrain_z + clearance,
            #                            terrain_z + clearance + z_band].
            # The dz step is half the band so A* samples the bottom and top of
            # the band — enough vertical freedom to get past grid-alignment
            # mismatches without flying above flood obstacles.
            dz_step = max(self.z_band / 2.0, self.step_size / 4.0)

            for dx in [-self.step_size, 0, self.step_size]:
                for dy in [-self.step_size, 0, self.step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    x2 = n[0] + dx
                    y2 = n[1] + dy

                    if x2 < self.bounds[0][0] or x2 >= self.bounds[0][1]:
                        continue
                    if y2 < self.bounds[1][0] or y2 >= self.bounds[1][1]:
                        continue

                    # Deep water (or any masked cell) is a no-go: route around it.
                    if self._is_impassable(x2, y2):
                        continue

                    tz = self._terrain_z(x2, y2)
                    if tz is None:
                        continue
                    z_lo = tz + self.clearance
                    z_hi = tz + self.clearance + self.z_band

                    z2 = z_lo
                    while z2 <= z_hi + 1e-9:
                        if not self.occ_map.check_if_included(
                                o3d.utility.Vector3dVector(np.array([[x2, y2, z2]])))[0]:
                            neighbors.append((x2, y2, z2))
                        z2 += dz_step

        else:
            # 3D search with static global Z bounds (backward-compatible fallback
            # for callers that do not supply a height map).
            for dx in [-self.step_size, 0, self.step_size]:
                for dy in [-self.step_size, 0, self.step_size]:
                    for dz in [-self.step_size, 0, self.step_size]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        x2 = n[0] + dx
                        y2 = n[1] + dy
                        z2 = n[2] + dz

                        if x2 < self.bounds[0][0] or x2 >= self.bounds[0][1]:
                            continue
                        if y2 < self.bounds[1][0] or y2 >= self.bounds[1][1]:
                            continue
                        if z2 < self.bounds[2][0] or z2 >= self.bounds[2][1]:
                            continue
                        if self._is_impassable(x2, y2):
                            continue

                        if self.occ_map.check_if_included(
                                o3d.utility.Vector3dVector(np.array([[x2, y2, z2]])))[0]:
                            continue

                        neighbors.append((x2, y2, z2))

        return neighbors
                    


    def distance_between(self, n1, n2):
        return np.linalg.norm(np.array(n1) - np.array(n2))
            
    def heuristic_cost_estimate(self, current, goal):
        return np.linalg.norm(np.array(current) - np.array(goal))
    
    def is_goal_reached(self, current, goal):
        if self.height_map is not None:
            # In 2.5D mode Z is terrain-driven so only XY proximity matters.
            return np.linalg.norm(np.array(current[:2]) - np.array(goal[:2])) < self.step_size
        return np.linalg.norm(np.array(current) - np.array(goal)) < 0.5
    


def astar_path_spline_fit(start_point, end_point, means, voxel_size=0.25, num_control_points=20, z_range=None,
                           height_map=None, hm_x_min=0.0, hm_y_min=0.0, hm_cell=1.0,
                           clearance=0.0, z_band=0.2, impassable_mask=None):
    min_x = min(means[:,0].min(), start_point[0], end_point[0])
    max_x = max(means[:,0].max(), start_point[0], end_point[0])
    min_y = min(means[:,1].min(), start_point[1], end_point[1])
    max_y = max(means[:,1].max(), start_point[1], end_point[1]) 
    min_z = min(means[:,2].min(), start_point[2], end_point[2])
    max_z = max(means[:,2].max(), start_point[2], end_point[2])

    start_point = start_point[:3]
    end_point = end_point[:3]

    spread_x = max_x - min_x
    spread_y = max_y - min_y
    spread_z = max_z - min_z

    bounds = [[min_x - 0.2 * spread_x, max_x + 0.2 * spread_x],
              [min_y - 0.2 * spread_y, max_y + 0.2 * spread_y],
              [min_z - 0.2 * spread_z, max_z + 0.01 * spread_z]]
    # bounds[2] (Z) is only used by the static fallback path (height_map=None).
    # When height_map is provided A* is 2.5D and Z is terrain-driven; z_range is ignored.
    if height_map is None and z_range is not None:
        bounds[2] = [z_range[0], z_range[1]]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=voxel_size)

    astar_path_finder = BasicAStar(
        bounds, voxel_grid, step_size=voxel_size,
        height_map=height_map, hm_x_min=hm_x_min,
        hm_y_min=hm_y_min, hm_cell=hm_cell,
        clearance=clearance, z_band=z_band,
        impassable_mask=impassable_mask,
    )

    start = (start_point[0], start_point[1], start_point[2])
    end = (end_point[0], end_point[1], end_point[2])

    start_xyz = np.array([[start_point[0], start_point[1], start_point[2]]], dtype=float)
    end_xyz   = np.array([[end_point[0], end_point[1], end_point[2]]], dtype=float)

    start_occ = voxel_grid.check_if_included(o3d.utility.Vector3dVector(start_xyz))[0]
    end_occ   = voxel_grid.check_if_included(o3d.utility.Vector3dVector(end_xyz))[0]

    print("Start occupied?", start_occ, "start =", start_xyz[0])
    print("End occupied?  ", end_occ,   "end   =", end_xyz[0])
    path = astar_path_finder.astar(start, end)

    if path is None:
        raise ValueError("No path found")

    path = [p for p in path]
    path_arr = np.array(path, dtype=float)
    # path_arr[0] = start_point
    # path_arr[-1] = end_point
    print("Astar path:", path_arr)
    
    num_samples = len(path_arr)
    control_points = cas.SX.sym('control_points', num_control_points, 3)
    dec_vars = cas.vertcat(cas.vec(control_points))

    curve = spline_eval(control_points, num_samples)

    fitting_cost = 0
    #define the cost function
    for i in range(num_samples):
        fitting_cost += (curve[i, 0].T - path_arr[i,0])**2 + (curve[i, 1].T - path_arr[i,1])**2 + (curve[i, 2].T - path_arr[i,2])**2

    cost = fitting_cost 
    # define constraints same as cost
    cons = cas.SX([])
    lbg = []
    ubg = []
    # constrain start and end points
    for i in range(3):
        cons = cas.vertcat(cons, curve[0,i])
        lbg = np.concatenate((lbg, [path_arr[0,i] -0.005]))
        ubg = np.concatenate((ubg, [path_arr[0,i] + 0.005]))

    for i in range(3):
        cons = cas.vertcat(cons, curve[-1,i])
        lbg = np.concatenate((lbg, [path_arr[-1,i] -0.005]))
        ubg = np.concatenate((ubg, [path_arr[-1,i] +0.005]))


    ipop_options = {"ipopt.print_level": 0,
                     "ipopt.max_iter": 200, 
                     "ipopt.tol": 1e-3, 
                     "print_time": 0, 
                     "ipopt.acceptable_tol": 1e-3, 
                     "ipopt.acceptable_obj_change_tol": 1e-3, 
                     "ipopt.hessian_approximation": "limited-memory", 
                     "ipopt.mu_strategy": "adaptive", 
                     "ipopt.linear_solver": "ma27"}

    # define solver
    nlp = {'x': dec_vars, 'f': cost, 'g': cons}
    solver = cas.nlpsol('solver', 'ipopt', nlp, ipop_options)

    control_points_init = np.zeros((num_control_points, 3,1))
    sol = solver(lbg=lbg, ubg=ubg, x0=control_points_init.flatten())

    control_points_opt = np.array(sol['x']).reshape(3,num_control_points).T
    
    # a zero row to make it 4D
    control_points_opt = np.concatenate((control_points_opt, np.zeros((num_control_points,1))), axis = 1).T
    

    return control_points_opt.flatten()

