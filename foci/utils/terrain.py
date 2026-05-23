import numpy as np
from scipy.ndimage import generic_filter, minimum_filter


def build_height_map(ground_pts, cell_size=0.5):
    """Build a 2D height map from ground Gaussian means.

    Args:
        ground_pts: (N, 3) array of XYZ positions of ground-level Gaussians
        cell_size:  grid resolution in scene units (smaller = finer, better slope tracking)

    Returns:
        height_map: (nx, ny) array of surface z values (NaN where no data)
        x_min, y_min: origin of the grid
        cell_size: grid resolution (passed through for use in query)
    """
    x_min, x_max = ground_pts[:, 0].min(), ground_pts[:, 0].max()
    y_min, y_max = ground_pts[:, 1].min(), ground_pts[:, 1].max()

    nx = int((x_max - x_min) / cell_size) + 1
    ny = int((y_max - y_min) / cell_size) + 1

    # Initialize with -inf so np.maximum.at works correctly (max(nan, x) = nan in numpy)
    height_map = np.full((nx, ny), -np.inf)

    xi = np.clip(((ground_pts[:, 0] - x_min) / cell_size).astype(int), 0, nx - 1)
    yi = np.clip(((ground_pts[:, 1] - y_min) / cell_size).astype(int), 0, ny - 1)

    # Max z per cell = top of ground Gaussians = navigable surface
    np.maximum.at(height_map, (xi, yi), ground_pts[:, 2])

    # Convert unvisited cells (-inf) back to NaN for the fill step
    height_map[np.isinf(height_map)] = np.nan

    # Fill NaN cells with nearest neighbour so every cell has a value
    def _fill(values):
        centre = values[len(values) // 2]
        if not np.isnan(centre):
            return centre
        valid = values[~np.isnan(values)]
        return valid.mean() if len(valid) > 0 else np.nan

    for _ in range(10):
        if not np.any(np.isnan(height_map)):
            break
        height_map = generic_filter(height_map, _fill, size=3, mode='nearest')

    # Erode: each cell takes the minimum of its 3x3 neighbourhood.
    # Lets trench/slope-floor values bleed into adjacent rim cells so the
    # post-processed path dips down instead of bridging over depressions.
    height_map = minimum_filter(height_map, size=3, mode='nearest')

    return height_map, x_min, y_min, cell_size


def build_flood_obstacle_mask(flood_means, height_map, x_min, y_min, cell_size,
                               wading_depth=0.0):
    """Mark occupancy-grid cells as impassable where flood water exceeds wading depth.

    Water depth at each flood Gaussian is computed as its Z minus the local
    terrain height from the height map.  A cell is marked flooded when at least
    one flood Gaussian has depth > wading_depth.  Combine the returned mask with
    the terrain obstacle mask via logical OR before passing to the planner.

    Args:
        flood_means:  (N, 3) flood Gaussian centres in scaled terrain coords
        height_map:   (nx, ny) terrain height map from build_height_map
        x_min, y_min: grid origin from build_height_map
        cell_size:    grid resolution from build_height_map
        wading_depth: water depth (scaled units) the robot can wade through.
                      0.0 = any standing water is impassable; increase to allow
                      shallow crossings.

    Returns:
        flood_mask: (nx, ny) bool array — True where the cell is flooded
    """
    nx, ny = height_map.shape
    flood_mask = np.zeros((nx, ny), dtype=bool)

    xi = np.clip(((flood_means[:, 0] - x_min) / cell_size).astype(int), 0, nx - 1)
    yi = np.clip(((flood_means[:, 1] - y_min) / cell_size).astype(int), 0, ny - 1)

    local_terrain_z = height_map[xi, yi]
    depth = flood_means[:, 2] - local_terrain_z
    deep = depth > wading_depth

    flood_mask[xi[deep], yi[deep]] = True
    return flood_mask


def build_flood_navigation(height_map, flood_means, x_min, y_min, cell_size,
                           wading_depth=0.0, depth_eps=0.02):
    """Derive the navigable surface and impassable mask for a flooded scene.

    Combines the dry-terrain height map with flood Gaussians so the robot rides
    *on top of* shallow (wadeable) water and routes *around* deep water, instead
    of navigating beneath the water surface.  Feed ``nav_surface`` to the planner
    in place of the dry height map, and ``impassable`` to the A* search.

    Args:
        height_map:   (nx, ny) dry-terrain surface Z from build_height_map
        flood_means:  (N, 3) flood Gaussian centres in the same scaled coords
        x_min, y_min, cell_size: grid metadata from build_height_map
        wading_depth: water depth (scaled units) the robot can wade through; a
                      cell deeper than this is impassable. 0.0 => any standing
                      water blocks.
        depth_eps:    water shallower than this above terrain is treated as
                      noise and ignored (cell stays dry).

    Returns:
        nav_surface:   (nx, ny) surface the robot navigates above — the water
                       height where wadeable, terrain elsewhere. Always
                       >= height_map, so a path at nav_surface + clearance can
                       never sit under the water.
        impassable:    (nx, ny) bool — True where water depth > wading_depth.
        water_surface: (nx, ny) max flood Z per cell, NaN where no flood.
    """
    nx, ny = height_map.shape

    # Max flood Z per cell = the water surface (mirrors build_height_map's
    # per-cell max idiom). -inf sentinel so np.maximum.at works, then -> NaN.
    water_surface = np.full((nx, ny), -np.inf)
    xi = np.clip(((flood_means[:, 0] - x_min) / cell_size).astype(int), 0, nx - 1)
    yi = np.clip(((flood_means[:, 1] - y_min) / cell_size).astype(int), 0, ny - 1)
    np.maximum.at(water_surface, (xi, yi), flood_means[:, 2])
    water_surface[np.isinf(water_surface)] = np.nan

    depth = water_surface - height_map                 # NaN where dry
    wet = ~np.isnan(water_surface) & (depth > depth_eps)
    impassable = wet & (depth > wading_depth)
    wadeable = wet & (depth <= wading_depth)

    # Ride on the water where it is wadeable; stay on terrain otherwise.
    nav_surface = height_map.astype(float).copy()
    nav_surface[wadeable] = water_surface[wadeable]
    return nav_surface, impassable, water_surface


def build_flood_obstacles(impassable, water_surface, height_map,
                          x_min, y_min, cell_size,
                          layer_dz=None, sigma_xy_scale=0.75):
    """Obstacle Gaussians that fill the water *column* over each impassable cell.

    The planner's obstacle cost is a Gaussian-Gaussian convolution, which decays
    sharply away from each obstacle centre. A single thin disk at the water
    surface is therefore invisible to a trajectory passing lower in the column —
    the optimiser slips underneath it. Stacking Gaussians from the terrain floor
    up to the water surface gives the cost vertical extent, so a curve at *any*
    height inside the water feels it and is pushed out laterally (in XY).

    Args:
        impassable:    (nx, ny) bool — deep-water cells (from build_flood_navigation)
        water_surface: (nx, ny) water height per cell, NaN where dry
        height_map:    (nx, ny) terrain floor
        x_min, y_min, cell_size: grid metadata
        layer_dz:      vertical spacing between stacked Gaussians (scaled units).
                       Defaults to cell_size. Smaller => denser column.
        sigma_xy_scale: lateral std-dev as a fraction of cell_size.

    Returns:
        means (M, 3), covs (M, 3, 3)  — diagonal covariances. Empty (0, 3) /
        (0, 3, 3) when there are no impassable cells.
    """
    if layer_dz is None:
        layer_dz = cell_size

    sigma_xy = sigma_xy_scale * cell_size
    means, covs = [], []

    for ix, iy in np.argwhere(impassable):
        floor = float(height_map[ix, iy])
        top = float(water_surface[ix, iy])
        depth = top - floor
        if not np.isfinite(depth) or depth <= 0:
            continue

        # Number of stacked layers and their (slightly overlapping) Z extent.
        n = max(1, int(round(depth / layer_dz)))
        seg = depth / n
        sigma_z = 0.75 * seg

        x = x_min + (ix + 0.5) * cell_size
        y = y_min + (iy + 0.5) * cell_size
        for i in range(n):
            z = floor + (i + 0.5) * seg
            means.append([x, y, z])
            covs.append(np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2]))

    if means:
        return np.asarray(means, dtype=float), np.asarray(covs, dtype=float)
    return np.zeros((0, 3), dtype=float), np.zeros((0, 3, 3), dtype=float)


def build_pin_surface(height_map, water_surface):
    """Surface to snap the final trajectory Z onto: the higher of terrain and
    water wherever there is water, terrain elsewhere.

    The NLP leaves Z loose and the pipeline snaps it to a surface afterwards.
    Pinning to bare terrain puts the robot on the dry floor *beneath* standing
    water; pinning to this ceiling guarantees the path rides on top of any water
    it strays over and never sits below the surface.
    """
    pin = height_map.astype(float).copy()
    wet = ~np.isnan(water_surface)
    pin[wet] = np.maximum(height_map[wet], water_surface[wet])
    return pin


def query_height(height_map, x_min, y_min, cell_size, x, y, default=0.0):
    """Look up terrain height at position (x, y) using bilinear interpolation.

    Args:
        height_map: output of build_height_map
        x_min, y_min, cell_size: grid metadata from build_height_map
        x, y: query position in scene units
        default: fallback if position is outside the map

    Returns:
        float: terrain surface z at (x, y)
    """
    fx = (x - x_min) / cell_size
    fy = (y - y_min) / cell_size

    x0, y0 = int(fx), int(fy)
    x1, y1 = x0 + 1, y0 + 1

    nx, ny = height_map.shape
    if x0 < 0 or x1 >= nx or y0 < 0 or y1 >= ny:
        xi = np.clip(x0, 0, nx - 1)
        yi = np.clip(y0, 0, ny - 1)
        h = height_map[xi, yi]
        return default if np.isnan(h) else float(h)

    tx = fx - x0
    ty = fy - y0

    h00 = height_map[x0, y0]
    h10 = height_map[x1, y0]
    h01 = height_map[x0, y1]
    h11 = height_map[x1, y1]

    vals = np.array([h00, h10, h01, h11])
    if np.all(np.isnan(vals)):
        return default
    fill = float(np.nanmean(vals))
    h00 = fill if np.isnan(h00) else h00
    h10 = fill if np.isnan(h10) else h10
    h01 = fill if np.isnan(h01) else h01
    h11 = fill if np.isnan(h11) else h11

    h = (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + \
        (1 - tx) * ty * h01 + tx * ty * h11
    return float(h)
