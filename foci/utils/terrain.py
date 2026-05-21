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
