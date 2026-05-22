"""Visualize flood PLY data overlaid on the terrain (flood zone only).

Strategy: load flood PLYs first (small), derive their XY bounding box in raw
coordinates, crop the terrain to that box + margin before filter_floaters,
then process only the small local subset.  Much faster than filtering 7M pts.

Saves flood_overview.png and launches viser.
Run: python demos/hillcounty_flood_vis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import plyfile as ply
from numpy.lib.recfunctions import structured_to_unstructured

from foci.utils.ply import extract_splat_data, filter_floaters, load_flood_data
from foci.visualisation.vis_utils import ViserVis

LOCAL = os.path.dirname(os.path.abspath(__file__))
TERRAIN_PLY = os.path.join(LOCAL, 'data/hillcounty_sm_30000.ply')
FLOOD_PLYS = {
    'low':    os.path.join(LOCAL, 'data/hillcountry_flood_low.ply'),
    'medium': os.path.join(LOCAL, 'data/hillcountry_flood_medium.ply'),
    'high':   os.path.join(LOCAL, 'data/hillcountry_flood_high.ply'),
}
MARGIN = 0.05   # extra margin around flood zone in raw (pre-scale) units

# ── 1. Load flood PLYs first — they are small (40k–200k pts) ─────────────────
print("Loading flood PLYs to get bounding box...")
raw_flood_pts = []
for level, path in FLOOD_PLYS.items():
    v = ply.PlyData.read(path)['vertex']
    pts = structured_to_unstructured(v[['x', 'y', 'z']])
    raw_flood_pts.append(pts)
    print(f"  {level}: {len(pts)} pts  "
          f"X={pts[:,0].min():.3f}..{pts[:,0].max():.3f}  "
          f"Y={pts[:,1].min():.3f}..{pts[:,1].max():.3f}")

all_flood = np.concatenate(raw_flood_pts)
x_lo = all_flood[:, 0].min() - MARGIN
x_hi = all_flood[:, 0].max() + MARGIN
y_lo = all_flood[:, 1].min() - MARGIN
y_hi = all_flood[:, 1].max() + MARGIN
print(f"Flood zone (raw, +margin): X={x_lo:.3f}..{x_hi:.3f}  Y={y_lo:.3f}..{y_hi:.3f}")

# ── 2. Load terrain — crop to flood zone BEFORE filter_floaters ───────────────
print("\nLoading terrain (raw)...")
t_raw = ply.PlyData.read(TERRAIN_PLY)['vertex']
t_all = structured_to_unstructured(t_raw[['x', 'y', 'z']])
print(f"  Total terrain pts: {len(t_all)}")

zone_mask = (
    (t_all[:, 0] >= x_lo) & (t_all[:, 0] <= x_hi) &
    (t_all[:, 1] >= y_lo) & (t_all[:, 1] <= y_hi)
)
print(f"  In flood zone: {zone_mask.sum()}")

# Re-read only the zone subset through extract_splat_data logic
# (we need means/covs/colors/opacities, so use the full reader on the full file
#  but index down immediately after)
print("Extracting splat data for flood zone only...")
means_all, covs_all, colors_all, opacities_all = extract_splat_data(TERRAIN_PLY)
means_z   = means_all[zone_mask]
covs_z    = covs_all[zone_mask]
colors_z  = colors_all[zone_mask]
opac_z    = opacities_all[zone_mask]
print(f"  Zone splats before filter: {len(means_z)}")

means_z, covs_z, colors_z, opac_z = filter_floaters(
    means_z, covs_z, colors_z, opac_z,
    opacity_threshold=0.3, density_radius=0.05, min_neighbors=5,
)
colors_z = np.clip(colors_z, 0.0, 1.0)

# ground_z from the zone subset (same 6th-percentile convention)
ground_z = np.percentile(means_z[:, 2], 6)
scale = 20
means_z[:, 2] -= ground_z
means_z = scale * means_z
covs_z  = scale**2 * covs_z

print(f"Terrain zone (scaled): X={means_z[:,0].min():.1f}..{means_z[:,0].max():.1f}  "
      f"Y={means_z[:,1].min():.1f}..{means_z[:,1].max():.1f}  "
      f"Z={means_z[:,2].min():.1f}..{means_z[:,2].max():.1f}")

# ── 3. Load flood data in scaled space ───────────────────────────────────────
FLOOD_COLORS_RGB = {
    'low':    np.array([0.0, 0.8, 1.0]),   # cyan
    'medium': np.array([0.0, 0.4, 1.0]),   # blue
    'high':   np.array([0.5, 0.0, 0.9]),   # purple
}

flood_data = {}
for level, path in FLOOD_PLYS.items():
    fm, fc, fo = load_flood_data(path, ground_z, scale)
    flood_data[level] = (fm, fc, fo)
    print(f"Flood {level} (scaled): X={fm[:,0].min():.1f}..{fm[:,0].max():.1f}  "
          f"Y={fm[:,1].min():.1f}..{fm[:,1].max():.1f}  "
          f"Z={fm[:,2].min():.1f}..{fm[:,2].max():.1f}  n={len(fm)}")

# ── 4. 2D overview PNG ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

step = max(1, len(means_z) // 5000)

ax = axes[0]
ax.scatter(means_z[::step, 0], means_z[::step, 1],
           c=colors_z[::step], s=4, alpha=0.4, label='terrain (zone)')
for level, (fm, _, _) in flood_data.items():
    s = max(1, len(fm) // 3000)
    ax.scatter(fm[::s, 0], fm[::s, 1],
               color=FLOOD_COLORS_RGB[level], s=6, alpha=0.8, label=f'flood {level}')
ax.set_xlabel('X (scaled)'); ax.set_ylabel('Y (scaled)')
ax.set_title('Overhead: flood zone terrain + flood levels')
ax.legend(markerscale=3); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(means_z[::step, 0], means_z[::step, 2],
            c=colors_z[::step], s=4, alpha=0.4, label='terrain (zone)')
for level, (fm, _, _) in flood_data.items():
    s = max(1, len(fm) // 3000)
    ax2.scatter(fm[::s, 0], fm[::s, 2],
                color=FLOOD_COLORS_RGB[level], s=6, alpha=0.8, label=f'flood {level}')
ax2.set_xlabel('X (scaled)'); ax2.set_ylabel('Z (scaled)')
ax2.set_title('Side view (X-Z): flood levels vs terrain')
ax2.legend(markerscale=3); ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_png = os.path.join(os.getcwd(), 'flood_overview.png')
plt.savefig(out_png, dpi=150)
print(f"\nSaved → {out_png}")
plt.close()

# ── 5. Viser 3D viewer ────────────────────────────────────────────────────────
print("Launching viser (http://localhost:8080)...")
vis = ViserVis()

step3d = max(1, len(means_z) // 8000)
vis.add_gaussians(means_z[::step3d], covs_z[::step3d],
                  color=colors_z[::step3d], opacity=opac_z[::step3d])

for level, (fm, _, _) in flood_data.items():
    s = max(1, len(fm) // 4000)
    rgb_arr = np.tile(FLOOD_COLORS_RGB[level], (len(fm[::s]), 1))
    vis.server.scene.add_point_cloud(f"flood_{level}", fm[::s],
                                     colors=rgb_arr, point_size=0.08)

vis.show()
