
import plyfile as ply
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial.transform import Rotation as R


def extract_splat_data(ply_file):
    """Extracts the splat data from a ply file.
    Args:
        ply_file (str): Path to the ply file.
    Returns:
        means (np.ndarray): Means of the splats.
        covs (np.ndarray): Covariances of the splats.
        opacities (np.ndarray): Opacities of the splats.
    """
    # read in ply file
    plydata = ply.PlyData.read(ply_file)
    print(len(plydata['vertex']))
    splat = plydata['vertex']   

    SH_C0 = 0.28209479177387814

    means = structured_to_unstructured(splat[["x", "y", "z"]])
    scales = np.exp(structured_to_unstructured(splat[['scale_0', 'scale_1', 'scale_2']]))
    wxyzs = structured_to_unstructured(splat[['rot_0', 'rot_1', 'rot_2', 'rot_3']])
    colors = 0.5 + SH_C0 * structured_to_unstructured(splat[['f_dc_0', 'f_dc_1', 'f_dc_2']])
    opacities = 1.0 / (1.0 + np.exp(-splat["opacity"][:, None]))

    Rs = np.zeros((len(splat), 3, 3))
    for i in range(len(splat)):
        Rs[i] = R.from_quat([wxyzs[i][1],
                             wxyzs[i][2],
                             wxyzs[i][3],
                             wxyzs[i][0]]).as_matrix()

    # recontruct covariances from scales and quat
    covs = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )

    return means, covs, colors, opacities


def load_flood_data(ply_file, ground_z, scale):
    """Load a flood PLY and align it to the terrain coordinate space.

    Applies the same ground-z shift and scale that was used on the terrain so
    that flood means are directly comparable to terrain means.

    Args:
        ply_file:  path to flood PLY (same 3DGS format as the terrain PLY)
        ground_z:  terrain ground-Z offset — np.percentile(terrain_means[:,2], 6)
                   computed *before* any scaling
        scale:     terrain scale factor (e.g. 20)

    Returns:
        means     (N, 3) flood Gaussian centres in scaled terrain coordinates
        colors    (N, 3) RGB [0..1]
        opacities (N, 1) sigmoid opacities
    """
    means, covs, colors, opacities = extract_splat_data(ply_file)
    means[:, 2] -= ground_z
    means = scale * means
    covs  = scale**2 * covs
    return means, covs, colors, opacities


def filter_floaters(means, covs, colors, opacities,
                    opacity_threshold=0.3,
                    density_radius=0.5,
                    min_neighbors=5):
    """Remove floater Gaussians using a combined opacity + local density filter.

    Args:
        opacity_threshold: minimum sigmoid opacity (0-1); lower than a pure
                           opacity filter since density handles isolated splats
        density_radius:    neighbourhood radius in scene units (after any scaling
                           you apply to means)
        min_neighbors:     minimum number of other Gaussians within density_radius
                           for a point to be kept
    Returns:
        filtered means, covs, colors, opacities
    """
    from scipy.spatial import KDTree

    opacity_mask = opacities[:, 0] > opacity_threshold

    tree = KDTree(means)
    # subtract 1 because each point counts itself
    neighbor_counts = np.array(tree.query_ball_point(means, r=density_radius, return_length=True)) - 1
    density_mask = neighbor_counts >= min_neighbors

    mask = opacity_mask & density_mask
    print(f"filter_floaters: kept {mask.sum()} / {len(means)} "
          f"(opacity>{opacity_threshold}, neighbors>={min_neighbors} within r={density_radius})")
    return means[mask], covs[mask], colors[mask], opacities[mask]