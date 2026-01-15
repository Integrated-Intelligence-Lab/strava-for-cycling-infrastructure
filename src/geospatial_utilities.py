from shapely.ops import nearest_points
from tqdm import tqdm
from shapely import LineString, Polygon
import numpy as np
import math

def find_closest_segment(point, line_segments):
    min_distance = float('inf')
    closest_segment = None
    for idx, line in tqdm(line_segments.iterrows()):
        nearest_point = nearest_points(point, line.geometry)[1]
        distance = point.distance(nearest_point)
        if distance < min_distance:
            min_distance = distance
            closest_segment = idx
    return closest_segment

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def bearing(line: LineString) -> float:
    """Return bearing in degrees, wrapped to 0-180 (parallel direction)."""
    x0, y0 = line.coords[0]
    x1, y1 = line.coords[-1]
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
    return ang                  # 0° = E/W, 90° = N/S

def angle_diff(a, b):
    """Smallest absolute difference of two bearings (0-90 °)."""
    d = abs(a - b) % 180
    return 180 - d if d > 90 else d


def create_polysplit_region(region,polygon_size = 1000):
    # Get the bounding box of the region
    xmin, ymin, xmax, ymax = region.total_bounds

    # Set the desired grid cell size (in the same units as the CRS, e.g., meters)
    grid_size = polygon_size

    # Build grid cells and clip them by the Brussels region
    polys = []
    region_union = region.unary_union
    x_coord = xmin
    while x_coord < xmax:
        y_coord = ymin
        while y_coord < ymax:
            cell = Polygon([
                (x_coord, y_coord),
                (x_coord + grid_size, y_coord),
                (x_coord + grid_size, y_coord + grid_size),
                (x_coord, y_coord + grid_size)
            ])
            cell_clipped = cell.intersection(region_union)
            if not cell_clipped.is_empty:
                polys.append(cell_clipped)
            y_coord += grid_size
        x_coord += grid_size
    return polys  