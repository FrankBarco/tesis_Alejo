import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPolygon, box
from matplotlib.patches import Polygon
from noise import pnoise2  # pip install noise shapely

# ==========================
# Parámetros físicos
# ==========================
width_um = 5000
height_um = 5000
num_points = 400
margin_factor = 0.2  # 20% más grande para evitar huecos
ferrite_fraction = 0.75
martensite_fraction = 0.25
np.random.seed(33)

# ==========================
# Generación de puntos extendidos
# ==========================
x = np.random.rand(num_points) * (1 + margin_factor*2) - margin_factor
y = np.random.rand(num_points) * (1 + margin_factor*2) - margin_factor
points = np.column_stack([x, y])
vor = Voronoi(points)

# ==========================
# Asignación de fases
# ==========================
num_martensite = int(num_points * martensite_fraction)
martensite_idx = np.random.choice(range(num_points), num_martensite, replace=False)
ferrite_idx = [i for i in range(num_points) if i not in martensite_idx]

# ==========================
# Distorsión (bordes más realistas)
# ==========================
def distort_polygon(polygon, scale=0.002, intensity=20):
    new_poly = []
    for px, py in polygon:
        dx = pnoise2(px * scale, py * scale) * intensity
        dy = pnoise2(py * scale, px * scale) * intensity
        new_poly.append([px + dx, py + dy])
    return np.array(new_poly)

# ==========================
# Recorte con área física
# ==========================
domain = box(0, 0, width_um, height_um)

def get_clipped_regions(vor, domain):
    polygons = []
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[j] for j in region]
            polygon = np.array(polygon)
            polygon[:, 0] *= width_um
            polygon[:, 1] *= height_um
            poly_shape = ShapelyPolygon(polygon)
            if not poly_shape.is_valid:
                continue
            clipped = poly_shape.intersection(domain)
            if not clipped.is_empty:
                polygons.append((i, np.array(clipped.exterior.coords)))
    return polygons

regions = get_clipped_regions(vor, domain)

# ==========================
# Graficar
# ==========================
fig, ax = plt.subplots(figsize=(7, 7))
for i, polygon in regions:
    polygon = distort_polygon(polygon, scale=0.002, intensity=20)
    color = "#d7d7d7" if i in ferrite_idx else "#4d4d4d"
    ax.add_patch(Polygon(polygon, facecolor=color, edgecolor='magenta', linewidth=0.3))

ax.set_xlim(0, width_um)
ax.set_ylim(0, height_um)
ax.set_aspect('equal')
ax.set_title("Microestructura rellena - Acero 1020 (5×5 mm, 75% ferrita / 25% martensita)", fontsize=9)
ax.set_xlabel("µm")
ax.set_ylabel("µm")
plt.tight_layout()
plt.show()
