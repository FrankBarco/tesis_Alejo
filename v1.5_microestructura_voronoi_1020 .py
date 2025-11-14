import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from shapely.ops import polygonize, unary_union
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURACIÓN DEL RVE Y TAMAÑOS DE GRANO
# ---------------------------------------------------------

RVE_SIZE = 500  # micras

grain_ferrite = 30   # micras (diámetro promedio)
grain_martensite = 8 # micras (diámetro promedio)

fraction_ferrite = 0.75
fraction_martensite = 0.25

# Distancia mínima entre puntos (radio)
min_dist_ferrite = grain_ferrite / 2
min_dist_martensite = grain_martensite / 2

# Número estimado de semillas (simple)
area_RVE = RVE_SIZE * RVE_SIZE
area_grain_ferrite = np.pi * (min_dist_ferrite**2)
area_grain_martensite = np.pi * (min_dist_martensite**2)

N_ferrite = int((area_RVE * fraction_ferrite) / area_grain_ferrite)
N_martensite = int((area_RVE * fraction_martensite) / area_grain_martensite)

print("FERRITA:", N_ferrite, "semillas")
print("MARTENSITA:", N_martensite, "semillas")

# ---------------------------------------------------------
# FUNCIÓN PARA GENERAR SEMILLAS CON DISTANCIA MÍNIMA
# ---------------------------------------------------------

def generate_points(num_points, min_dist, size):
    points = []
    attempts = 0
    max_attempts = num_points * 60

    while len(points) < num_points and attempts < max_attempts:
        p = np.random.rand(2) * size
        if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
            points.append(p)
        attempts += 1

    return np.array(points)

# ---------------------------------------------------------
# GENERACIÓN DE SEMILLAS
# ---------------------------------------------------------

points_ferrite = generate_points(N_ferrite, min_dist_ferrite, RVE_SIZE)
points_martensite = generate_points(N_martensite, min_dist_martensite, RVE_SIZE)

points = np.vstack([points_ferrite, points_martensite])

# ---------------------------------------------------------
# DIAGRAMA VORONOI Y RECORTE CON EL RVE
# ---------------------------------------------------------

vor = Voronoi(points)
RVE_box = box(0, 0, RVE_SIZE, RVE_SIZE)

regions = []
for i, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]
    if -1 in region or len(region) == 0:
        continue

    polygon = Polygon([vor.vertices[j] for j in region])

    # Recortar celda a la caja del RVE
    poly_clip = polygon.intersection(RVE_box)

    if not poly_clip.is_empty:
        regions.append((i, poly_clip))

# ---------------------------------------------------------
# PLOTEO FINAL
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

for i, poly in regions:
    phase = "Ferrite" if i < len(points_ferrite) else "Martensite"
    color = "#F2F2F2" if phase == "Ferrite" else "#2F2F2F"

    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, edgecolor='black', linewidth=0.25)

ax.set_xlim(0, RVE_SIZE)
ax.set_ylim(0, RVE_SIZE)
ax.set_aspect('equal')

ax.set_title("RVE - Microestructura Voronoi DP Steel\nFerrita 30 µm | Martensita 8 µm")
ax.set_xlabel("Micras (µm)")
ax.set_ylabel("Micras (µm)")

plt.tight_layout()
plt.show()
