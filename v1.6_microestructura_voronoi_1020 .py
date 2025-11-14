import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import ezdxf
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

min_dist_ferrite = grain_ferrite / 2
min_dist_martensite = grain_martensite / 2

area_RVE = RVE_SIZE * RVE_SIZE
area_grain_ferrite = np.pi * (min_dist_ferrite**2)
area_grain_martensite = np.pi * (min_dist_martensite**2)

N_ferrite = int((area_RVE * fraction_ferrite) / area_grain_ferrite)
N_martensite = int((area_RVE * fraction_martensite) / area_grain_martensite)

print("FERRITA:", N_ferrite, "semillas")
print("MARTENSITA:", N_martensite, "semillas")

# ---------------------------------------------------------
# FUNCIÓN PARA GENERAR SEMILLAS
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
# GENERACIÓN DE PUNTOS
# ---------------------------------------------------------

points_ferrite = generate_points(N_ferrite, min_dist_ferrite, RVE_SIZE)
points_martensite = generate_points(N_martensite, min_dist_martensite, RVE_SIZE)

points = np.vstack([points_ferrite, points_martensite])

# ---------------------------------------------------------
# VORONOI + RECORTE A LA CAJA
# ---------------------------------------------------------

vor = Voronoi(points)
RVE_box = box(0, 0, RVE_SIZE, RVE_SIZE)

regions = []
for i, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]
    if -1 in region or len(region) == 0:
        continue

    polygon = Polygon([vor.vertices[j] for j in region])
    poly_clip = polygon.intersection(RVE_box)

    if not poly_clip.is_empty:
        regions.append((i, poly_clip))

# ---------------------------------------------------------
# EXPORTAR A DXF
# ---------------------------------------------------------

doc = ezdxf.new(dxfversion="R2010")
msp = doc.modelspace()

for i, poly in regions:
    coords = list(poly.exterior.coords)

    # Elegir capa según fase
    if i < len(points_ferrite):
        layer = "Ferrite"
        color = 7  # blanco
    else:
        layer = "Martensite"
        color = 8  # gris

    if layer not in doc.layers:
        doc.layers.add(name=layer, color=color)

    msp.add_lwpolyline(
        coords,
        format="xy",
        close=True,
        dxfattribs={"layer": layer}
    )

doc.saveas("RVE_DP.dxf")
print("DXF generado: RVE_DP.dxf")

# ---------------------------------------------------------
# PLOTEO PREVIEW
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

for i, poly in regions:
    phase = "Ferrita" if i < len(points_ferrite) else "Martensita"
    color = "#F2F2F2" if phase == "Ferrita" else "#2F2F2F"

    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, edgecolor='black', linewidth=0.25)

ax.set_xlim(0, RVE_SIZE)
ax.set_ylim(0, RVE_SIZE)
ax.set_aspect('equal')
ax.set_title("RVE Microestructura DP Steel – Exportado a DXF")
plt.tight_layout()
plt.savefig("RVE_DP.png", dpi=300)
plt.show()

print("Preview generado: RVE_DP.png")
