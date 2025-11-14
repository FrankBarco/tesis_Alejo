import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# ---------------------------------------------------------
# CONFIGURACIÓN DEL RVE
# ---------------------------------------------------------

RVE_SIZE = 500  # micras (500 µm = 0.5 mm)
N_GRAINS = 300  # total de granos del RVE

# Fracciones de fase del acero 1020 DP
fraction_ferrite = 0.75
fraction_martensite = 0.25

# Repartición de semillas (puntos Voronoi)
N_ferrite = int(N_GRAINS * fraction_ferrite)
N_martensite = int(N_GRAINS * fraction_martensite)

print("Ferrita:", N_ferrite, "granos")
print("Martensita:", N_martensite, "granos")

# ---------------------------------------------------------
# GENERACIÓN DE PUNTOS ALEATORIOS
# ---------------------------------------------------------

# Ferrita (75%) - granos más grandes → menos puntos por área
points_ferrite = np.random.rand(N_ferrite, 2) * RVE_SIZE

# Martensita (25%) - islas pequeñas → pueden estar más juntas
points_martensite = np.random.rand(N_martensite, 2) * RVE_SIZE

# Unimos todos los puntos
points = np.vstack([points_ferrite, points_martensite])

# Creamos el diagrama de Voronoi
vor = Voronoi(points)

# ---------------------------------------------------------
# PLOTEO DEL DIAGRAMA VORONOI
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=0.6)

# Pintamos las regiones según el tipo de fase
for i, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]

    # Regiones abiertas se descartan
    if -1 in region or len(region) == 0:
        continue

    polygon = [vor.vertices[j] for j in region]

    if i < N_ferrite:
        color = "#F2F2F2"  # ferrita (claro)
    else:
        color = "#404040"  # martensita (oscuro)

    ax.fill(*zip(*polygon), color=color, edgecolor='black', linewidth=0.2)

ax.set_xlim(0, RVE_SIZE)
ax.set_ylim(0, RVE_SIZE)
ax.set_aspect('equal')
ax.set_title("Microestructura Voronoi – Acero 1020 Dual-Phase (RVE: 500×500 µm)")
ax.set_xlabel("Micras (µm)")
ax.set_ylabel("Micras (µm)")

plt.tight_layout()
plt.show()
