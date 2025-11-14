import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# ---------------------------------------------------------
# CONFIGURACIÓN DEL RVE
# ---------------------------------------------------------

RVE_SIZE = 500  # micras (500 µm)
N_GRAINS = 300  # total de granos del RVE

# Fracciones de fase
fraction_ferrite = 0.75
fraction_martensite = 0.25

N_ferrite = int(N_GRAINS * fraction_ferrite)
N_martensite = int(N_GRAINS * fraction_martensite)

# ---------------------------------------------------------
# GENERACIÓN DE PUNTOS
# ---------------------------------------------------------

points_ferrite = np.random.rand(N_ferrite, 2) * RVE_SIZE
points_martensite = np.random.rand(N_martensite, 2) * RVE_SIZE

points = np.vstack([points_ferrite, points_martensite])

# Diagrama Voronoi
vor = Voronoi(points)

# ---------------------------------------------------------
# PLOTEO SIN PUNTOS
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=0.6, show_points=False)

# Pintar regiones según fase
for i, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]

    if -1 in region or len(region) == 0:
        continue

    polygon = [vor.vertices[j] for j in region]

    if i < N_ferrite:
        color = "#F2F2F2"  # Ferrita (clara)
    else:
        color = "#404040"  # Martensita (oscura)

    ax.fill(*zip(*polygon), color=color, edgecolor='black', linewidth=0.2)

ax.set_xlim(0, RVE_SIZE)
ax.set_ylim(0, RVE_SIZE)
ax.set_aspect('equal')
ax.set_title("Microestructura Voronoi – Acero 1020 Dual-Phase (RVE: 500×500 µm)")
ax.set_xlabel("Micras (µm)")
ax.set_ylabel("Micras (µm)")

plt.tight_layout()
plt.show()
