import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

# ============================================================
# CONFIGURACIÓN DEL DOMINIO
# ============================================================

AREA = 28.27           # mm^2
L = np.sqrt(AREA)      # mm → lado del dominio
print(f"Dominio: {L:.3f} mm × {L:.3f} mm")

# Número razonable para simulación FEA
N_ferrite = 600
N_martensite = 200
N_total = N_ferrite + N_martensite

# ============================================================
# GENERACIÓN DE PUNTOS CON PESOS (tamaño de grano)
# ============================================================

# Ferrita: granos grandes → puntos más separados → peso mayor
weights_ferrite = np.random.normal(loc=1.0, scale=0.15, size=N_ferrite)

# Martensita: islas pequeñas → peso menor
weights_martensite = np.random.normal(loc=0.35, scale=0.10, size=N_martensite)

weights = np.hstack([weights_ferrite, weights_martensite])

# Puntos uniformes en el dominio
points = np.random.rand(N_total, 2) * L  

# ============================================================
# ESCALADO TIPO "POWER DIAGRAM" (Weighted Voronoi)
# ============================================================

# Método simple: desplazamiento radial por peso
# (Aproxima un Voronoi Potencial o Laguerre Diagram)

points_scaled = points * weights[:, None]

# Voronoi final
vor = Voronoi(points_scaled)

# ============================================================
# GRÁFICA FINAL
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

for i, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]
    if -1 in region or len(region) == 0:
        continue

    polygon = [vor.vertices[j] for j in region]

    # Fase según índice
    if i < N_ferrite:
        color = "#F0F0F0"       # ferrita clara
    else:
        color = "#505050"       # martensita oscura

    ax.fill(*zip(*polygon), color=color, edgecolor='black', linewidth=0.15)

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_title(f"Microestructura Voronoi – Acero 1020 DP (Área: {AREA} mm²)")
ax.set_xlabel("mm")
ax.set_ylabel("mm")

plt.tight_layout()
plt.show()

