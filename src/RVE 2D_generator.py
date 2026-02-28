import numpy as np
import random
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
import os

# =====================================================
# CONFIGURACION DE RUTA DE SALIDA
# =====================================================

OUTPUT_DIR = r"D:\37-Alejo U\Generar_Voronio_2D\outputs\APDL_inp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# VERSIONAMIENTO AUTOMATICO
# =====================================================

def get_next_version_filename(base_name="RVE_APDL_v", extension=".inp"):
    version = 1
    while os.path.exists(os.path.join(OUTPUT_DIR, f"{base_name}{version}{extension}")):
        version += 1
    return os.path.join(OUTPUT_DIR, f"{base_name}{version}{extension}")

# =====================================================
# PARAMETROS RVE
# =====================================================

RVE_SIZE = 0.5  # mm

grain_ferrite = 0.03
grain_martensite = 0.008

fraction_ferrite = 0.75
fraction_martensite = 0.25

min_dist_f = grain_ferrite / 2.0
min_dist_m = grain_martensite / 2.0

area_RVE = RVE_SIZE ** 2
area_f_seed = np.pi * min_dist_f**2
area_m_seed = np.pi * min_dist_m**2

N_ferrite = int((area_RVE * fraction_ferrite) / area_f_seed)
N_martensite = int((area_RVE * fraction_martensite) / area_m_seed)

print("=================================")
print("SEMILLAS GENERADAS")
print("Ferrita:", N_ferrite)
print("Martensita:", N_martensite)
print("=================================")

# =====================================================
# GENERACION DE PUNTOS CON DISTANCIA MINIMA
# =====================================================

def generate_points(n, min_dist):
    points = []
    attempts = 0
    max_attempts = n * 800
    
    while len(points) < n and attempts < max_attempts:
        x = random.uniform(0, RVE_SIZE)
        y = random.uniform(0, RVE_SIZE)
        p = np.array([x, y])
        
        if all(np.linalg.norm(p - np.array(q)) > min_dist for q in points):
            points.append([x, y])
        
        attempts += 1
    
    return np.array(points)

points_f = generate_points(N_ferrite, min_dist_f)
points_m = generate_points(N_martensite, min_dist_m)

points = np.vstack((points_f, points_m))

# =====================================================
# VORONOI
# =====================================================

vor = Voronoi(points)
bbox = box(0, 0, RVE_SIZE, RVE_SIZE)

regions = []
region_phase = []

for i, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]
    if -1 not in region and len(region) > 0:
        polygon = Polygon([vor.vertices[v] for v in region])
        clipped = polygon.intersection(bbox)
        
        if clipped.is_valid and clipped.area > 1e-12:
            regions.append(clipped)
            if i < len(points_f):
                region_phase.append("ferrite")
            else:
                region_phase.append("martensite")

print("Regiones finales creadas:", len(regions))

# =====================================================
# CALCULO DE FRACCION REAL POR AREA
# =====================================================

area_f_real = 0.0
area_m_real = 0.0

for poly, phase in zip(regions, region_phase):
    if phase == "ferrite":
        area_f_real += poly.area
    else:
        area_m_real += poly.area

fraction_f_real = area_f_real / area_RVE * 100
fraction_m_real = area_m_real / area_RVE * 100

print("=================================")
print("FRACCIONES REALES")
print(f"Ferrita real: {fraction_f_real:.2f}%")
print(f"Martensita real: {fraction_m_real:.2f}%")
print("=================================")

# =====================================================
# PREVISUALIZACION PNG
# =====================================================

fig, ax = plt.subplots(figsize=(7,7))

for poly, phase in zip(regions, region_phase):
    x, y = poly.exterior.xy
    if phase == "ferrite":
        ax.fill(x, y, color="#4CAF50")
    else:
        ax.fill(x, y, color="#F44336")

ax.set_xlim(0, RVE_SIZE)
ax.set_ylim(0, RVE_SIZE)
ax.set_aspect('equal')
ax.set_xlabel("mm")
ax.set_ylabel("mm")

title_text = (
    "RVE Voronoi 2D\n\n"
    f"RVE size = {RVE_SIZE} mm\n\n"
    f"Ferrita:\n"
    f"  Tamaño grano = {grain_ferrite} mm\n"
    f"  % objetivo = {fraction_ferrite*100:.1f}%\n"
    f"  % real = {fraction_f_real:.2f}%\n\n"
    f"Martensita:\n"
    f"  Tamaño grano = {grain_martensite} mm\n"
    f"  % objetivo = {fraction_martensite*100:.1f}%\n"
    f"  % real = {fraction_m_real:.2f}%"
)

ax.set_title(title_text, fontsize=9)

plt.tight_layout()
plt.savefig("RVE_preview.png", dpi=600)
plt.close()

print("PNG generado: RVE_preview.png")

# =====================================================
# EXPORTAR SCRIPT APDL
# =====================================================

def export_apdl(polygons, filename=None):

    if filename is None:
        filename = get_next_version_filename()

    with open(filename, "w") as f:
        f.write("/PREP7\n")
        f.write("! --- RVE Voronoi generado desde Python ---\n\n")
        
        kp_id = 1
        
        for poly in polygons:
            coords = list(poly.exterior.coords)
            kp_list = []
            
            for x, y in coords[:-1]:
                f.write(f"K,{kp_id},{x},{y},0\n")
                kp_list.append(kp_id)
                kp_id += 1
            
            kp_string = ",".join(str(k) for k in kp_list)
            f.write(f"A,{kp_string}\n\n")
        
        f.write("NUMMRG,ALL\n")
        f.write("NUMCMP,ALL\n")
        f.write("FINISH\n")

    print(f"Archivo APDL generado: {filename}")

export_apdl(regions)

print("\nProceso completado correctamente.")
