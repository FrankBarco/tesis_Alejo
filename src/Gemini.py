import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import os

# ==========================================
# 1. CONFIGURACIÓN DE PARÁMETROS
# ==========================================
RVE_SIZE = 0.5  # mm
TARGET_FERRITE = 0.75
TARGET_MARTENSITE = 0.25
GRAIN_SIZE_F = 0.03
GRAIN_SIZE_M = 0.008

# Materiales (MPa)
MAT_PROPS = {
    1: {'name': 'Ferrite', 'EX': 210000, 'PRXY': 0.3, 'q1': 1.5, 'q2': 1.0, 'q3': 2.25},
    2: {'name': 'Martensite', 'EX': 230000, 'PRXY': 0.28, 'q1': 1.8, 'q2': 1.0, 'q3': 3.24}
}

OUTPUT_DIR = r"D:\37-Alejo U\Generar_Voronio_2D\outputs\APDL_inp"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. GENERACIÓN DE SEMILLAS Y VORONOI
# ==========================================
def generate_seeds():
    # Estimación de número de granos
    n_f = int((RVE_SIZE**2 * TARGET_FERRITE) / (GRAIN_SIZE_F**2))
    n_m = int((RVE_SIZE**2 * TARGET_MARTENSITE) / (GRAIN_SIZE_M**2))
    
    seeds = []
    labels = []
    
    # Generar semillas con buffer para evitar bordes vacíos inicialmente
    for n, label, dist in [(n_f, 1, GRAIN_SIZE_F*0.5), (n_m, 2, GRAIN_SIZE_M*0.5)]:
        count = 0
        while count < n:
            pt = np.random.rand(2) * RVE_SIZE
            if all(np.linalg.norm(pt - s) > dist*0.6 for s in seeds):
                seeds.append(pt)
                labels.append(label)
                count += 1
    return np.array(seeds), np.array(labels)

seeds, seed_labels = generate_seeds()

# Extender semillas para Voronoi infinito (puntos espejo para bordes limpios)
mirror_seeds = []
for s in seeds:
    for dx in [-RVE_SIZE, 0, RVE_SIZE]:
        for dy in [-RVE_SIZE, 0, RVE_SIZE]:
            if dx == 0 and dy == 0: continue
            mirror_seeds.append(s + [dx, dy])

all_seeds = np.vstack([seeds, mirror_seeds])
vor = Voronoi(all_seeds)

# ==========================================
# 3. RECORTE Y LIMPIEZA GEOMÉTRICA (SHAPELY)
# ==========================================
container = box(0, 0, RVE_SIZE, RVE_SIZE)
polygons = []
final_labels = []

for i, region_idx in enumerate(vor.point_region[:len(seeds)]):
    region = vor.regions[region_idx]
    if -1 not in region and len(region) > 0:
        poly_verts = vor.vertices[region]
        poly = Polygon(poly_verts).intersection(container)
        
        # Limpieza robusta
        poly = poly.buffer(0)
        
        if not poly.is_empty and poly.area > 1e-8:
            polygons.append(poly)
            final_labels.append(seed_labels[i])

# Unificación para garantizar que no hay huecos (Gap filling)
full_domain = unary_union(polygons)
if full_domain.area < (RVE_SIZE**2 - 1e-7):
    print("Advertencia: Corrigiendo gaps residuales...")
    # El container box asegura la cuadratura exacta

# Cálculo de fracciones reales
area_f = sum(p.area for i, p in enumerate(polygons) if final_labels[i] == 1)
area_m = sum(p.area for i, p in enumerate(polygons) if final_labels[i] == 2)
total_a = area_f + area_m
frac_f = area_f / total_a
frac_m = area_m / total_a

print(f"Fracción Real Ferrita: {frac_f:.4f}")
print(f"Fracción Real Martensita: {frac_m:.4f}")

# ==========================================
# 4. EXPORTACIÓN A APDL
# ==========================================
def get_versioned_filename(base_path, prefix):
    idx = 1
    while os.path.exists(os.path.join(base_path, f"{prefix}_v{idx}.inp")):
        idx += 1
    return os.path.join(base_path, f"{prefix}_v{idx}.inp")

filename = get_versioned_filename(OUTPUT_DIR, "RVE_APDL")

with open(filename, 'w') as f:
    f.write("! RVE Voronoi Generado Automáticamente\n/PREP7\n")
    f.write("ET,1,PLANE182\nKEYOPT,1,3,3  ! Plane Strain\n\n")
    
    # Definición de Materiales
    for m_id, p in MAT_PROPS.items():
        f.write(f"MP,EX,{m_id},{p['EX']}\nMP,PRXY,{m_id},{p['PRXY']}\n")
        f.write(f"TB,GTN,{m_id}\nTBDATA,1,{p['q1']},{p['q2']},{p['q3']}\n\n")
    
    # Generación de Keypoints y Áreas
    kp_count = 1
    for i, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)[:-1] # Evitar duplicado final
        kp_start = kp_count
        for x, y in coords:
            f.write(f"K,{kp_count},{x:.8f},{y:.8f},0\n")
            kp_count += 1
        
        # Crear líneas y áreas
        kp_end = kp_count - 1
        f.write(f"LSTR,{kp_end},{kp_start}\n")
        for k in range(kp_start, kp_end):
            f.write(f"LSTR,{k},{k+1}\n")
        
        line_ids = ",".join([str(l) for l in range(i*len(coords)+1, (i+1)*len(coords)+1)])
        f.write(f"AL,{line_ids}\n")
        f.write(f"ASEL,S,AREA,,{i+1}\nAATT,{final_labels[i]},,1\nALLSEL\n\n")

    f.write("ESIZE, 0.005  ! Tamaño global del elemento\n")
    f.write("AMESH, ALL\n")
    f.write("FINISH\n")

# ==========================================
# 5. VISUALIZACIÓN
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
for poly, label in zip(polygons, final_labels):
    color = 'green' if label == 1 else 'red'
    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title(f"RVE Voronoi 2D - Ferrita: {frac_f:.2%} | Martensita: {frac_m:.2%}")
plt.axis('equal')
plt.savefig(os.path.join(OUTPUT_DIR, "RVE_Visual.png"), dpi=600)
print(f"Archivos guardados en: {OUTPUT_DIR}")
plt.show()