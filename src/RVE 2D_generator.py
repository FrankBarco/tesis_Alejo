import numpy as np
import os
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURACION
# =====================================================

np.random.seed(1)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

RVE_SIZE = 0.5
area_RVE = RVE_SIZE**2

target_d_f = 0.03
target_d_m = 0.008

fraction_f = 0.75
fraction_m = 0.25

tolerance = 0.05
max_iterations = 15

# =====================================================
# ESTIMACION INICIAL
# =====================================================

def estimate_N(target_d, fraction):
    target_area = np.pi*(target_d**2)/4
    return int((area_RVE*fraction)/target_area)

N_f = estimate_N(target_d_f, fraction_f)
N_m = estimate_N(target_d_m, fraction_m)

# =====================================================
# FUNCIONES
# =====================================================

def generate_points(n):
    return np.random.rand(n,2)*RVE_SIZE

def build_periodic_voronoi(points, N_f):

    # Replicar semillas 3x3
    shifts = [-RVE_SIZE, 0, RVE_SIZE]
    all_points = []

    for dx in shifts:
        for dy in shifts:
            shifted = points + np.array([dx,dy])
            all_points.append(shifted)

    all_points = np.vstack(all_points)

    vor = Voronoi(all_points)
    bbox = box(0,0,RVE_SIZE,RVE_SIZE)

    regions = []
    phases = []

    original_indices = range(len(points))
    central_offset = len(points)*4  # bloque central en 3x3

    for i in original_indices:
        region_index = vor.point_region[i + central_offset]
        region = vor.regions[region_index]

        if len(region) == 0:
            continue

        polygon = Polygon([vor.vertices[v] for v in region if v != -1])
        clipped = polygon.intersection(bbox)

        if clipped.is_valid and clipped.area > 1e-12:
            regions.append(clipped)
            if i < N_f:
                phases.append("ferrite")
            else:
                phases.append("martensite")

    return regions, phases

def equivalent_diameter(area):
    return np.sqrt(4*area/np.pi)

# =====================================================
# ITERACION ESTADISTICA
# =====================================================

for iteration in range(max_iterations):

    points_f = generate_points(N_f)
    points_m = generate_points(N_m)
    points = np.vstack((points_f,points_m))

    regions, phases = build_periodic_voronoi(points, N_f)

    d_f=[]
    d_m=[]

    for poly,phase in zip(regions,phases):
        d = equivalent_diameter(poly.area)
        if phase=="ferrite":
            d_f.append(d)
        else:
            d_m.append(d)

    if len(d_f)==0 or len(d_m)==0:
        print("Iteracion invalida, regenerando...")
        continue

    mean_f=np.mean(d_f)
    mean_m=np.mean(d_m)

    error_f = abs(mean_f-target_d_f)/target_d_f
    error_m = abs(mean_m-target_d_m)/target_d_m

    print(f"\nIter {iteration+1}")
    print("Ferrita mean:",mean_f,"error:",error_f)
    print("Martensita mean:",mean_m,"error:",error_m)

    if error_f < tolerance and error_m < tolerance:
        print("Convergencia alcanzada.")
        break

    N_f = int(N_f*(mean_f/target_d_f)**2)
    N_m = int(N_m*(mean_m/target_d_m)**2)

# =====================================================
# FRACCION REAL
# =====================================================

area_f_real = sum(poly.area for poly,phase in zip(regions,phases) if phase=="ferrite")
area_m_real = sum(poly.area for poly,phase in zip(regions,phases) if phase=="martensite")

fraction_f_real = area_f_real/area_RVE
fraction_m_real = area_m_real/area_RVE

# =====================================================
# MICROESTRUCTURA
# =====================================================

plt.figure(figsize=(6,6))
for poly,phase in zip(regions,phases):
    x,y = poly.exterior.xy
    if phase=="ferrite":
        plt.fill(x,y,color="lightgray")
    else:
        plt.fill(x,y,color="black")

plt.xlim(0,RVE_SIZE)
plt.ylim(0,RVE_SIZE)
plt.gca().set_aspect('equal')
plt.title("Periodic RVE Voronoi")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"RVE_microstructure.png"),dpi=600)
plt.close()

# =====================================================
# HISTOGRAMA
# =====================================================

plt.figure(figsize=(6,4))
plt.hist(d_f,bins=20,alpha=0.7,label="Ferrita")
plt.hist(d_m,bins=20,alpha=0.7,label="Martensita")
plt.axvline(target_d_f,linestyle='--')
plt.axvline(target_d_m,linestyle='--')
plt.xlabel("Diametro equivalente (mm)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Histogram_grain_size.png"),dpi=600)
plt.close()

# =====================================================
# RESULTADOS TXT
# =====================================================

with open(os.path.join(output_dir,"Resultados_estadisticos.txt"),"w") as f:
    f.write("=== RESULTADOS FINALES ===\n")
    f.write(f"Ferrita promedio: {mean_f}\n")
    f.write(f"Martensita promedio: {mean_m}\n")
    f.write(f"Fraccion ferrita real: {fraction_f_real}\n")
    f.write(f"Fraccion martensita real: {fraction_m_real}\n")
    f.write(f"N final ferrita: {N_f}\n")
    f.write(f"N final martensita: {N_m}\n")

print("\nModelo Voronoi Periodico generado correctamente.")
print("Resultados guardados en carpeta 'outputs'")
