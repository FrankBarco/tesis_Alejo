import numpy as np
import random
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import os

# =====================================================
# CONFIGURACIÓN
# =====================================================

OUTPUT_DIR = r"D:\37-Alejo U\Generar_Voronio_2D\outputs\APDL_inp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_next_version_filename(base_name="RVE_APDL_v", extension=".inp"):
    version = 1
    while os.path.exists(os.path.join(OUTPUT_DIR, f"{base_name}{version}{extension}")):
        version += 1
    return os.path.join(OUTPUT_DIR, f"{base_name}{version}{extension}")

RVE_SIZE = 0.5

grain_ferrite = 0.03
grain_martensite = 0.008

fraction_ferrite = 0.75
fraction_martensite = 0.25

min_dist_f = grain_ferrite / 2
min_dist_m = grain_martensite / 2

area_RVE = RVE_SIZE**2
area_f_seed = np.pi * min_dist_f**2
area_m_seed = np.pi * min_dist_m**2

N_ferrite = int((area_RVE * fraction_ferrite) / area_f_seed)
N_martensite = int((area_RVE * fraction_martensite) / area_m_seed)

print("Ferrita:", N_ferrite)
print("Martensita:", N_martensite)

# =====================================================
# GENERAR SEMILLAS
# =====================================================

def generate_points(n, min_dist):
    points = []
    attempts = 0
    max_attempts = n * 1000

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
# VORONOI FINITO
# =====================================================

def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = RVE_SIZE * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

vor = Voronoi(points)
regions_finite, vertices = voronoi_finite_polygons_2d(vor)

bbox = box(0, 0, RVE_SIZE, RVE_SIZE)

regions = []
region_phase = []

for i, region in enumerate(regions_finite):
    polygon = Polygon(vertices[region])

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    clipped = polygon.intersection(bbox)

    if clipped.is_empty:
        continue

    if not clipped.is_valid:
        clipped = clipped.buffer(0)

    if clipped.area < 1e-12:
        continue

    regions.append(clipped)

    if i < len(points_f):
        region_phase.append("ferrite")
    else:
        region_phase.append("martensite")

# =====================================================
# GARANTIZAR DOMINIO COMPLETO
# =====================================================

union_geom = unary_union(regions)
missing = bbox.difference(union_geom)

if not missing.is_empty:
    print("Rellenando huecos...")

    if missing.geom_type == "Polygon":
        regions.append(missing)
        region_phase.append("ferrite")

    elif missing.geom_type == "MultiPolygon":
        for sub in missing.geoms:
            regions.append(sub)
            region_phase.append("ferrite")

print("Regiones finales:", len(regions))

# =====================================================
# EXPORTAR APDL
# =====================================================

def export_apdl(polygons, phases):

    filename = get_next_version_filename()

    with open(filename, "w") as f:

        f.write("/PREP7\n")

        # ---- Material 1 Ferrita ----
        f.write("MP,EX,1,210000\n")
        f.write("MP,PRXY,1,0.3\n")
        f.write("TB,GURSON,1\n")
        f.write("TBDATA,1,1.5,1.0,2.25\n\n")

        # ---- Material 2 Martensita ----
        f.write("MP,EX,2,230000\n")
        f.write("MP,PRXY,2,0.28\n")
        f.write("TB,GURSON,2\n")
        f.write("TBDATA,1,1.8,1.0,3.24\n\n")

        kp_id = 1

        for poly, phase in zip(polygons, phases):

            if poly.geom_type == "MultiPolygon":
                sub_polys = list(poly.geoms)
            else:
                sub_polys = [poly]

            for sub in sub_polys:

                coords = list(sub.exterior.coords)
                kp_list = []

                for x, y in coords[:-1]:
                    f.write(f"K,{kp_id},{x},{y},0\n")
                    kp_list.append(kp_id)
                    kp_id += 1

                kp_string = ",".join(str(k) for k in kp_list)
                f.write(f"A,{kp_string}\n")

                f.write("*GET,LASTA,AREA,0,NUM,MAX\n")
                f.write("ASEL,S,AREA,,LASTA\n")

                if phase == "ferrite":
                    f.write("AATT,1,1,1\n")
                else:
                    f.write("AATT,2,1,1\n")

                f.write("ALLSEL,ALL\n")

        # Limpieza
        f.write("BTOL,1E-4\n")
        f.write("NUMMRG,KP\n")
        f.write("NUMMRG,LINE\n")
        f.write("NUMCMP,ALL\n")

        # Mallado
        f.write("ET,1,PLANE182\n")
        f.write("KEYOPT,1,3,3\n")
        f.write("ESIZE,0.0025\n")
        f.write("ALLSEL,ALL\n")
        f.write("TYPE,1\n")
        f.write("AMESH,ALL\n")

        f.write("FINISH\n")

    print("Archivo generado:", filename)

export_apdl(regions, region_phase)
print("Proceso completado correctamente.")