# ============================================================
# VERSIONAMIENTO AUTOMÁTICO POR CARPETA
# ============================================================

import os
import re

BASE_DIR = r"D:\37-Alejo U\Generar_Voroni 3D\Step"

def get_next_version_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and re.match(r"RVE_v\d+", d)
    ]

    if not existing:
        next_version = 1
    else:
        versions = [int(re.findall(r"\d+", d)[0]) for d in existing]
        next_version = max(versions) + 1

    new_folder = os.path.join(base_dir, f"RVE_v{next_version}")
    os.makedirs(new_folder)

    return new_folder

VERSION_FOLDER = get_next_version_folder(BASE_DIR)

print(f"✔ Carpeta creada: {VERSION_FOLDER}")


# ============================================================
#  RVE DP STEEL – Voronoi → STEP (CAD sólido) + PNG preview
#  Unidades: mm
# ============================================================

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import shapely.ops
import matplotlib.pyplot as plt

# -------- OpenCascade --------
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeFace
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

# ============================================================
# CONFIGURACIÓN DEL RVE
# ============================================================

RVE_SIZE = 0.3       # mm (500 µm)
THICKNESS = 0.1       # mm (espesor 3D)

grain_ferrite = 0.02
grain_martensite = 0.01

fraction_ferrite = 0.75
fraction_martensite = 0.25

min_dist_f = grain_ferrite / 2.0
min_dist_m = grain_martensite / 2.0

area_RVE = RVE_SIZE ** 2
area_f = np.pi * min_dist_f**2
area_m = np.pi * min_dist_m**2

N_ferrite = int((area_RVE * fraction_ferrite) / area_f)
N_martensite = int((area_RVE * fraction_martensite) / area_m)

print("Ferrita:", N_ferrite, "Martensita:", N_martensite)

# ============================================================
# GENERACIÓN DE PUNTOS (Poisson-disc)
# ============================================================

rng = np.random.default_rng(42)

def generar_puntos(n, min_dist):
    pts = []
    intentos = 0
    max_intentos = n * 300

    while len(pts) < n and intentos < max_intentos:
        p = rng.random(2) * RVE_SIZE
        if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in pts):
            pts.append(p)
        intentos += 1

    return np.array(pts)

points_f = generar_puntos(N_ferrite, min_dist_f)
points_m = generar_puntos(N_martensite, min_dist_m)
points = np.vstack([points_f, points_m])

# ============================================================
# LLOYD RELAXATION (Voronoi más realista)
# ============================================================

def lloyd_relaxation(points, iterations=3):

    for _ in range(iterations):

        offsets = [-RVE_SIZE, 0, RVE_SIZE]
        ext_points = []

        for ox in offsets:
            for oy in offsets:
                for p in points:
                    ext_points.append(p + np.array([ox, oy]))

        ext_points = np.array(ext_points)

        vor = Voronoi(ext_points)

        new_points = []

        for i in range(len(points)):

            region = vor.regions[vor.point_region[i + 4*len(points)]]

            if -1 in region or len(region) == 0:
                new_points.append(points[i])
                continue

            polygon = Polygon([vor.vertices[v] for v in region])
            polygon = polygon.intersection(box(0,0,RVE_SIZE,RVE_SIZE))

            if polygon.area > 0:
                new_points.append(np.array(polygon.centroid.coords[0]))
            else:
                new_points.append(points[i])

        points = np.array(new_points)

    return points


points = lloyd_relaxation(points, iterations=4)
# ============================================================
# VORONOI CON MIRRORING
# ============================================================

offsets = [-RVE_SIZE, 0.0, RVE_SIZE]
ext_points = []

for ox in offsets:
    for oy in offsets:
        for p in points:
            ext_points.append(p + np.array([ox, oy]))

ext_points = np.array(ext_points)
vor = Voronoi(ext_points)

domain = box(0, 0, RVE_SIZE, RVE_SIZE)

def fix_polygon(p):

    if p.is_empty:
        return None

    # corregir geometría
    p = p.buffer(0)

    # eliminar agujeros
    if hasattr(p, "interiors") and len(p.interiors) > 0:
        p = Polygon(p.exterior)

    # eliminar áreas muy pequeñas
    if p.area < 1e-12:
        return None

    # ------------------------------------------------
    # SUAVIZADO DE BORDES (microestructura más realista)
    # ------------------------------------------------
    smooth = RVE_SIZE * 0.002
    p = p.buffer(smooth).buffer(-smooth)

    return p

# ============================================================
# EXTRAER CELDAS CENTRALES
# ============================================================

regions = []
n_total = len(points)
start = 4 * n_total
end = 5 * n_total

for local_idx, global_idx in enumerate(range(start, end)):
    region_id = vor.point_region[global_idx]
    region = vor.regions[region_id]

    if not region or -1 in region:
        continue

    poly = Polygon([vor.vertices[v] for v in region])
    poly = poly.intersection(domain)
    poly = fix_polygon(poly)

    if poly:
        regions.append((local_idx, poly))

# ============================================================
# FILTRAR GRANOS PROBLEMÁTICOS (mejora mallado en ANSYS)
# ============================================================

# ------------------------------------------------------------
# 1) FILTRO DE ÁREA MÍNIMA (elimina granos muy pequeños)
# ------------------------------------------------------------

grain_areas = np.array([poly.area for (_, poly) in regions])

min_area = 0.05 * np.mean(grain_areas)

regions = [(i, poly) for (i, poly) in regions if poly.area > min_area]

print("Granos después del filtro de área:", len(regions))


# ------------------------------------------------------------
# 2) FILTRO DE FORMA (elimina granos muy alargados)
# ------------------------------------------------------------

def aspect_ratio(poly):
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    return max(w, h) / (min(w, h) + 1e-12)

regions = [(i, poly) for (i, poly) in regions if aspect_ratio(poly) < 8]

print("Granos después del filtro de forma:", len(regions))


# ------------------------------------------------------------
# 3) SEPARAR FASES
# ------------------------------------------------------------

ferrite_polys = [p for (i, p) in regions if i < len(points_f)]
martensite_polys = [p for (i, p) in regions if i >= len(points_f)]

print(
    "Regiones finales → Ferrita:", len(ferrite_polys),
    "Martensita:", len(martensite_polys)
)

# ============================================================
# PREVISUALIZACIÓN PNG
# ============================================================

plt.figure(figsize=(7, 7))

for p in ferrite_polys:
    x, y = p.exterior.xy
    plt.fill(x, y, color="#4C72B0", linewidth=0)

for p in martensite_polys:
    x, y = p.exterior.xy
    plt.fill(x, y, color="#DDDDDD", linewidth=0)

plt.gca().set_aspect("equal")
plt.axis("off")
preview_path = os.path.join(VERSION_FOLDER, "RVE_DP_voronoi_preview.png")
plt.savefig(preview_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ PNG generado: RVE_DP_voronoi_preview.png")

# ============================================================
# SHAPELY → SOLID CAD (STEP)
# ============================================================

def shapely_to_solid(poly, thickness):
    coords = list(poly.exterior.coords)

    wire = BRepBuilderAPI_MakePolygon()
    for x, y in coords:
        wire.Add(gp_Pnt(float(x), float(y), 0.0))
    wire.Close()

    face = BRepBuilderAPI_MakeFace(wire.Wire())
    vec = gp_Vec(0, 0, thickness)

    solid = BRepPrimAPI_MakePrism(face.Face(), vec).Shape()
    return solid

def export_step(polys, filename):
    writer = STEPControl_Writer()
    for p in polys:
        try:
            s = shapely_to_solid(p, THICKNESS)
            writer.Transfer(s, STEPControl_AsIs)
        except:
            pass
    writer.Write(filename)

# ============================================================
# EXPORTAR STEP
# ============================================================

ferrite_path = os.path.join(VERSION_FOLDER, "RVE_ferrite_mm.stp")
martensite_path = os.path.join(VERSION_FOLDER, "RVE_martensite_mm.stp")

export_step(ferrite_polys, ferrite_path)
export_step(martensite_polys, martensite_path)

print("✔ STEP exportados:")
print(" - RVE_ferrite_mm.stp")
print(" - RVE_martensite_mm.stp")
print("\n✔ RVE DP 3D CAD listo para ANSYS / GTN")