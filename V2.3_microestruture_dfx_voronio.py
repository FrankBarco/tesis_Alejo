"""
RVE Voronoi periódico (sin huecos) — dominio 0.5 x 0.5 mm
Exporta: DXF (capas FERRITE/MARTENSITE), PNG preview y STL (ferrite/martensite).
Requisitos: numpy, scipy, shapely, trimesh, matplotlib, ezdxf
"""

import numpy as np
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon, box
import shapely
import trimesh
import matplotlib.pyplot as plt
import ezdxf
import os

# -------------------------
# CONFIGURACIÓN (dominio fijo)
# -------------------------
RVE_SIZE = 0.5  # mm (0.5 mm = 500 µm)
# tamaño de grano (entrada en µm, convertimos a mm)
grain_ferrite_um = 30.0
grain_martensite_um = 8.0

grain_ferrite = grain_ferrite_um * 0.001  # mm
grain_martensite = grain_martensite_um * 0.001  # mm

fraction_ferrite = 0.75
fraction_martensite = 0.25

min_dist_f = grain_ferrite / 2.0
min_dist_m = grain_martensite / 2.0

area_RVE = RVE_SIZE * RVE_SIZE
area_f = np.pi * min_dist_f**2
area_m = np.pi * min_dist_m**2

N_ferrite = max(1, int((area_RVE * fraction_ferrite) / area_f))
N_martensite = max(1, int((area_RVE * fraction_martensite) / area_m))

print("Objetivo de granos -> Ferrita:", N_ferrite, " Martensita:", N_martensite)

# -------------------------
# Generación de puntos (simple Poisson-like)
# -------------------------
rng = np.random.default_rng(42)

def generar_puntos(n, min_dist, tries_factor=200):
    pts = []
    max_tries = n * tries_factor
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = rng.random(2) * RVE_SIZE
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            pts.append(p)
        tries += 1
    if len(pts) < n:
        print(f"WARNING: sólo generé {len(pts)} de {n} puntos (aumenta tries_factor).")
    return np.array(pts)

points_f = generar_puntos(N_ferrite, min_dist_f)
points_m = generar_puntos(N_martensite, min_dist_m)
points = np.vstack([points_f, points_m])
n_total = len(points)
print("Puntos reales totales generados:", n_total)

# -------------------------
# Mirroring (3x3 tiling) para voronoi periódico
# -------------------------
offsets = [-RVE_SIZE, 0.0, RVE_SIZE]
ext_points = []
# También guardamos array de (ox,oy) para cada ext_point para debug si hace falta
ext_offsets = []
for ox in offsets:
    for oy in offsets:
        for p in points:
            ext_points.append(p + np.array([ox, oy]))
            ext_offsets.append((ox, oy))
ext_points = np.array(ext_points)
ext_offsets = np.array(ext_offsets)

# Construimos KDTree para localizar la copia central de cada semilla
tree = cKDTree(ext_points)

# calculamos voronoi sobre los puntos extendidos
vor = Voronoi(ext_points)

domain = box(0.0, 0.0, RVE_SIZE, RVE_SIZE)

# -------------------------
# Función para limpiar polígonos
# -------------------------
def fix_polygon(p):
    if p is None:
        return None
    if p.is_empty:
        return None
    p = p.buffer(0)
    if hasattr(p, "interiors") and len(p.interiors) > 0:
        p = Polygon(p.exterior)
    if p.area <= 0:
        return None
    return p

# -------------------------
# EXTRAER LAS CELDAS CORRECTAS (sin huecos)
# Para cada punto ORIGINAL, buscamos su copia exacta en ext_points (la del centro)
# KDTree query devuelve la copia central porque está a distancia 0; otras réplicas están a distancia >= RVE_SIZE.
# -------------------------
regions = []
for idx_orig, p in enumerate(points):
    # buscar la copia más cercana en ext_points (debe ser la copia con offset (0,0))
    dist, idx_ext = tree.query(p, k=1)
    # seguridad: si dist > tiny tol algo salió mal
    if dist > 1e-9:
        # fallback: buscar la copia con offset (0,0)
        # buscamos índice donde ext_offsets == (0,0) y ext_point equals p
        matches = np.where((np.abs(ext_points - p) < 1e-12).all(axis=1))[0]
        if len(matches) > 0:
            idx_ext = matches[0]
        else:
            # si no hay coincidencia exacta, use el más cercano igualmente
            idx_ext = int(idx_ext)

    region_id = vor.point_region[idx_ext]
    region = vor.regions[region_id]
    if not region or -1 in region:
        # región infinita o inválida
        continue

    poly = Polygon([vor.vertices[v] for v in region])
    poly_clipped = poly.intersection(domain)
    poly_fixed = fix_polygon(poly_clipped)
    if poly_fixed:
        regions.append((idx_orig, poly_fixed))

print("Total regiones extraídas (sin huecos):", len(regions))

# -------------------------
# separar ferrita / martensita por índice original
# -------------------------
ferrite_polys = [p for (i, p) in regions if i < len(points_f)]
martensite_polys = [p for (i, p) in regions if i >= len(points_f)]

print("Polígonos finales -> Ferrita:", len(ferrite_polys), "Martensita:", len(martensite_polys))

# -------------------------
# Exportar DXF (capas)
# -------------------------
dxf_name = "RVE_0p5mm_voronoi_nom_gap.dxf"
doc = ezdxf.new()
msp = doc.modelspace()
for poly in ferrite_polys:
    coords = list(poly.exterior.coords)
    msp.add_lwpolyline(coords, close=True, dxfattribs={"layer": "FERRITE"})
for poly in martensite_polys:
    coords = list(poly.exterior.coords)
    msp.add_lwpolyline(coords, close=True, dxfattribs={"layer": "MARTENSITE"})
doc.saveas(dxf_name)
print("DXF guardado:", dxf_name)

# -------------------------
# PNG preview
# -------------------------
png_name = "RVE_0p5mm_voronoi_nom_gap.png"
plt.figure(figsize=(6,6), dpi=200)
for p in ferrite_polys:
    x,y = p.exterior.xy
    plt.fill(x,y, color="#6A6AAF", linewidth=0.2)
for p in martensite_polys:
    x,y = p.exterior.xy
    plt.fill(x,y, color="#DDDDDD", linewidth=0.2)
plt.gca().set_aspect("equal")
plt.axis("off")
plt.savefig(png_name, bbox_inches="tight", pad_inches=0)
plt.close()
print("PNG guardado:", png_name)

# -------------------------
# Extrusión sencilla y export STL (cada fase por separado)
# -------------------------
def triangulate_polygon(poly):
    """triangula la cara plana del polygon usando shapely.ops.triangulate"""
    tris = shapely.ops.triangulate(poly)
    triangles = []
    for t in tris:
        coords = list(t.exterior.coords)[:-1]
        # convert to list of tuples
        triangles.append(coords)
    return triangles

def extrude_polygon_to_trimesh(poly, thickness=0.1):
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) < 3:
        return None
    # construir vértices bottom/top
    verts = []
    for (x,y) in coords:
        verts.append([x, y, 0.0])
    for (x,y) in coords:
        verts.append([x, y, thickness])
    n = len(coords)
    faces = []
    # triangulación de la tapa/bottom usando triangulate
    tris = triangulate_polygon(poly)
    # para cada tri, obtener índices locales en coords
    for tri in tris:
        ids = [coords.index((round(x,12), round(y,12))) if (round(x,12),round(y,12)) in [(round(xx,12),round(yy,12)) for (xx,yy) in coords] else None for (x,y) in tri]
        # fallback simple: construir fan (si triangulate falla con índices)
        # Aquí simplificamos: crear caras de la tapa con fan respecto al primer vértice
    # Usaremos una construcción simple de caras laterales y dos caras poligonales (no trianguladas) convertidas por trimesh
    # Método robusto: usar trimesh.path.extrude_polygon si disponible
    try:
        mesh = trimesh.creation.extrude_polygon(poly, thickness)
        return mesh
    except Exception:
        # fallback: crear una malla muy básica extruida (posible que no sea óptima)
        # construir caras laterales
        for i in range(n):
            i2 = (i+1)%n
            faces.append([i, i2, i2+n])
            faces.append([i, i2+n, i+n])
        # top/bottom triangulación simple (fan)
        for i in range(1, n-1):
            faces.append([0, i, i+1])         # bottom
            faces.append([0+n, i+1+n, i+n])   # top (reversa)
        mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)
        return mesh

# combinar y exportar
def combine_and_export(poly_list, filename, thickness=0.1):
    meshes = []
    for p in poly_list:
        try:
            m = extrude_polygon_to_trimesh(p, thickness=thickness)
            if m is not None and m.volume > 0:
                meshes.append(m)
        except Exception as e:
            # ignora polígonos problemáticos
            print("Warning triangulation/extrusion:", e)
    if len(meshes) == 0:
        print("No mesh para exportar:", filename)
        return
    combo = trimesh.util.concatenate(meshes)
    combo.export(filename)
    print("Exportado:", filename)

# Export STL con 0.1 mm de espesor
combine_and_export(ferrite_polys, "RVE_ferrite_0p5mm.stl", thickness=0.1)
combine_and_export(martensite_polys, "RVE_martensite_0p5mm.stl", thickness=0.1)

print("\nHECHO. Revisa los archivos generados en el directorio actual:")
print(" -", dxf_name)
print(" -", png_name)
print(" - RVE_ferrite_0p5mm.stl")
print(" - RVE_martensite_0p5mm.stl")
print("\nIMPORTANTE: Si importas el DXF en SpaceClaim y te aparece a 12.7 mm, cambia la unidad de importación a MILIMÉTROS (no PULGADAS).")
