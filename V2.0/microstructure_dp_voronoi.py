import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import shapely
import trimesh
import matplotlib.pyplot as plt
import ezdxf
import os

# ================================
#   CONFIGURACIÓN DEL RVE
# ================================

RVE_SIZE = 500.0    # micras
grain_ferrite = 30.0
grain_martensite = 8.0

fraction_ferrite = 0.75
fraction_martensite = 0.25

min_dist_f = grain_ferrite / 2.0
min_dist_m = grain_martensite / 2.0

area_RVE = RVE_SIZE * RVE_SIZE
area_f = np.pi * min_dist_f**2
area_m = np.pi * min_dist_m**2

N_ferrite = int((area_RVE * fraction_ferrite) / area_f)
N_martensite = int((area_RVE * fraction_martensite) / area_m)

print("Ferrita:", N_ferrite, "Martensita:", N_martensite)

# ================================
#   FUNCIÓN: generar puntos
# ================================

rng = np.random.default_rng(42)

def generar_puntos(n, min_dist):
    pts = []
    intentos = 0
    max_intentos = n * 100

    while len(pts) < n and intentos < max_intentos:
        p = rng.random(2) * RVE_SIZE
        if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in pts):
            pts.append(p)
        intentos += 1

    return np.array(pts)

points_f = generar_puntos(N_ferrite, min_dist_f)
points_m = generar_puntos(N_martensite, min_dist_m)

points = np.vstack([points_f, points_m])
n_total = len(points)

# ================================
#   MIRRORING (9-dominios)
# ================================

offsets = [-RVE_SIZE, 0.0, RVE_SIZE]
ext_points = []

for ox in offsets:
    for oy in offsets:
        for p in points:
            ext_points.append(p + np.array([ox, oy]))

ext_points = np.array(ext_points)
vor = Voronoi(ext_points)
domain = box(0, 0, RVE_SIZE, RVE_SIZE)

# ================================
#   FIX de celdas Voronoi
# ================================

def fix_polygon(p):
    if p.is_empty:
        return None

    # corregir autointersecciones
    p = p.buffer(0)

    # eliminar agujeros
    if hasattr(p, "interiors") and len(p.interiors) > 0:
        p = Polygon(p.exterior)

    if p.area < 1e-6:
        return None

    return p


# ================================
#   EXTRAER SOLO EL BLOQUE CENTRAL
# ================================

regions = []
start = 4 * n_total
end = 5 * n_total
indices_centro = np.arange(start, end)

for local_idx, global_idx in enumerate(indices_centro):
    region_id = vor.point_region[global_idx]
    region = vor.regions[region_id]

    if not region or -1 in region:
        continue

    poly = Polygon([vor.vertices[v] for v in region])
    poly_clip = poly.intersection(domain)

    poly_fixed = fix_polygon(poly_clip)
    if poly_fixed:
        regions.append((local_idx, poly_fixed))

# separación ferrita / martensita
ferrite_polys = [p for (i, p) in regions if i < len(points_f)]
martensite_polys = [p for (i, p) in regions if i >= len(points_f)]

print("Ferrita:", len(ferrite_polys), "Martensita:", len(martensite_polys))

# ================================
#   EXPORTAR DXF 2D
# ================================

doc = ezdxf.new()
msp = doc.modelspace()

for poly in ferrite_polys:
    coords = list(poly.exterior.coords)
    msp.add_lwpolyline(coords, close=True, dxfattribs={"color": 8, "layer": "FERRITE"})

for poly in martensite_polys:
    coords = list(poly.exterior.coords)
    msp.add_lwpolyline(coords, close=True, dxfattribs={"color": 1, "layer": "MARTENSITE"})

doc.saveas("RVE_DP_v2.dxf")

# ================================
#   PNG PREVIEW
# ================================

plt.figure(figsize=(7,7))
for p in ferrite_polys:
    x,y = p.exterior.xy
    plt.fill(x,y,color="#63639A")
for p in martensite_polys:
    x,y = p.exterior.xy
    plt.fill(x,y,color="#C6C6C6")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.savefig("RVE_DP_v2.png", dpi=300, bbox_inches="tight")
plt.close()

# ================================
#   EXTRUSIÓN MANUAL → STL
# ================================

def earclip(coords):
    poly = Polygon(coords)
    tri = shapely.ops.triangulate(poly)
    triangles = []
    for t in tri:
        x,y = t.exterior.xy
        triangles.append(list(zip(x[:-1], y[:-1])))
    return triangles

def extrude_shapely(poly, h=100.0):
    coords = list(poly.exterior.coords)
    tris = earclip(coords)

    verts = []
    faces = []

    # bottom
    for x,y in coords[:-1]:
        verts.append([x,y,0])
    # top
    for x,y in coords[:-1]:
        verts.append([x,y,h])

    n = len(coords)-1

    # triangles bottom/top
    idx = 0
    for tri in tris:
        ids = []
        for (x,y) in tri:
            ids.append(coords.index((x,y)))
        i,j,k = ids
        faces.append([i,k,j])
        faces.append([i+n,j+n,k+n])

    # paredes
    for i in range(n):
        j = (i+1)%n
        faces.append([i,j,j+n])
        faces.append([i,j+n,i+n])

    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))

def combine_meshes(poly_list):
    meshes = []
    for p in poly_list:
        m = extrude_shapely(p, h=100.0)
        if m: meshes.append(m)
    return trimesh.util.concatenate(meshes)

mesh_f = combine_meshes(ferrite_polys)
mesh_m = combine_meshes(martensite_polys)

mesh_f.export("RVE_ferrite.stl")
mesh_m.export("RVE_martensite.stl")

print("\nArchivos generados:")
print(" - RVE_DP_v2.dxf")
print(" - RVE_DP_v2.png")
print(" - RVE_ferrite.stl")
print(" - RVE_martensite.stl")
print("\nRVE generado sin huecos ✔")
