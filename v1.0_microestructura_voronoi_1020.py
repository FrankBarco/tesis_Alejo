"""
microestructura_voronoi_1020.py

Genera y representa gráficamente una microestructura Voronoi para acero 1020 dúplex
(75% Ferrita clara, 25% Martensita oscura) en un dominio circular de radio r=3.0 mm.

Requisitos: numpy, scipy, shapely, matplotlib
Instalación: pip install numpy scipy shapely matplotlib
"""

import numpy as np
import random
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import math
import os

# ---------------------------
# Parámetros configurables
# ---------------------------
RADIUS = 3.0                       # mm (A = pi * R^2 ≈ 28.274333...)
AREA_TOTAL = math.pi * RADIUS**2
TARGET_FERRITA = 0.75              # fracción por área (Ferrita clara)
TARGET_MARTENSITA = 1.0 - TARGET_FERRITA
TOLERANCE = 0.01                   # tolerancia ±1 (0.01 -> ±1%)
N_SITES = 300                      # número de semillas Voronoi (ajustable)
LLOYD_ITER = 3                     # número de iteraciones de relajación de Lloyd
SEED = 1234                        # semilla reproducible
OUT_BASENAME = "micro_voronoi_1020" # nombre base para archivos de salida
SAVE_PNG = True
SAVE_SVG = True

# ---------------------------
# Funciones auxiliares
# ---------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def random_points_in_circle(n, radius):
    """Genera n puntos uniformes dentro de un círculo (radio 'radius')."""
    r = radius * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack((x, y)).T

def voronoi_finite_polygons_2d(vor, radius=1e3):
    """
    Convierte regiones infinitas de scipy.spatial.Voronoi a polígonos finitos.
    Basado en receta común (adaptado).
    Devuelve lista de regiones (listas de vértices) y array de vértices.
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    # un radio suficientemente grande para cerrar las regiones infinitas
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # mapa punto -> regiones
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            # región finita
            new_regions.append(vertices)
            continue

        # región infinita: reconstruir
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0 or v1 < 0:
                # arista infinita: calcular un punto final lejano
                if v1 >= 0:
                    v = vor.vertices[v1]
                else:
                    v = vor.vertices[v2]

                # dirección perpendicular a la arista entre los puntos
                tangent = vor.points[p2] - vor.points[p1]
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = v + direction * radius

                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

        # ordenar los índices de vértice de la nueva región en sentido antihorario
        vs = np.asarray([new_vertices[i] for i in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [v for _,v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)

def clip_polygons_to_circle(regions, vertices, radius):
    """Recorta polígonos Voronoi a un círculo de radio 'radius' centrado en 0,0.
       Devuelve lista de shapely polygons (posiblemente vacíos si clipped to nothing)."""
    circle = Point(0,0).buffer(radius, resolution=256)
    poly_list = []
    for region in regions:
        pts = vertices[region]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(circle)
        if clipped.is_empty:
            continue
        # puede devolver MultiPolygon; lo convertimos a polígonos separados
        if clipped.geom_type == 'Polygon':
            poly_list.append(clipped)
        else:
            for p in clipped:
                if p.area > 1e-12:
                    poly_list.append(p)
    return poly_list

def lloyd_relaxation(points, n_iter, radius):
    """Aplica iteraciones de Lloyd: calcular Voronoi -> centroides dentro del círculo -> reemplazar puntos."""
    pts = points.copy()
    for i in range(n_iter):
        vor = Voronoi(pts)
        regions, vertices = voronoi_finite_polygons_2d(vor, radius=radius*10)
        polygons = clip_polygons_to_circle(regions, vertices, radius)
        # calcular centroides y reemplazar (manteniendo mismo número de puntos: si alguna celda desaparece, ignorar)
        new_pts = []
        # Mapeo: asumiendo orden similar a vor.points, pero safe method: for each original point find corresponding polygon by centroid closeness
        # Construimos centroid list and pick nearest centroid per point
        centroids = [p.centroid.coords[0] for p in polygons]
        centroids = np.array(centroids) if len(centroids)>0 else np.zeros((0,2))
        if len(centroids)==0:
            break
        # Para asignación, simplemente samplear n original points by nearest centroid
        # Si cantidad de centroides < pts, re-muestreamos
        if len(centroids) >= len(pts):
            # asignar primer len(pts) centroides aleatoriamente
            idx = np.random.choice(len(centroids), size=len(pts), replace=False)
            new_pts = centroids[idx]
        else:
            # si menos centroides que puntos (raro), rellenar con centroides repetidos
            idx = np.random.choice(len(centroids), size=len(pts), replace=True)
            new_pts = centroids[idx]
        pts = new_pts
    return pts

def chaikin_smooth(coords, iterations=2):
    """
    Suavizado Chaikin para una lista de coordenadas de un polígono cerrado.
    coords: array-like de forma (N,2)
    """
    coords = np.asarray(coords)
    # aseguramos que esté cerrado
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    for _ in range(iterations):
        new = []
        n = len(coords)
        for i in range(n-1):
            p0 = coords[i]
            p1 = coords[i+1]
            q = 0.75*p0 + 0.25*p1
            r = 0.25*p0 + 0.75*p1
            new.append(q)
            new.append(r)
        new.append(new[0])  # cerrar
        coords = np.array(new)
    return coords

def assign_phases_by_area(polygons, target_mart_area, seed=None):
    """
    Asigna un conjunto de polígonos como martensita (oscura) y el resto ferrita (clara)
    intentando que el área total de martensita se aproxime a target_mart_area.
    Estrategia: barajar polígonos (reproducible) y acumular hasta aproximar objetivo (heurística).
    """
    if seed is not None:
        random.seed(seed)
    idxs = list(range(len(polygons)))
    random.shuffle(idxs)
    assigned_mart = set()
    acc = 0.0
    for i in idxs:
        a = polygons[i].area
        # si agregarlo no excede demasiado (pero permitir superar si aún lejos)
        if acc + a <= target_mart_area or acc < target_mart_area*0.5:
            assigned_mart.add(i)
            acc += a
        else:
            # si estamos aún lejos y el elemento es muy grande, aceptarlo
            if acc < target_mart_area*0.95:
                assigned_mart.add(i)
                acc += a
            else:
                # ya cerca, saltamos
                continue
        # si ya sobrepasamos el objetivo por más de un pequeño margen, romper
        if acc >= target_mart_area:
            break
    # si no llegamos, intentar añadir más pequeños hasta acercarnos
    if acc < target_mart_area:
        remaining = sorted([i for i in range(len(polygons)) if i not in assigned_mart],
                            key=lambda j: polygons[j].area)
        for j in remaining:
            if acc >= target_mart_area:
                break
            assigned_mart.add(j)
            acc += polygons[j].area
    return assigned_mart, acc

# ---------------------------
# Pipeline principal
# ---------------------------

def generate_microstructure(radius=RADIUS, n_sites=N_SITES, lloyd_iter=LLOYD_ITER,
                            target_ferrita=TARGET_FERRITA, seed=SEED, tol=TOLERANCE):
    set_seed(seed)

    # 1) generar puntos iniciales dentro del círculo
    pts = random_points_in_circle(n_sites, radius)

    # 2) aplicar relajación de Lloyd (centroides)
    pts = lloyd_relaxation(pts, lloyd_iter, radius)

    # 3) construir Voronoi y regiones finitas
    vor = Voronoi(pts)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=radius*10)
    polygons = clip_polygons_to_circle(regions, vertices, radius)

    # Asegurarse de que las áreas sumen el área del círculo (con tolerancia numérica)
    total_area_polys = sum(p.area for p in polygons)
    # si falta o sobra por numerics, escalar áreas no es correcto; pero el union de polígonos debería igualar el círculo
    # calculamos el círculo
    circle = Point(0,0).buffer(radius, resolution=256)
    circle_area = circle.area

    # 4) asignación de fases por área
    target_mart_area = circle_area * (1.0 - target_ferrita)
    assigned_mart, mart_area = assign_phases_by_area(polygons, target_mart_area, seed=seed)

    ferrita_area = sum(polygons[i].area for i in range(len(polygons)) if i not in assigned_mart)
    # porcentajes reales
    real_mart_frac = mart_area / circle_area
    real_ferr_frac = ferrita_area / circle_area

    return {
        "polygons": polygons,
        "assigned_mart": assigned_mart,
        "radius": radius,
        "circle": circle,
        "real_mart_frac": real_mart_frac,
        "real_ferr_frac": real_ferr_frac,
        "circle_area": circle_area
    }

def plot_microstructure(result, smooth_iterations=2, figsize=(8,8), show=True,
                        save_png=SAVE_PNG, save_svg=SAVE_SVG, out_basename=OUT_BASENAME):
    polygons = result["polygons"]
    assigned_mart = result["assigned_mart"]
    circle = result["circle"]
    r = result["radius"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(-r*1.05, r*1.05)
    ax.set_ylim(-r*1.05, r*1.05)
    ax.axis('off')

    # Colors: Ferrita (clara), Martensita (oscura)
    color_ferr = "#f2f0e6"  # beige claro
    color_mart = "#2b2b2b"  # gris muy oscuro

    # dibujar cada polígono suavizado
    patches_drawn = []
    for i, poly in enumerate(polygons):
        # suavizar contorno con Chaikin (obtén coords)
        exterior = np.array(poly.exterior.coords)
        smooth_coords = chaikin_smooth(exterior, iterations=smooth_iterations)
        xs, ys = smooth_coords[:,0], smooth_coords[:,1]
        if i in assigned_mart:
            ax.fill(xs, ys, facecolor=color_mart, edgecolor='k', linewidth=0.4)
        else:
            ax.fill(xs, ys, facecolor=color_ferr, edgecolor='k', linewidth=0.4)

    # Dibujar borde circular más grueso
    xcircle, ycircle = circle.exterior.xy
    ax.plot(xcircle, ycircle, color='k', linewidth=1.0)

    # Leyenda con porcentajes reales
    real_f = result["real_ferr_frac"] * 100.0
    real_m = result["real_mart_frac"] * 100.0
    # Crear leyenda manual
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=color_ferr, edgecolor='k', label=f'Ferrita: {real_f:.2f}%'),
        Patch(facecolor=color_mart, edgecolor='k', label=f'Martensita: {real_m:.2f}%')
    ]
    ax.legend(handles=legend_patches, loc='upper right', framealpha=0.9)

    # Guardar archivos
    if save_png:
        png_name = f"{out_basename}.png"
        plt.savefig(png_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
        print(f"Guardado PNG: {png_name}")
    if save_svg:
        svg_name = f"{out_basename}.svg"
        plt.savefig(svg_name, format='svg', bbox_inches='tight', pad_inches=0.02)
        print(f"Guardado SVG: {svg_name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------------------------
# Ejecutar pipeline (si se ejecuta como script)
# ---------------------------
if __name__ == "__main__":
    # parámetros visibles para cambio rápido
    seed = SEED
    n_sites = N_SITES
    lloyd_iter = LLOYD_ITER
    tol = TOLERANCE
    radius = RADIUS

    print("Generando microestructura Voronoi para acero 1020 dúplex")
    print(f"Radio = {radius} mm, Área objetivo = {AREA_TOTAL:.5f} mm^2")
    print(f"Semillas = {n_sites}, Lloyd iter = {lloyd_iter}, seed = {seed}")

    result = generate_microstructure(radius=radius, n_sites=n_sites, lloyd_iter=lloyd_iter,
                                     target_ferrita=TARGET_FERRITA, seed=seed, tol=tol)

    print(f"Porcentajes reales: Ferrita = {result['real_ferr_frac']*100:.3f}% | Martensita = {result['real_mart_frac']*100:.3f}%")
    # comprobar tolerancia
    ferr_err = abs(result['real_ferr_frac'] - TARGET_FERRITA)
    if ferr_err <= tol:
        print(f"La fracción de ferrita está dentro de la tolerancia: error = {ferr_err*100:.3f}%")
    else:
        print(f"Atención: la fracción de ferrita está fuera de la tolerancia: error = {ferr_err*100:.3f}%")
        print("Puedes aumentar N_SITES o ajustar la heurística de asignación de fase para mejorar la aproximación.")

    # mostrar y guardar
    plot_microstructure(result, smooth_iterations=2, figsize=(6,6), show=True,
                        save_png=SAVE_PNG, save_svg=SAVE_SVG, out_basename=OUT_BASENAME)
