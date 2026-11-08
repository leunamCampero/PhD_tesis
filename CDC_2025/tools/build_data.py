"""
Build the input files in ``data/``.  You do NOT need to run this: the outputs are in data/.

Sources
-------
* ../ECC_2025/data/network/grenoble_{nodes,edges}.csv   the same Grenoble network as ECC 2025
                                                       (G_pygmo.pkl: 3452 nodes, 4993 arcs, DoS)
* ../ECC_2025/data/demand/{od_iris_grenoble,iris_zones}.csv   EMD 2020 IRIS x IRIS trips
* _archive_old_files/iris.gpkg                          national IRIS contours (IGN), we keep
                                                       the 70 IRIS of Grenoble (code_insee 38185)

Outputs
-------
data/raw/iris_grenoble.gpkg          the 70 IRIS of Grenoble (EPSG:2154), extracted from iris.gpkg
data/regions/grenoble_iris.geojson   same polygons in EPSG:4326, with zone_id / code_iris / nom_iris
data/network/grenoble_edges.csv      copy from ECC_2025
data/network/grenoble_nodes.csv      ECC_2025 nodes, zone_id re-assigned with the iris.gpkg
                                     polygons (point in polygon, nearest IRIS for the few nodes
                                     outside the city limits); zone weights re-normalised per IRIS
data/demand/*.csv                    copies from ECC_2025

Run:  python tools/build_data.py
"""
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ECC = ROOT.parent / "ECC_2025" / "data"
DATA = ROOT / "data"

for sub in ("network", "demand", "regions", "raw"):
    (DATA / sub).mkdir(parents=True, exist_ok=True)

# ---- copies from ECC_2025 ---------------------------------------------------------------
shutil.copy(ECC / "network" / "grenoble_edges.csv", DATA / "network")
for f in ("od_iris_grenoble.csv", "iris_zones.csv"):
    shutil.copy(ECC / "demand" / f, DATA / "demand")

# ---- the 70 IRIS of Grenoble from the national geopackage --------------------------------
iris = gpd.read_file(ROOT / "_archive_old_files" / "iris.gpkg", where="code_insee = '38185'")
iris.to_file(DATA / "raw" / "iris_grenoble.gpkg", driver="GPKG")
zones = pd.read_csv(DATA / "demand" / "iris_zones.csv")          # zone_id <-> iris_code (OD rows)
code_to_zone = dict(zip(zones.iris_code.astype(str), zones.zone_id))
iris["zone_id"] = iris.code_iris.astype(str).map(code_to_zone)
assert iris.zone_id.notna().all(), "an IRIS of iris.gpkg is missing from the OD matrix"
iris = iris.to_crs(4326)
iris[["zone_id", "code_iris", "nom_iris", "type_iris", "geometry"]].to_file(
    DATA / "regions" / "grenoble_iris.geojson", driver="GeoJSON")

# ---- assign every network node to its IRIS -------------------------------------------------
nodes = pd.read_csv(ECC / "network" / "grenoble_nodes.csv")
pts = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat), crs=4326).to_crs(2154)
poly = iris.to_crs(2154)[["zone_id", "geometry"]]
hit = gpd.sjoin(pts, poly, how="left", predicate="within")
hit = hit[~hit.index.duplicated()]
outside = hit.zone_id_right.isna()
near = gpd.sjoin_nearest(pts[outside.values], poly, how="left")
near = near[~near.index.duplicated()]
new_zone = hit.zone_id_right.copy()
new_zone[outside] = near.zone_id_right
changed = int((new_zone.values != nodes.zone_id.values).sum())
print(f"nodes: {len(nodes)}, outside the 70 IRIS (snapped to nearest): {int(outside.sum())}, "
      f"zone different from ECC_2025: {changed}")
nodes["zone_id"] = new_zone.astype(int).values

# demand weight a_u of a node inside its IRIS (sum = 1 per IRIS); nodes that had no weight
# keep 0, IRIS that lost all weighted nodes get uniform weights
w = nodes.zone_weight.to_numpy(float).copy()
for z, idx in nodes.groupby("zone_id").groups.items():
    idx = np.asarray(idx)
    s = w[idx].sum()
    w[idx] = w[idx] / s if s > 0 else 1.0 / len(idx)
nodes["zone_weight"] = w
nodes.to_csv(DATA / "network" / "grenoble_nodes.csv", index=False, float_format="%.10g")
print("data/ written")
