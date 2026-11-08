"""
Build the input files in ``data/``.  You do NOT need to run this: the outputs are in data/.

Sources (read-only)
-------------------
* ../CDC_2025/data/network/grenoble_{nodes,edges}.csv
      the SAME bicycle network as ECC 2025 / CDC 2025 (G_pygmo.pkl: 3452 nodes, 4993 arcs,
      DoS 1-6, slope), nodes already assigned to the 70 IRIS of Grenoble.
* ../CDC_2025/data/demand/*.csv, ../CDC_2025/data/regions/grenoble_iris.geojson
      EMD 2020 IRIS x IRIS trips and the IRIS polygons.
* ../../Grenoble_H_simplified.pkl
      OSM drive network of Grenoble (osmnx, simplified, 4720 nodes, 6175 roads) with highway
      type, lanes, maxspeed, width, oneway and geometry.  It replaces the 2024 car graph
      (Grenoble_car_capacity.pkl), which re-used the bicycle topology and gave a RANDOM highway
      type to every road that did not coincide with an OSM road.

Outputs
-------
data/network/bike_nodes.csv, bike_edges.csv   bicycle layer (+ column ``car_edge``: the car road
                                              the bike arc runs along, -1 if none)
data/network/car_nodes.csv                    car nodes (lon, lat, zone_id, is_trip_end)
data/network/car_edges.csv                    car roads (u, v, length, highway, lanes per
                                              direction, speed, capacity per direction, width)
data/demand/*.csv, data/regions/*.geojson     copies from CDC_2025

Bike arc -> car road matching
-----------------------------
The two layers come from different osmnx simplifications, so only 1217 bike arcs share both
end nodes with a car road.  We therefore match by geometry: a bike arc runs along car road r
if the points at 20 %, 50 % and 80 % of the arc are all within ``MATCH_M`` metres of r
(nearest road kept).  2931 bike arcs (148 km; 135 km of DoS 1-4) are matched.

Run:  python tools/build_data.py
"""
import ast
import pickle
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from itsc_cycling import config as C  # noqa: E402

CDC = ROOT.parent / "CDC_2025" / "data"
OSM_PICKLE = ROOT.parents[1] / "Grenoble_H_simplified.pkl"
DATA = ROOT / "data"
MATCH_M = 8.0                 # max distance (m) between a bike arc and the road it runs along

for sub in ("network", "demand", "regions"):
    (DATA / sub).mkdir(parents=True, exist_ok=True)

# ---- copies from CDC_2025 ---------------------------------------------------------------
for f in ("od_iris_grenoble.csv", "iris_zones.csv"):
    shutil.copy(CDC / "demand" / f, DATA / "demand")
shutil.copy(CDC / "regions" / "grenoble_iris.geojson", DATA / "regions")
bike_nodes = pd.read_csv(CDC / "network" / "grenoble_nodes.csv")
bike_edges = pd.read_csv(CDC / "network" / "grenoble_edges.csv")
iris = gpd.read_file(DATA / "regions" / "grenoble_iris.geojson").to_crs(2154)


# ---- helpers ------------------------------------------------------------------------------
def first(x):
    """OSM tags may be lists (merged ways): keep the first value."""
    if isinstance(x, list):
        return x[0] if x else None
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)[0]
        except (ValueError, SyntaxError):
            return x
    return x


def to_float(x, default=np.nan):
    x = first(x)
    try:
        return float(str(x).replace("m", "").replace(",", ".").strip())
    except (TypeError, ValueError):
        return default


# ---- car layer from the OSM drive network ---------------------------------------------------
H = pickle.load(open(OSM_PICKLE, "rb"))
car_xy = list(H.nodes)
nid = {n: i for i, n in enumerate(car_xy)}
rows, geoms = [], []
for u, v, d in H.edges(data=True):
    hw = first(d.get("highway")) or "unclassified"
    if hw not in C.ROAD_CLASSES:
        hw = "unclassified"
    spec = C.ROAD_CLASSES[hw]
    oneway = bool(first(d.get("oneway")) in (True, "True", "yes"))
    lanes_total = to_float(d.get("lanes"))
    if np.isnan(lanes_total):
        lanes_total = to_float(d.get("n_lanes"), 2.0)
    # every road is modelled in both directions (the simplified graph is undirected); a one-way
    # road keeps all its lanes per direction, a two-way road half of them (at least 1)
    lanes_dir = max(1, int(lanes_total if oneway else lanes_total // 2))
    lanes_dir = min(lanes_dir, 4)
    speed = to_float(d.get("maxspeed"), spec["speed"])
    if not 5 <= speed <= 130:
        speed = spec["speed"]
    width = to_float(d.get("width"))
    lane_width = width / max(lanes_total, 1) if not np.isnan(width) else C.DEFAULT_LANE_WIDTH_M
    rows.append(dict(u=nid[u], v=nid[v], length_m=float(d.get("distance", d.get("length"))),
                     highway=hw, oneway=oneway, lanes_dir=lanes_dir, speed_kmh=speed,
                     capacity_dir=spec["cap_per_lane"] * lanes_dir, lane_width_m=lane_width))
    geoms.append(d.get("geometry") or LineString([u, v]))
car_edges = pd.DataFrame(rows)
car_edges.insert(0, "edge_id", np.arange(len(car_edges)))

# car nodes -> IRIS (point in polygon); nodes outside the city get zone_id -1 (no trips)
cn = gpd.GeoDataFrame({"lon": [p[0] for p in car_xy], "lat": [p[1] for p in car_xy]},
                      geometry=gpd.points_from_xy([p[0] for p in car_xy], [p[1] for p in car_xy]),
                      crs=4326).to_crs(2154)
hit = gpd.sjoin(cn, iris[["zone_id", "geometry"]], how="left", predicate="within")
hit = hit[~hit.index.duplicated()]
cn["zone_id"] = hit.zone_id.fillna(-1).astype(int).values

# representative car trip ends: CAR_NODES_PER_IRIS nodes per IRIS, spread with k-means over the
# nodes that touch a non-motorway road (trips do not start on a motorway)
local = np.zeros(len(cn), bool)
for hw, uu, vv in zip(car_edges.highway, car_edges.u, car_edges.v):
    if not hw.startswith(("motorway", "trunk")):
        local[uu] = local[vv] = True
xy = np.c_[cn.geometry.x, cn.geometry.y]
trip_end = np.zeros(len(cn), bool)
for z in sorted(set(cn.zone_id) - {-1}):
    idx = np.flatnonzero((cn.zone_id.values == z) & local)
    k = min(C.CAR_NODES_PER_IRIS, len(idx))
    km = KMeans(k, n_init=10, random_state=0).fit(xy[idx])
    for c in km.cluster_centers_:
        trip_end[idx[np.argmin(((xy[idx] - c) ** 2).sum(1))]] = True
cn["is_trip_end"] = trip_end
cn.insert(0, "node_id", np.arange(len(cn)))
cn[["node_id", "lon", "lat", "zone_id", "is_trip_end"]].to_csv(DATA / "network" / "car_nodes.csv",
                                                                index=False, float_format="%.8g")
car_edges.to_csv(DATA / "network" / "car_edges.csv", index=False, float_format="%.6g")

# ---- bike arc -> car road (geometry) --------------------------------------------------------
cg = gpd.GeoSeries(geoms, crs=4326).to_crs(2154)
bx, by = bike_nodes.lon.to_numpy(), bike_nodes.lat.to_numpy()
bl = gpd.GeoSeries([LineString([(bx[u], by[u]), (bx[v], by[v])])
                    for u, v in zip(bike_edges.u, bike_edges.v)], crs=4326).to_crs(2154)
tree = cg.sindex
match = np.full(len(bl), -1)
for i, line in enumerate(bl):
    probes = [line.interpolate(f, normalized=True) for f in (0.2, 0.5, 0.8)]
    best, best_d = -1, np.inf
    for j in tree.query(line.buffer(MATCH_M + 2)):
        dmax = max(cg.iloc[j].distance(p) for p in probes)
        if dmax < best_d:
            best, best_d = j, dmax
    if best_d <= MATCH_M:
        match[i] = best
bike_edges["car_edge"] = match
bike_nodes.to_csv(DATA / "network" / "bike_nodes.csv", index=False, float_format="%.10g")
bike_edges.to_csv(DATA / "network" / "bike_edges.csv", index=False, float_format="%.10g")

m = match >= 0
print(f"car layer: {len(cn)} nodes ({int((cn.zone_id < 0).sum())} outside the 70 IRIS), "
      f"{len(car_edges)} roads, {car_edges.length_m.sum() / 1000:.0f} km, "
      f"{int(trip_end.sum())} trip ends")
print(f"bike arcs running along a car road: {int(m.sum())} / {len(match)} "
      f"({bike_edges.length_m[m].sum() / 1000:.0f} km; DoS 1-4: "
      f"{bike_edges.length_m[m & (bike_edges.dos < 5)].sum() / 1000:.0f} km)")
print("data/ written")
