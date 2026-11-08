"""
Build the clean, portable input files in ``data/`` from the original (legacy) pickles.

You do NOT need to run this script to reproduce the paper: its outputs are already in
``data/``.  It is kept to document exactly where every number comes from.

Sources (legacy files, 2024)
----------------------------
* ``data/raw/G_pygmo.pkl``                      bike network of Grenoble with the DoS of every
                                                arc (slopes checked against node elevations)
* ``<LEGACY>/od_matrix.pkl``                    210 x 210 IRIS-to-IRIS daily trip matrix of the
                                                Grenoble metropole (EMD 2020, all modes, symmetric)
* ``<LEGACY>/region_nodes_dict.pkl``            the 70 IRIS of the city of Grenoble -> network nodes
                                                + IRIS polygon
* ``<LEGACY>/origins.pkl``                      IRIS codes and names (rows of ``od_matrix``)
* ``_archive_old_files/node_demand_dict.pkl``   node-to-node bike demand used in the paper (345 MB)
* ``data/raw/grenoble_historical_od_2017_2022.pkl`` + ``<OUTPUT_MANUEL>/{origins,destinations}.pkl``
                                                the newer IRIS -> facility OD (optional demand)

Outputs (all in ``data/``)
--------------------------
network/grenoble_nodes.csv      node_id, lon, lat, zone_id, zone_weight
network/grenoble_edges.csv      edge_id, u, v, length_m, slope_pct (from u to v), dos
network/grenoble_iris.geojson   the 70 IRIS polygons (for maps)
demand/iris_zones.csv           zone_id, iris_code, name
demand/od_iris_grenoble.csv     zone_o, zone_d, trips_matrix (raw symmetric matrix entry)
demand/historical_unit_od.csv   optional: newer OD (unit demand) snapped to the network

Run:  python tools/build_data.py        (needs the legacy folder, ~2 min, ~6 GB RAM)
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Folder where the 2024 notebooks and their pickles live (only needed to rebuild data/).
LEGACY = ROOT.parents[2]  # .../Second_project_2024
OUTPUT_MANUEL = ROOT.parents[1] / "output_manuel" / "output_manuel" / "grenoble"

# DoS is stored in the legacy graphs as a "key" = the safety/comfort score of each class.
KEY_TO_DOS = {1.0: 1, 1.67: 2, 2.22: 3, 2.83: 4, 4.64: 5, 5.23: 6}
FIRST_IRIS_ROW = 48  # rows/columns 48..117 of od_matrix.pkl are the 70 IRIS of Grenoble


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_network():
    """Nodes/edges CSV from G_pygmo.pkl; zones and demand weights from the legacy demand."""
    G = load(DATA / "raw" / "G_pygmo.pkl")
    nodes = list(G.nodes())
    node_id = {n: i for i, n in enumerate(nodes)}

    edges = pd.DataFrame(
        [
            dict(edge_id=i, u=node_id[u], v=node_id[v], length_m=d["distance"],
                 slope_pct=d["slope"], dos=KEY_TO_DOS[d["key"]])
            for i, (u, v, d) in enumerate(G.edges(data=True))
        ]
    )

    # --- zones: which IRIS each node belongs to -------------------------------------------
    regions = load(LEGACY / "region_nodes_dict.pkl")  # {iris_row: [nodes, polygon]}
    node_zone = {n: r for r, (ns, _poly) in regions.items() for n in ns}

    # --- zone weights a_v ----------------------------------------------------------------
    # The paper's node demand is separable:  q_uv = c * OD[r, s] * a_u * a_v,
    # with u in zone r, v in zone s, sum_{u in r} a_u = 1 and c = 0.0025 (see README).
    # We recover a_u from the legacy dictionary, then check the reconstruction.
    od = load(LEGACY / "od_matrix.pkl")
    nd = load(ROOT / "_archive_old_files" / "node_demand_dict.pkl")
    row_sum = defaultdict(float)
    for (u, v), q in nd.items():
        r, s = node_zone[u], node_zone[v]
        if od[r, s] > 0:
            row_sum[(u, s)] += q / od[r, s]  # = c * a_u * sum_{v in s} a_v = c * a_u
    c = 0.0025
    weight = {}
    for (u, s), val in row_sum.items():
        weight.setdefault(u, val / c)
    # check: every entry of the legacy dictionary is reproduced
    err = max(
        abs(q - c * od[node_zone[u], node_zone[v]] * weight[u] * weight[v]) / max(q, 1e-30)
        for (u, v), q in list(nd.items())[:: max(1, len(nd) // 200000)]
    )
    print(f"node demand reconstructed from 70x70 OD + node weights, max rel. error = {err:.2e}")
    del nd

    lon, lat = zip(*nodes)
    nodes_df = pd.DataFrame(
        dict(node_id=range(len(nodes)), lon=lon, lat=lat,
             zone_id=[node_zone[n] for n in nodes],
             zone_weight=[weight.get(n, 0.0) for n in nodes])
    )
    (DATA / "network").mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(DATA / "network" / "grenoble_nodes.csv", index=False, float_format="%.10g")
    edges.to_csv(DATA / "network" / "grenoble_edges.csv", index=False, float_format="%.10g")

    # IRIS polygons -> GeoJSON (for the map figures)
    feats = [
        dict(type="Feature", properties=dict(zone_id=int(r)), geometry=poly.__geo_interface__)
        for r, (_ns, poly) in regions.items()
    ]
    (DATA / "network" / "grenoble_iris.geojson").write_text(
        json.dumps(dict(type="FeatureCollection", features=feats))
    )
    return G, node_id, od, regions


def build_demand(G, node_id, od, regions):
    """Zone OD (70 x 70) and the optional newer OD."""
    dem = DATA / "demand"
    dem.mkdir(parents=True, exist_ok=True)

    origins = load(LEGACY / "origins.pkl")
    zones = sorted(regions)
    pd.DataFrame(
        dict(zone_id=zones,
             iris_code=[origins.iloc[z]["iris_code"] for z in zones],
             name=[str(origins.iloc[z]["name"]) for z in zones])
    ).to_csv(dem / "iris_zones.csv", index=False)

    rows = [dict(zone_o=r, zone_d=s, trips_matrix=od[r, s]) for r in zones for s in zones]
    pd.DataFrame(rows).to_csv(dem / "od_iris_grenoble.csv", index=False, float_format="%.10g")

    # ---- optional: the newer OD (grenoble_historical_od_2017_2022.pkl) ------------------
    # It stores a BOOLEAN IRIS -> facility matrix (1 = the pair exists), so every pair has
    # demand 1.  We snap the 68 Grenoble IRIS and 248 facilities to the nearest node of
    # G_pygmo (the OSM node ids inside the pickle belong to other graphs).
    hist = load(DATA / "raw" / "grenoble_historical_od_2017_2022.pkl")
    o_pos, d_pos, mask = (hist["common_origin_positions"], hist["common_destination_positions"],
                          hist["od_common"])
    o_gdf = load(OUTPUT_MANUEL / "origins.pkl")
    d_gdf = load(OUTPUT_MANUEL / "destinations.pkl")
    nodes = pd.read_csv(DATA / "network" / "grenoble_nodes.csv")
    xy = nodes[["lon", "lat"]].to_numpy()
    coslat = np.cos(np.radians(xy[:, 1].mean()))

    def nearest(points):
        p = np.array([[pt.x, pt.y] for pt in points])
        d2 = ((p[:, None, 0] - xy[None, :, 0]) * coslat) ** 2 + (p[:, None, 1] - xy[None, :, 1]) ** 2
        return d2.argmin(1), np.sqrt(d2.min(1)) * 111_320

    o_node, o_dist = nearest(o_gdf.iloc[o_pos]["coord"])
    d_node, d_dist = nearest(d_gdf.iloc[d_pos]["coord"])
    print(f"historical OD snapping distance: origins max {o_dist.max():.0f} m, "
          f"destinations max {d_dist.max():.0f} m")
    agg = defaultdict(float)
    for i, j in zip(*np.nonzero(mask)):
        if o_node[i] != d_node[j]:
            agg[(o_node[i], d_node[j])] += 1.0
    pd.DataFrame([dict(origin=a, destination=b, demand=q) for (a, b), q in agg.items()]).to_csv(
        dem / "historical_unit_od.csv", index=False
    )




if __name__ == "__main__":
    G, node_id, od, regions = build_network()
    build_demand(G, node_id, od, regions)
    print("data/ written")
