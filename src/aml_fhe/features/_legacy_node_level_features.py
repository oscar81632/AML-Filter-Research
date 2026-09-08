import sys, time

print('Running converted notebook:', __file__, flush=True)


print('--- cell 0 start ---', flush=True)

_cell_t0 = time.time()

edges = data.groupby(["source", "target"]).agg(
    sf.sum("amount").alias("amount")
).toPandas()
weights = get_weights(edges)
edges_agg = edges.set_index(["source", "target"]).join(
    weights.set_index(["source", "target"]), how="left"
).reset_index()
edges_agg.loc[:, "amount_weighted"] = (
    edges_agg.loc[:, "amount"] * 
    (edges_agg.loc[:, "weight"] / edges_agg.loc[:, "weight"].max())
)

print('--- cell 0 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 1 start ---', flush=True)

_cell_t0 = time.time()

TOP_N = 50
NUM_HOPS = 5

print('--- cell 1 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 2 start ---', flush=True)

_cell_t0 = time.time()

data_input = spark.createDataFrame(edges_agg)
nodes_source = set(edges_agg["source"].unique())
nodes_target = set(edges_agg["target"].unique())
nodes_passthrough = nodes_source.intersection(nodes_target)

from aml_fhe.legacy_runtime import run_script_in_globals
run_script_in_globals(os.path.join(os.path.dirname(__file__), "_legacy_generate_flow_features.py"), globals())

comm_as_source_features.to_parquet(location_comm_as_source_features)
comm_as_target_features.to_parquet(location_comm_as_target_features)
comm_as_passthrough_features.to_parquet(location_comm_as_passthrough_features)
comm_as_passthrough_features_reverse.to_parquet(location_comm_as_passthrough_features_reverse)

del comm_as_source_features
del comm_as_target_features
del comm_as_passthrough_features
del comm_as_passthrough_features_reverse

print('--- cell 2 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 3 start ---', flush=True)

_cell_t0 = time.time()

# GPU Leiden implementations can be considered for larger-scale deployments.

print("Constructing Leiden communities")

graph = ig.Graph.DataFrame(edges_agg.loc[:, ["source", "target", "amount_weighted"]], use_vids=False, directed=True)
nodes_mapping = {x.index: x["name"] for x in graph.vs()}
communities_leiden = la.find_partition(
    graph, la.ModularityVertexPartition, n_iterations=100, weights="amount_weighted", seed=SEED,
)
communities_leiden = [[nodes_mapping[_] for _ in x] for x in communities_leiden]
communities_leiden = [(str(uuid.uuid4()), set(x)) for x in communities_leiden]
sizes_leiden = [len(x[1]) for x in communities_leiden]

with open(location_communities_leiden, "wb") as fl:
    pickle.dump(communities_leiden, fl)

print('--- cell 3 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 4 start ---', flush=True)

_cell_t0 = time.time()

with open(location_communities_leiden, "rb") as fl:
    communities_leiden = pickle.load(fl)

print('--- cell 4 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 5 start ---', flush=True)

_cell_t0 = time.time()

data_agg_weights = get_weights(
    data.groupby(["source", "target"])
    .agg(
        sf.sum("amount").alias("amount")
    ).toPandas()
)

data_agg_weights_rev = data_agg_weights.rename(
    columns={"target": "source", "source": "target"}
).loc[:, ["source", "target", "weight"]]
data_agg_weights_ud = pd.concat([data_agg_weights, data_agg_weights_rev], ignore_index=True)
data_agg_weights_ud = data_agg_weights_ud.groupby(["source", "target"]).agg(weight=("weight", "sum")).reset_index()

data_agg_weights_ud.sort_values("weight", ascending=False, inplace=True)
grouped_ud = data_agg_weights_ud.groupby("source").head(KEEP_TOP_N).reset_index(drop=True)
grouped_ud = grouped_ud.groupby("source").agg(targets=("target", set))

total = grouped_ud.index.nunique()
nodes_neighborhoods = {}
for index, (source, targets) in enumerate(grouped_ud.iterrows()):
    community_candidates = {source}
    for target in targets["targets"]:
        community_candidates |= (grouped_ud.loc[target, "targets"] | {target})
    nodes_neighborhoods[source] = set(community_candidates)
    if not (index % 250_000):
        print(index, total)

del data_agg_weights_rev
del data_agg_weights_ud
del grouped_ud

print('--- cell 5 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 6 start ---', flush=True)

_cell_t0 = time.time()

print("Constructing 2-hop communities")

communities_2_hop = get_communities_spark(
    nodes_neighborhoods,
    ig.Graph.DataFrame(edges_agg.loc[:, ["source", "target", "amount_weighted"]], use_vids=False, directed=True), 
    os.cpu_count(), spark, 2, "all", 0.01, "amount_weighted"
)
sizes_2_hop = [len(x[1]) for x in communities_2_hop]

print('--- cell 6 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 7 start ---', flush=True)

_cell_t0 = time.time()

ts_min = data.select(sf.min("timestamp").alias("x")).collect()[0]["x"] - timedelta(minutes=1)
data_graph_agg = data.groupby(["source", "target", "source_bank", "target_bank", "source_currency"]).agg(
    sf.count("source").alias("num_transactions"),
    sf.sum("amount").alias("amount"),
    sf.sum("source_amount").alias("source_amount"),
    sf.collect_list(sf.array((sf.col("timestamp") - ts_min).cast("long"), sf.col("amount"))).alias("timestamps_amounts"),
)
data_graph_agg_sdf = data_graph_agg.persist(StorageLevel.DISK_ONLY)
print(data_graph_agg_sdf.count())
data_graph_agg = data_graph_agg_sdf.toPandas()
index = ["source", "target"]
edges_agg.loc[:, index + ["weight"]].set_index(index)
data_graph_agg = data_graph_agg.set_index(index).join(
    edges_agg.loc[:, index + ["weight"]].set_index(index), how="left"
).reset_index()
data_graph_agg.loc[:, "amount_weighted"] = (
    data_graph_agg.loc[:, "amount"] * 
    (data_graph_agg.loc[:, "weight"] / data_graph_agg.loc[:, "weight"].max())
)

print('--- cell 7 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 8 start ---', flush=True)

_cell_t0 = time.time()

print("Leiden communitites features creation")

features_leiden = generate_features_spark(communities_leiden, data_graph_agg, spark)
features_leiden = features_leiden.rename(columns={"key": "key_fake"})
communities_leiden_dict = dict(communities_leiden)
features_leiden.loc[:, "key"] = features_leiden.loc[:, "key_fake"].apply(lambda x: communities_leiden_dict[x])
features_leiden = features_leiden.explode("key")
del features_leiden["key_fake"]
features_leiden.set_index("key").to_parquet(location_features_leiden)

print('--- cell 8 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 9 start ---', flush=True)

_cell_t0 = time.time()

print("2-hop communitites features creation")

features_2_hop = generate_features_spark(communities_2_hop, data_graph_agg, spark)
features_2_hop.set_index("key").to_parquet(location_features_2_hop)

print('--- cell 9 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 10 start ---', flush=True)

_cell_t0 = time.time()

print("Temporal flows features creation")

edges_totals = data.select("source", "target", "amount").groupby(
    ["source", "target"]
).agg(sf.count("amount").alias("amount")).toPandas()
edges_totals = edges_totals.sort_values("amount", ascending=False).reset_index(drop=True)
left_edges = spark.createDataFrame(edges_totals.groupby("target").head(TOP_N).loc[:, ["source", "target"]])
right_edges = spark.createDataFrame(edges_totals.groupby("source").head(TOP_N).loc[:, ["source", "target"]])

columns = ["source", "target", "timestamp", "amount"]

left = left_edges.select(sf.col("source").alias("src"), sf.col("target").alias("tgt")).join(
    data.select(*columns),
    on=(sf.col("src") == sf.col("source")) & (sf.col("tgt") == sf.col("target")),
    how="left"
).drop("src", "tgt").persist(StorageLevel.DISK_ONLY)
select = []
for column in left.columns:
    select.append(sf.col(column).alias(f"left_{column}"))
left = left.select(*select)
right = right_edges.select(sf.col("source").alias("src"), sf.col("target").alias("tgt")).join(
    data.select(*columns),
    on=(sf.col("src") == sf.col("source")) & (sf.col("tgt") == sf.col("target")),
    how="left"
).drop("src", "tgt").persist(StorageLevel.DISK_ONLY)

print(left.count(), right.count())

flows_temporal = left.join(
    right,
    (left["left_target"] == right["source"]) &
    (left["left_timestamp"] <= right["timestamp"]),
    how="inner"
).groupby(["left_source", "left_target", "source", "target"]).agg(
    sf.sum("left_amount").alias("left_amount"),
    sf.sum("amount").alias("amount"),
).drop("left_target").select(
    sf.col("left_source").alias("dispense"),
    sf.col("source").alias("passthrough"),
    sf.col("target").alias("sink"),
    sf.least("left_amount", "amount").alias("amount"),
)

aggregate = [
    sf.sum("amount").alias("amount_sum"),
    sf.mean("amount").alias("amount_mean"),
    sf.median("amount").alias("amount_median"),
    sf.max("amount").alias("amount_max"),
    sf.stddev("amount").alias("amount_std"),
    sf.countDistinct("dispense").alias("dispense_count"),
    sf.countDistinct("passthrough").alias("passthrough_count"),
    sf.countDistinct("sink").alias("sink_count"),
]
for flow_location, flow_type in [
    (location_flow_dispense, "dispense"), (location_flow_passthrough, "passthrough"), (location_flow_sink, "sink")
]:
    print(flow_type)
    flows_temporal_stats = flows_temporal.groupby(flow_type).agg(*aggregate).toPandas()
    flows_temporal_cyclic_stats = flows_temporal.where(
        (sf.col("dispense") == sf.col("sink"))
    ).groupby(flow_type).agg(*aggregate).toPandas()
    flows_temporal_stats = flows_temporal_stats.set_index(flow_type).join(
        flows_temporal_cyclic_stats.set_index(flow_type),
        how="left", rsuffix="_cycle"
    )
    flows_temporal_stats.index.name = "key"
    flows_temporal_stats.to_parquet(flow_location)
    del flows_temporal_stats
    del flows_temporal_cyclic_stats

left.unpersist()
right.unpersist()

del edges_totals
del left_edges
del right_edges

print('--- cell 10 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 11 start ---', flush=True)

_cell_t0 = time.time()

print("1-hop-source features creation")

features_source = create_spark_graph_dataframe(data_graph_agg, spark).withColumn(
    "key", sf.col("source")
).repartition(os.cpu_count() * 5, "key").groupby("key").applyInPandas(
    generate_features_udf_wrapper(False), schema=SCHEMA_FEAT_UDF
).toPandas()
features_source = pd.DataFrame(features_source["features"].apply(json.loads).tolist())
features_source.columns = [f"{s.G_1HOP_PREFIX}{x}" if x != "key" else x for x in features_source.columns]
features_source.set_index("key").to_parquet(location_features_source)

print('--- cell 11 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 12 start ---', flush=True)

_cell_t0 = time.time()

print("1-hop-target features creation")

features_target = create_spark_graph_dataframe(data_graph_agg, spark).withColumn(
    "key", sf.col("target")
).repartition(os.cpu_count() * 5, "key").groupby("key").applyInPandas(
    generate_features_udf_wrapper(False), schema=SCHEMA_FEAT_UDF
).toPandas()
features_target = pd.DataFrame(features_target["features"].apply(json.loads).tolist())
features_target.columns = [f"{s.G_1HOP_PREFIX}{x}" if x != "key" else x for x in features_target.columns]
features_target.set_index("key").to_parquet(location_features_target)

print('--- cell 12 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 13 start ---', flush=True)

_cell_t0 = time.time()

del data_graph_agg

print('--- cell 13 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 14 start ---', flush=True)

_cell_t0 = time.time()

ENABLED_FEATURES = [
    ("leiden", location_features_leiden),
    ("2_hop", location_features_2_hop),
    ("as_source", location_features_source),
    ("as_target", location_features_target),
    ("comm_as_source_features", location_comm_as_source_features),
    ("comm_as_target_features", location_comm_as_target_features),
    ("comm_as_passthrough_features", location_comm_as_passthrough_features),
    ("comm_as_passthrough_features_reverse", location_comm_as_passthrough_features_reverse),
    ("flow_dispense", location_flow_dispense),
    ("flow_passthrough", location_flow_passthrough),
    ("flow_sink", location_flow_sink),   
]

print('--- cell 14 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 15 start ---', flush=True)

_cell_t0 = time.time()

def get_or_fit_object(group_name, object_to_fit, data_to_fit):
    object_name = str(object_to_fit.__class__.__name__)
    location_object = os.path.join(location_main, f"fitted_{object_name}_{group_name}.pickle")
    if os.path.exists(location_object):
        with open(location_object, "rb") as fl:
            return pickle.load(fl)
    object_to_fit.fit(data_to_fit)
    with open(location_object, "wb") as fl:
        pickle.dump(object_to_fit, fl)
    return object_to_fit

print('--- cell 15 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 16 start ---', flush=True)

_cell_t0 = time.time()

all_features = pd.DataFrame()
all_features.index.name = "key"

if DIM_REDUCTION_PERC < 1:
    print("Applying dimensionality reduction!")
    sources = set(data.select("source").distinct().toPandas()["source"])
    targets = set(data.select("target").distinct().toPandas()["target"])
    for feature_group, location in ENABLED_FEATURES:
        group_features = pd.read_parquet(location)

        constants = []
        for column in group_features.columns:
            if group_features[column].nunique(dropna=True) <= 1:
                del group_features[column]
                constants.append(column)
        print(f"Deleted {len(constants)} constant columns for {feature_group}")
        
        group_features = group_features.loc[:, sorted(group_features.columns)]
        n_clusters = int(np.ceil(len(group_features.columns) * DIM_REDUCTION_PERC))
        print(feature_group, n_clusters)
        group_features = group_features.fillna(0)
        group_scaled = get_or_fit_object(
            feature_group, StandardScaler(), group_features
        ).transform(group_features)
        group_features_red = get_or_fit_object(
            feature_group, FeatureAgglomeration(n_clusters=n_clusters), group_scaled
        ).transform(group_scaled).astype(np.float32)
        if SCALE_TO_FLOAT_16:
            group_features_red = get_or_fit_object(
                feature_group, MinMaxScaler((0, 2048)), group_features_red
            ).transform(group_features_red).round(decimals=4).astype(np.float32)
        group_features = pd.DataFrame(group_features_red, index=group_features.index)
        group_features.columns = [f"fa_{feature_group}_{x + 1}" for x in group_features.columns]
        all_features = all_features.join(group_features.copy(deep=True), how="outer").fillna(0)
    all_features.loc[:, "has_credits"] = all_features.index.map(lambda x: bool({x}.intersection(sources)))
    all_features.loc[:, "has_debits"] = all_features.index.map(lambda x: bool({x}.intersection(targets)))
else:
    for feature_group, location in ENABLED_FEATURES:
        all_features = all_features.join(
            pd.read_parquet(location), how="outer", rsuffix=f"_{feature_group}"
        )

print('--- cell 16 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 17 start ---', flush=True)

_cell_t0 = time.time()

constants = []
for column in all_features.columns:
    if all_features[column].nunique(dropna=True) <= 1:
        del all_features[column]
        constants.append(column)
print(f"Deleted {len(constants)} constant columns")

print('--- cell 17 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 18 start ---', flush=True)

_cell_t0 = time.time()

all_features = all_features.loc[:, sorted(all_features.columns)]
print("Features:", all_features.shape)

print('--- cell 18 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 19 start ---', flush=True)

_cell_t0 = time.time()

if SKIP_ANOMALY_DETECTION:
    print("Skipping anomaly detection")
else:
    print("Training the anomaly detection model")
    anomalies = all_features.loc[:, []]
    model_ad = get_or_fit_object(
        "ad", IsolationForest(n_estimators=1_000, random_state=SEED), all_features.fillna(0)
    )
    anomalies.loc[:, "anomaly_score"] = model_ad.decision_function(all_features.fillna(0))
    all_features = all_features.join(anomalies, how="outer", rsuffix=f"_{feature_group}")

print('--- cell 19 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 20 start ---', flush=True)

_cell_t0 = time.time()

all_features.to_parquet(location_features_node_level)
del all_features

print('--- cell 20 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)
