import sys, time

print('Running converted notebook:', __file__, flush=True)


print('--- cell 0 start ---', flush=True)

_cell_t0 = time.time()

import json
import os
import pickle
import random
import time
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import timedelta
from glob import glob
from itertools import product
from pyspark.sql import functions as sf
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, RocCurveDisplay
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import xgboost as xgb

import settings as s

os.environ["EXSTRAQT_DATA_TYPE_FOLDER"] = s.OUTPUT_POSTFIX.lstrip("-")

from common import get_weights, delete_large_vars, MULTI_PROC_STAGING_LOCATION
from communities import get_communities_spark
from features import (
    generate_features_spark, generate_features_udf_wrapper, get_edge_features_udf,
    create_spark_graph_dataframe, SCHEMA_FEAT_UDF, CURRENCY_RATES
)

# NOTE: notebook magic preserved but disabled: %load_ext autoreload
# NOTE: notebook magic preserved but disabled: %autoreload 2

print('--- cell 0 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 1 start ---', flush=True)

_cell_t0 = time.time()

SEED = int(os.environ.get("EXSTRAQT_SEED", 42))
print(f"{SEED=}")
random.seed(SEED)
np.random.seed(SEED)

print('--- cell 1 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 2 start ---', flush=True)

_cell_t0 = time.time()

EXSTRAQT_NUM_PROCS = int(os.environ.get("EXSTRAQT_NUM_PROCS", os.cpu_count()))

DIM_REDUCTION_PERC = float(os.environ.get("EXSTRAQT_DIM_REDUCTION_PERC", 1))
SCALE_TO_FLOAT_16 = bool(int(os.environ.get("EXSTRAQT_SCALE_TO_FLOAT_16", 0)))
SKIP_ANOMALY_DETECTION = bool(int(os.environ.get("EXSTRAQT_SKIP_ANOMALY_DETECTION", 0)))

print('--- cell 2 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 3 start ---', flush=True)

_cell_t0 = time.time()

print('Skipping upstream Python version guard; running on', sys.version, flush=True)

print('--- cell 3 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 4 start ---', flush=True)

_cell_t0 = time.time()

SPARK_CONF = [
    ("spark.driver.memory", "32g"),
    ("spark.worker.memory", "32g"),
    ("spark.driver.maxResultSize", "32g"),
    ("spark.sql.execution.arrow.pyspark.enabled", "true"),
    ("spark.sql.execution.arrow.pyspark.fallback.enabled", "true"),
    ("spark.network.timeout", "600s"),
    ("spark.sql.autoBroadcastJoinThreshold", -1),
    ("spark.local.dir", f".{os.sep}temp-spark"),
]

if "EXSTRAQT_SEED" in os.environ:
    SPARK_CONF.append(("spark.log.level", "ERROR"))

shutil.rmtree("artifacts", ignore_errors=True)
shutil.rmtree("temp-spark", ignore_errors=True)
spark = (
    SparkSession.builder.master(f"local[{EXSTRAQT_NUM_PROCS}]").appName("testing")
    .config(conf=SparkConf().setAll(SPARK_CONF))
    .getOrCreate()
)

print('--- cell 4 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 5 start ---', flush=True)

_cell_t0 = time.time()

TRAIN_PERC = float(os.environ.get("EXSTRAQT_TRAIN_PERC", 0.60))
VALIDATION_PERC = float(os.environ.get("EXSTRAQT_VALIDATION_PERC", 0.20))
TEST_PERC = float(os.environ.get("EXSTRAQT_TEST_PERC", 0.20))

KEEP_TOP_N = 100

assert(sum([TRAIN_PERC, VALIDATION_PERC, TEST_PERC]) == 1)

location_main = os.path.join("features", os.environ["EXSTRAQT_DATA_TYPE_FOLDER"])
shutil.rmtree(location_main, ignore_errors=True)

location_communities_leiden = f"{location_main}{os.sep}communities_leiden.parquet"

location_features_leiden = f"{location_main}{os.sep}features_leiden.parquet"
location_features_ego = f"{location_main}{os.sep}features_ego.parquet"
location_features_2_hop = f"{location_main}{os.sep}features_2_hop.parquet"
location_features_2_hop_out = f"{location_main}{os.sep}features_2_hop_out.parquet"
location_features_2_hop_in = f"{location_main}{os.sep}features_2_hop_in.parquet"
location_features_2_hop_combined = f"{location_main}{os.sep}features_2_hop_combined.parquet"
location_features_source = f"{location_main}{os.sep}features_source.parquet"
location_features_target = f"{location_main}{os.sep}features_target.parquet"

location_flow_dispense = f"{location_main}{os.sep}flow_dispense.parquet"
location_flow_passthrough = f"{location_main}{os.sep}flow_passthrough.parquet"
location_flow_sink = f"{location_main}{os.sep}flow_sink.parquet"

location_comm_as_source_features = f"{location_main}{os.sep}comm_as_source_features.parquet"
location_comm_as_target_features = f"{location_main}{os.sep}comm_as_target_features.parquet"
location_comm_as_passthrough_features = f"{location_main}{os.sep}comm_as_passthrough_features.parquet"
location_comm_as_passthrough_features_reverse = f"{location_main}{os.sep}comm_as_passthrough_features_reverse.parquet"

location_features_node_level = f"{location_main}{os.sep}features_node_level.parquet"
location_features_edges = f"{location_main}{os.sep}features_edges.parquet"

location_features_edges_train = f"{location_main}{os.sep}features_edges_train.parquet"
location_features_edges_valid = f"{location_main}{os.sep}features_edges_valid.parquet"
location_features_edges_test = f"{location_main}{os.sep}features_edges_test.parquet"

location_train_trx_features = f"{location_main}{os.sep}train_trx_features.parquet"
location_valid_trx_features = f"{location_main}{os.sep}valid_trx_features.parquet"
location_test_trx_features = f"{location_main}{os.sep}test_trx_features.parquet"

location_train_features = f"{location_main}{os.sep}train_features.parquet"
location_valid_features = f"{location_main}{os.sep}valid_features.parquet"
location_test_features = f"{location_main}{os.sep}test_features.parquet"

location_train_features_dm = f"{location_main}{os.sep}train_dm.bin"
location_valid_features_dm = f"{location_main}{os.sep}valid_dm.bin"
location_test_features_dm = f"{location_main}{os.sep}test_dm.bin"

try:
    os.makedirs(location_main)
except FileExistsError:
    pass

print('--- cell 5 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 6 start ---', flush=True)

_cell_t0 = time.time()

data = spark.read.parquet(s.STAGED_DATA_LOCATION)
data = data.withColumn("is_laundering", sf.col("is_laundering").cast("boolean"))
data_count_original = data.count()

# Also not used in the benchmarks
data = data.drop("source_entity", "target_entity")

print('--- cell 6 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 7 start ---', flush=True)

_cell_t0 = time.time()

trx_ids_sorted = data.sort("timestamp").select("transaction_id").toPandas()["transaction_id"].values
trx_count = len(trx_ids_sorted)

last_train_index = int(np.floor(trx_count * TRAIN_PERC))
last_validation_index = last_train_index + int(np.floor(trx_count * VALIDATION_PERC))
train_indexes = trx_ids_sorted[:last_train_index]
validation_indexes = trx_ids_sorted[last_train_index:last_validation_index]
test_indexes = trx_ids_sorted[last_validation_index:]

train_indexes_loc = os.path.join(location_main, "temp_train_indexes.parquet")
validation_indexes_loc = os.path.join(location_main, "temp_validation_indexes.parquet")
test_indexes_loc = os.path.join(location_main, "temp_test_indexes.parquet")

pd.DataFrame(train_indexes, columns=["transaction_id"]).to_parquet(train_indexes_loc)
pd.DataFrame(validation_indexes, columns=["transaction_id"]).to_parquet(validation_indexes_loc)
pd.DataFrame(test_indexes, columns=["transaction_id"]).to_parquet(test_indexes_loc)

train_indexes = spark.read.parquet(train_indexes_loc)
validation_indexes = spark.read.parquet(validation_indexes_loc)
test_indexes = spark.read.parquet(test_indexes_loc)

train = train_indexes.join(
    data, on="transaction_id", how="left"
).persist(StorageLevel.DISK_ONLY)
validation = validation_indexes.join(
    data, on="transaction_id", how="left"
).persist(StorageLevel.DISK_ONLY)
test = test_indexes.join(
    data, on="transaction_id", how="left"
).persist(StorageLevel.DISK_ONLY)
train_count, validation_count, test_count = train.count(), validation.count(), test.count()
print()
print(trx_count, train_count, validation_count, test_count)
print()

os.remove(train_indexes_loc)
os.remove(validation_indexes_loc)
os.remove(test_indexes_loc)

print('--- cell 7 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 8 start ---', flush=True)

_cell_t0 = time.time()

def generate_edge_features(input_data):
    print(f"Generating edge features")
    to_select = ["source", "target", "format", "source_currency", "source_amount", "amount", "timestamp"]
    edges_features_input = input_data.select(*to_select).groupby(
        ["source", "target", "format", "source_currency"]
    ).agg(
        sf.sum("source_amount").alias("source_amount"), 
        sf.sum("amount").alias("amount"),
        sf.unix_timestamp(sf.min("timestamp")).alias("min_ts"),
        sf.unix_timestamp(sf.max("timestamp")).alias("max_ts"),
    ).repartition(os.cpu_count() * 2, "source", "target").persist(StorageLevel.DISK_ONLY)
    _ = edges_features_input.count()
    edge_features = edges_features_input.groupby(["source", "target"]).applyInPandas(
        get_edge_features_udf, schema=SCHEMA_FEAT_UDF
    ).toPandas()
    edge_features = pd.DataFrame(edge_features["features"].apply(json.loads).tolist())
    edge_features.to_parquet(location_features_edges)
    del edge_features

print('--- cell 8 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 9 start ---', flush=True)

_cell_t0 = time.time()

def add_node_features_to_edges(features_in, location):
    features_in = features_in.set_index("target").join(
        pd.read_parquet(location_features_node_level), how="left", rsuffix="_target"
    ).reset_index().set_index("source").join(
        pd.read_parquet(location_features_node_level), how="left", rsuffix="_source"
    ).reset_index()

    if "anomaly_score" in features_in.columns:
        features_in.loc[:, "anomaly_scores_diff"] = features_in.loc[:, "anomaly_score"] - features_in.loc[:, "anomaly_score_source"]
        features_in.loc[:, "anomaly_scores_min"] = np.array(
            [
                features_in.loc[:, "anomaly_score"].values, 
                features_in.loc[:, "anomaly_score_source"].values
            ],
        ).min(axis=0)
        features_in.loc[:, "anomaly_scores_max"] = np.array(
            [
                features_in.loc[:, "anomaly_score"].values, 
                features_in.loc[:, "anomaly_score_source"].values
            ],
        ).max(axis=0)
        features_in.loc[:, "anomaly_scores_mean"] = np.array(
            [
                features_in.loc[:, "anomaly_score"].values, 
                features_in.loc[:, "anomaly_score_source"].values
            ],
        ).mean(axis=0)
    else:
        print("`anomaly_score` calculations missing in the nodes features!")

    features_in.to_parquet(location)

print('--- cell 9 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 10 start ---', flush=True)

_cell_t0 = time.time()

def save_trx_features(data_in, location):
    columns = [
        "source", "target", "source_currency", "target_currency", "format", "amount", 
        "source_dispensation",
        "target_accumulation",
        "source_positive_balance",
        "source_negative_balance",
        "target_positive_balance",
        "target_negative_balance",
        "source_active_for",
        "target_active_for",
        "is_laundering"
    ]
    missing_columns = set(columns) - set(data_in.columns)
    if missing_columns:
        print(f"Skipping the missing transaction columns: {sorted(missing_columns)}")
    columns_common = list(set(columns).intersection(data_in.columns))
    trx_features = data_in.select(*columns_common).toPandas()
    if "source_positive_balance" in columns_common:
        trx_features.loc[:, "source_balance_ratio"] = (
            trx_features["source_positive_balance"] / trx_features["source_negative_balance"]
        ).fillna(0).replace(np.inf, 0)
        trx_features.loc[:, "target_balance_ratio"] = (
            trx_features["target_positive_balance"] / trx_features["target_negative_balance"]
        ).fillna(0).replace(np.inf, 0)
    trx_features.loc[:, "inter_currency"] = trx_features["source_currency"] != trx_features["target_currency"]
    trx_features.to_parquet(location)
    del trx_features

print('--- cell 10 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 11 start ---', flush=True)

_cell_t0 = time.time()

def free_up_memory(to_keep_in, locals_in):
    to_reset = list(globals().keys())
    if "to_keep" in to_reset:
        to_reset.remove("to_keep")
    to_reset = set(to_reset) - set(to_keep_in)
    for var_to_reset in list(to_reset):
        var_to_reset = f"^{var_to_reset}$"
# NOTE: notebook magic preserved but disabled:         %reset_selective -f {var_to_reset}
    
    delete_large_vars(globals(), locals_in)

print('--- cell 11 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 12 start ---', flush=True)

_cell_t0 = time.time()

# Later on, we will reset the variables (to free up memory), while still keeping these intact
to_keep = list(globals().keys())

print('--- cell 12 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 14 start ---', flush=True)

_cell_t0 = time.time()

data = train.select("*")
print(f"Constructing node-level features for `train` data: {data.count():,}")

from aml_fhe.legacy_runtime import run_script_in_globals
run_script_in_globals(os.path.join(os.path.dirname(__file__), "_legacy_node_level_features.py"), globals())

generate_edge_features(data)
train_edges = train.select("source", "target").drop_duplicates().toPandas().set_index(
    ["source", "target"]
)
train_features = train_edges.join(
    pd.read_parquet(location_features_edges).set_index(["source", "target"]), how="left"
).reset_index()
add_node_features_to_edges(train_features, location_features_edges_train)
save_trx_features(train, location_train_trx_features)

print('--- cell 14 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 15 start ---', flush=True)

_cell_t0 = time.time()

free_up_memory(to_keep, locals())

print('--- cell 15 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 16 start ---', flush=True)

_cell_t0 = time.time()

data = train.union(validation).select("*")
print(f"Constructing node-level features for `validation` data: {data.count():,}")

run_script_in_globals(os.path.join(os.path.dirname(__file__), "_legacy_node_level_features.py"), globals())

generate_edge_features(data)
validation_edges = validation.select("source", "target").drop_duplicates().toPandas().set_index(
    ["source", "target"]
)
validation_features = validation_edges.join(
    pd.read_parquet(location_features_edges).set_index(["source", "target"]), how="left"
).reset_index()
add_node_features_to_edges(validation_features, location_features_edges_valid)
save_trx_features(validation, location_valid_trx_features)

print('--- cell 16 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 17 start ---', flush=True)

_cell_t0 = time.time()

free_up_memory(to_keep, locals())

print('--- cell 17 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 18 start ---', flush=True)

_cell_t0 = time.time()

data = train.union(validation).union(test).select("*")
print(f"Constructing node-level features for `test` data: {data.count():,}")

run_script_in_globals(os.path.join(os.path.dirname(__file__), "_legacy_node_level_features.py"), globals())

generate_edge_features(data)
test_edges = test.select("source", "target").drop_duplicates().toPandas().set_index(
    ["source", "target"]
)
test_features = test_edges.join(
    pd.read_parquet(location_features_edges).set_index(["source", "target"]), how="left"
).reset_index()
add_node_features_to_edges(test_features, location_features_edges_test)
save_trx_features(test, location_test_trx_features)

print('--- cell 18 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 19 start ---', flush=True)

_cell_t0 = time.time()

free_up_memory(to_keep, locals())

print('--- cell 19 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 20 start ---', flush=True)

_cell_t0 = time.time()

def combine_features(location_features_trx, location_features_edges, location_features):
    features_input = spark.read.parquet(location_features_edges)
    trx_features_input = spark.read.parquet(location_features_trx)
    features_input = trx_features_input.join(
        features_input,
        on=["source", "target"],
        how="left"
    ).drop("source", "target")
    features_input.repartition(12).write.parquet(location_features, mode="overwrite")

print('--- cell 20 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 21 start ---', flush=True)

_cell_t0 = time.time()

combine_features(location_train_trx_features, location_features_edges_train, location_train_features)
combine_features(location_valid_trx_features, location_features_edges_valid, location_valid_features)
combine_features(location_test_trx_features, location_features_edges_test, location_test_features)

print('--- cell 21 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 22 start ---', flush=True)

_cell_t0 = time.time()

shutil.rmtree(MULTI_PROC_STAGING_LOCATION, ignore_errors=True)

print('--- cell 22 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 23 start ---', flush=True)

_cell_t0 = time.time()

def get_new_types(input_part_df):
    types = {"related_for": np.uint32}
    for key, value in input_part_df.dtypes.to_dict().items():
        if key.startswith("fa_"):
            types[key] = np.float16 if SCALE_TO_FLOAT_16 else np.float32
        elif pd.api.types.is_object_dtype(value) or pd.api.types.is_string_dtype(value):
            types[key] = "category"
        elif value == np.float64:
            types[key] = np.float32
    return types

print('--- cell 23 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 24 start ---', flush=True)

_cell_t0 = time.time()

columns_to_keep = sorted(
    set(spark.read.parquet(location_train_features).columns) &
    set(spark.read.parquet(location_valid_features).columns) &
    set(spark.read.parquet(location_test_features).columns)
)

print('--- cell 24 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 25 start ---', flush=True)

_cell_t0 = time.time()

positives_train = spark.read.parquet(location_train_features).where(
    sf.col("is_laundering")
).toPandas()

print('--- cell 25 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 26 start ---', flush=True)

_cell_t0 = time.time()

spark.stop()
del spark

print('--- cell 26 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 27 start ---', flush=True)

_cell_t0 = time.time()

def save_as_dmatrix(
    location_input_features, columns_to_keep, out_location, 
    split_at=-1, positives=None
):
    if split_at > 0:
        shutil.rmtree(out_location, ignore_errors=True)
        try:
            os.makedirs(out_location)
        except FileExistsError:
            pass
    input_features = pd.DataFrame()
    count = 0
    for fl in glob(f"{location_input_features}{os.sep}*.parquet"):
        count += 1
        inner = pd.read_parquet(fl, columns=columns_to_keep)
        if positives is not None:
            inner = inner.loc[~inner["is_laundering"], :]
            inner = pd.concat([inner, positives], ignore_index=True)
        inner = inner.astype(get_new_types(inner))
        input_features = pd.concat([input_features, inner], ignore_index=True)
        if (split_at > 0) and not(count % split_at):
            del inner
            input_features_labels = input_features.loc[:, ["is_laundering"]].copy(deep=True)
            del input_features["is_laundering"]
            input_dm = xgb.DMatrix(data=input_features, label=input_features_labels, enable_categorical=True)
            del input_features
            del input_features_labels
            input_dm.save_binary(f"{out_location}{os.sep}{count}.bin")
            del input_dm
            input_features = pd.DataFrame()
    if not input_features.empty:
        del inner
        input_features_labels = input_features.loc[:, ["is_laundering"]].copy(deep=True)
        del input_features["is_laundering"]
        input_dm = xgb.DMatrix(data=input_features, label=input_features_labels, enable_categorical=True)
        del input_features
        del input_features_labels
        if split_at > 0:
            input_dm.save_binary(f"{out_location}{os.sep}{count}.bin")
        else:
            input_dm.save_binary(out_location)
        del input_dm

print('--- cell 27 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 28 start ---', flush=True)

_cell_t0 = time.time()

split_at = 12
if s.FILE_SIZE == "Medium":
    split_at = 6
elif s.FILE_SIZE == "Large":
    split_at = 3

positives_train = None

save_as_dmatrix(
    location_train_features, columns_to_keep, location_train_features_dm, 
    split_at=split_at, positives=positives_train,
)

print('--- cell 28 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 29 start ---', flush=True)

_cell_t0 = time.time()

save_as_dmatrix(location_valid_features, columns_to_keep, location_valid_features_dm)

print('--- cell 29 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 30 start ---', flush=True)

_cell_t0 = time.time()

save_as_dmatrix(location_test_features, columns_to_keep, location_test_features_dm)

print('--- cell 30 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)
