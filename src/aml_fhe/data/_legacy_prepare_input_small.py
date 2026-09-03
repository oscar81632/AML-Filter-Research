import sys, time

print('Running converted notebook:', __file__, flush=True)


print('--- cell 0 start ---', flush=True)

_cell_t0 = time.time()

import bisect
from datetime import datetime, timedelta
import os
import sys
import shutil
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import types as st
from pyspark.sql import functions as sf
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

import settings as s

print('--- cell 0 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 1 start ---', flush=True)

_cell_t0 = time.time()

assert s.FILE_SIZE == "Small", "Script suitable for `small` datasets"

print('--- cell 1 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 2 start ---', flush=True)

_cell_t0 = time.time()

STRIP_IDS = True
CALCULATE_USD_AMOUNT = True

TRX_PARTITIONS = 16

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
    # ("spark.log.level", "ERROR"),
]
shutil.rmtree("artifacts", ignore_errors=True)
shutil.rmtree("temp-spark", ignore_errors=True)
spark = (
    SparkSession.builder.appName("testing")
    .config(conf=SparkConf().setAll(SPARK_CONF))
    .getOrCreate()
)

print('--- cell 4 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 5 start ---', flush=True)

_cell_t0 = time.time()

data_account = pd.read_csv(s.ACCOUNT_FILE).set_index("Account Number")["Entity ID"].to_dict()
data_account_func = sf.udf(lambda x: data_account[x[:9]], st.StringType())

print('--- cell 5 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 6 start ---', flush=True)

_cell_t0 = time.time()

def get_id(value):
    return f"id-{hash(value)}"

print('--- cell 6 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 7 start ---', flush=True)

_cell_t0 = time.time()

try:
    os.remove(s.STAGED_DATA_CSV_LOCATION)
except FileNotFoundError:
    pass


mapping = {}
batch = []
batch_size = 200_000
with open(s.DATA_FILE) as in_file, open(s.STAGED_DATA_CSV_LOCATION, "w") as out_file:
    cnt = -1
    for line in in_file:
        cnt += 1
        if cnt == 0:
            continue
        line = line.strip()
        line_id = get_id(line)
        mapping[line_id] = cnt
        batch.append(f"{cnt},{line}\n")
        if len(batch) >= batch_size:
            out_file.writelines(batch)
            batch.clear()
            print(f"staged transactions: {cnt:,}", flush=True)
    if batch:
        if batch[-1].endswith("\n"):
            batch[-1] = batch[-1].rstrip("\n")
        out_file.writelines(batch)
        del batch

print('--- cell 7 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 8 start ---', flush=True)

_cell_t0 = time.time()

try:
    os.remove(s.STAGED_PATTERNS_CSV_LOCATION)
except FileNotFoundError:
    pass

batch = []
batch_size = 10_000
with open(s.PATTERNS_FILE) as in_file, open(s.STAGED_PATTERNS_CSV_LOCATION, "w") as out_file:
    for raw_line in in_file:
        line = raw_line.strip()
        if not line:
            batch.append("\n")
        elif line[:4].isnumeric():
            line_id = get_id(line)
            cnt = mapping[line_id]
            batch.append(f"{cnt},{line}\n")
        else:
            batch.append(f"{line}\n")
        if len(batch) >= batch_size:
            out_file.writelines(batch)
            batch.clear()
    if batch:
        out_file.writelines(batch)
        del batch

print('--- cell 8 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 9 start ---', flush=True)

_cell_t0 = time.time()

del mapping

print('--- cell 9 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 10 start ---', flush=True)

_cell_t0 = time.time()

schema = st.StructType(
    [
        st.StructField("transaction_id", st.IntegerType(), False),
        st.StructField("timestamp", st.TimestampType(), False),
        st.StructField("source_bank", st.StringType(), False),
        st.StructField("source", st.StringType(), False),
        st.StructField("target_bank", st.StringType(), False),
        st.StructField("target", st.StringType(), False),
        st.StructField("received_amount", st.FloatType(), False),
        st.StructField("receiving_currency", st.StringType(), False),
        st.StructField("sent_amount", st.FloatType(), False),
        st.StructField("sending_currency", st.StringType(), False),
        st.StructField("format", st.StringType(), False),
        st.StructField("is_laundering", st.IntegerType(), False),
    ]
)
columns = [x.name for x in schema]

print('--- cell 10 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 11 start ---', flush=True)

_cell_t0 = time.time()

with open(s.STAGED_PATTERNS_CSV_LOCATION, "r") as fl:
    patterns = fl.read()

cases = []
case_id = 0
for pattern in patterns.split("\n\n"):
    case_id += 1
    if not pattern.strip():
        continue
    pattern = pattern.split("\n")
    name = pattern.pop(0).split(" - ")[1]
    category, sub_category = name, name
    if ": " in name:
        category, sub_category = name.split(": ")
    pattern.pop()
    case = pd.DataFrame([x.split(",") for x in pattern], columns=columns)
    case.loc[:, "id"] = case_id
    case.loc[:, "type"] = category.strip().lower()
    case.loc[:, "sub_type"] = sub_category.strip().lower()
    cases.append(case)
cases = pd.concat(cases, ignore_index=True)
cases = spark.createDataFrame(cases)
cases = cases.withColumn("timestamp", sf.to_timestamp("timestamp", s.TIMESTAMP_FORMAT))
cases = cases.select("transaction_id", "id", "type", "sub_type")

print('--- cell 11 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 12 start ---', flush=True)

_cell_t0 = time.time()

CURRENCY_MAPPING = {
    "Australian Dollar": "aud",
    "Bitcoin": "btc",
    "Brazil Real": "brl",
    "Canadian Dollar": "cad",
    "Euro": "eur",
    "Mexican Peso": "mxn",
    "Ruble": "rub",
    "Rupee": "inr",
    "Saudi Riyal": "sar",
    "Shekel": "ils",
    "Swiss Franc": "chf",
    "UK Pound": "gbp",
    "US Dollar": "usd",
    "Yen": "jpy",
    "Yuan": "cny",
}

currency_code = sf.udf(lambda x: CURRENCY_MAPPING[x], st.StringType())

print('--- cell 12 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 13 start ---', flush=True)

_cell_t0 = time.time()

data = spark.read.csv(
    s.STAGED_DATA_CSV_LOCATION,
    header=False,
    schema=schema,
    timestampFormat=s.TIMESTAMP_FORMAT,
)
group_by = [
    "timestamp",
    "source_bank",
    "source",
    "target_bank",
    "target",
    "receiving_currency",
    "sending_currency",
    "format",
]
data = data.groupby(group_by).agg(
    sf.first("transaction_id").alias("transaction_id"),
    sf.collect_set("transaction_id").alias("transaction_ids"),
    sf.sum("received_amount").alias("received_amount"),
    sf.sum("sent_amount").alias("sent_amount"),
    sf.max("is_laundering").alias("is_laundering"),
)
data = data.withColumn(
    "source_currency", currency_code(sf.col("sending_currency"))
).withColumn(
    "target_currency",
    currency_code(sf.col("receiving_currency")),
)
data = data.join(cases, on="transaction_id", how="left").repartition(
    TRX_PARTITIONS, "transaction_id"
)
data = data.select(
    "transaction_id",
    "transaction_ids",
    "timestamp",
    sf.concat(sf.col("source"), sf.lit("-"), sf.col("source_currency")).alias("source"),
    sf.concat(sf.col("target"), sf.lit("-"), sf.col("target_currency")).alias("target"),
    "source_bank",
    "target_bank",
    "source_currency",
    "target_currency",
    sf.col("sent_amount").alias("source_amount"),
    sf.col("received_amount").alias("target_amount"),
    "format",
    "is_laundering",
)

data = data.withColumn("source_entity", data_account_func(sf.col("source")))
data = data.withColumn("target_entity", data_account_func(sf.col("target")))

data = data.persist(StorageLevel.DISK_ONLY)
data.count()

print('--- cell 13 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 14 start ---', flush=True)

_cell_t0 = time.time()

data.select(sf.explode("transaction_ids")).count() - data.count()

print('--- cell 14 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 15 start ---', flush=True)

_cell_t0 = time.time()

cases_data = (
    cases.join(
        data.withColumnRenamed("transaction_id", "x")
        .drop(*cases.columns)
        .select(sf.explode("transaction_ids").alias("transaction_id"), "*"),
        on="transaction_id",
        how="left",
    )
    .drop("is_laundering", "transaction_id", "transaction_ids")
    .withColumnRenamed("x", "transaction_id")
)
cases_data.toPandas().to_parquet(s.STAGED_CASES_DATA_LOCATION)
cases_data = pd.read_parquet(s.STAGED_CASES_DATA_LOCATION)

print('--- cell 15 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 16 start ---', flush=True)

_cell_t0 = time.time()

currency_rates = {
    "jpy": np.float32(0.009487665410827868),
    "cny": np.float32(0.14930721887033868),
    "cad": np.float32(0.7579775434031815),
    "sar": np.float32(0.2665884611958837),
    "aud": np.float32(0.7078143121927827),
    "ils": np.float32(0.29612081311363503),
    "chf": np.float32(1.0928961554056371),
    "usd": np.float32(1.0),
    "eur": np.float32(1.171783425225877),
    "rub": np.float32(0.012852809604990688),
    "gbp": np.float32(1.2916554735187644),
    "btc": np.float32(11879.132698717296),
    "inr": np.float32(0.013615817231245796),
    "mxn": np.float32(0.047296753463246695),
    "brl": np.float32(0.1771008654705292),
}

@sf.pandas_udf(st.FloatType())
def get_usd_amount(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    for a, b in iterator:
        yield [currency_rates[x] for x in a] * b

print('--- cell 16 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 17 start ---', flush=True)

_cell_t0 = time.time()

if STRIP_IDS:
    data = data.withColumn("source", sf.substring("source", 1, 8))
    data = data.withColumn("target", sf.substring("target", 1, 8))
if CALCULATE_USD_AMOUNT:
    data = data.withColumn("amount", get_usd_amount("source_currency", "source_amount"))

print('--- cell 17 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 18 start ---', flush=True)

_cell_t0 = time.time()

data.repartition(1).write.parquet("temp.parquet", mode="overwrite")
data_pd = pd.read_parquet("temp.parquet").sort_values("timestamp")
shutil.rmtree("temp.parquet", ignore_errors=True)

print('--- cell 18 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 19 start ---', flush=True)

_cell_t0 = time.time()

location_main = os.path.abspath(f".{os.sep}data")

location_source_dispensation = os.path.join(location_main, "source_dispensation.parquet")
location_target_accumulation = os.path.join(location_main, "target_accumulation.parquet")

print('--- cell 19 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 20 start ---', flush=True)

_cell_t0 = time.time()

num_unique = data_pd["source"].nunique()
source_dispensation = []
for index, (_, group) in enumerate(data_pd[["source", "amount"]].groupby("source")):
    group.loc[:, "source_dispensation"] = group["amount"].cumsum()
    source_dispensation.append(group)
    if not (index % 200_000):
        print(index, num_unique)
source_dispensation = pd.concat(source_dispensation, ignore_index=False)
source_dispensation.to_parquet(location_source_dispensation)

print('--- cell 20 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 21 start ---', flush=True)

_cell_t0 = time.time()

source_dispensation = pd.read_parquet(location_source_dispensation)

print('--- cell 21 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 22 start ---', flush=True)

_cell_t0 = time.time()

num_unique = data_pd["target"].nunique()
target_accumulation = []
for index, (_, group) in enumerate(data_pd[["target", "amount"]].groupby("target")):
    group.loc[:, "target_accumulation"] = group["amount"].cumsum()
    target_accumulation.append(group)
    if not (index % 200_000):
        print(index, num_unique)
target_accumulation = pd.concat(target_accumulation, ignore_index=False)
target_accumulation.to_parquet(location_target_accumulation)

print('--- cell 22 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 23 start ---', flush=True)

_cell_t0 = time.time()

target_accumulation = pd.read_parquet(location_target_accumulation)

print('--- cell 23 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 24 start ---', flush=True)

_cell_t0 = time.time()

data_pd = source_dispensation[["source_dispensation"]].join(
    target_accumulation[["target_accumulation"]], how="outer"
).join(data_pd)
data_pd.sort_index(inplace=True)

print('--- cell 24 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 25 start ---', flush=True)

_cell_t0 = time.time()

dispensation_mapping = {}
for source, group in data_pd[["source", "source_dispensation"]].groupby("source"):
    dispensation_mapping[source] = (group.index.tolist(), group["source_dispensation"].tolist())

accumulation_mapping = {}
for target, group in data_pd[["target", "target_accumulation"]].groupby("target"):
    accumulation_mapping[target] = (group.index.tolist(), group["target_accumulation"].tolist())

print('--- cell 25 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 26 start ---', flush=True)

_cell_t0 = time.time()

def get_dis_acc_data(node, mapping_dis, mapping_acc, trx_id):
    data_dis = mapping_dis.get(node)
    if data_dis is None:
        data_acc = mapping_acc[node]
        index_acc = bisect.bisect_right(data_acc[0], trx_id)
        if index_acc:
            index_acc -= 1
        else:
            return 0, 0
        return 0, data_acc[1][index_acc]
    data_acc = mapping_acc.get(node)
    if data_acc is None:
        data_dis = mapping_dis[node]
        index_dis = bisect.bisect_right(data_dis[0], trx_id)
        if index_dis:
            index_dis -= 1
        else:
            return 0, 0
        return data_dis[1][index_dis], 0
    index_dis = bisect.bisect_right(data_dis[0], trx_id)
    index_acc = bisect.bisect_right(data_acc[0], trx_id)
    so_far_dispensed = 0
    if index_dis:
        index_dis -= 1
        so_far_dispensed = data_dis[1][index_dis]
    so_far_accumulated = 0
    if index_acc:
        index_acc -= 1
        so_far_accumulated = data_acc[1][index_acc]
    return so_far_dispensed, so_far_accumulated


def source_dis_acc_data(row):
    return get_dis_acc_data(row["source"], dispensation_mapping, accumulation_mapping, row.name)


def target_dis_acc_data(row):
    return get_dis_acc_data(row["target"], dispensation_mapping, accumulation_mapping, row.name)

print('--- cell 26 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 27 start ---', flush=True)

_cell_t0 = time.time()

data_pd.loc[:, "dis_acc_source"] = data_pd.apply(source_dis_acc_data, axis=1)
data_pd.loc[:, "dis_acc_target"] = data_pd.apply(target_dis_acc_data, axis=1)

data_pd.loc[:, "source_positive_balance"] = data_pd.loc[:, "dis_acc_source"].apply(
    lambda x: x[1] - x[0] if x[1] > x[0] else 0
)
data_pd.loc[:, "source_negative_balance"] = data_pd.loc[:, "dis_acc_source"].apply(
    lambda x: x[0] - x[1] if x[0] > x[1] else 0
)
data_pd.loc[:, "target_positive_balance"] = data_pd.loc[:, "dis_acc_target"].apply(
    lambda x: x[1] - x[0] if x[1] > x[0] else 0
)
data_pd.loc[:, "target_negative_balance"] = data_pd.loc[:, "dis_acc_target"].apply(
    lambda x: x[0] - x[1] if x[0] > x[1] else 0
)

del data_pd["dis_acc_source"]
del data_pd["dis_acc_target"]

print('--- cell 27 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 28 start ---', flush=True)

_cell_t0 = time.time()

source_firsts = data_pd.groupby("source").agg(first_trx=("timestamp", "min"))
target_firsts = data_pd.groupby("target").agg(first_trx=("timestamp", "min"))
active_since = source_firsts.join(target_firsts, lsuffix="_left", how="outer").fillna(datetime.now())
active_since.loc[:, "active_since"] = active_since.apply(lambda x: min([x["first_trx_left"], x["first_trx"]]), axis=1)
active_since = active_since.loc[:, ["active_since"]]
active_since.sort_values("active_since", inplace=True)

active_since = active_since["active_since"].to_dict()
last_trx_ts = data_pd["timestamp"].max() + timedelta(hours=1)
first_trx_ts = data_pd["timestamp"].min() - timedelta(hours=1)
active_for = {k : (last_trx_ts - v).total_seconds() for k, v in active_since.items()}

data_pd.loc[:, "source_active_for"] = data_pd.apply(
    lambda x: (x["timestamp"] - active_since[x["source"]]).total_seconds(), axis=1
)
data_pd.loc[:, "target_active_for"] = data_pd.apply(
    lambda x: (x["timestamp"] - active_since[x["target"]]).total_seconds(), axis=1
)

print('--- cell 28 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 29 start ---', flush=True)

_cell_t0 = time.time()

spark.createDataFrame(data_pd).repartition(TRX_PARTITIONS).write.parquet(s.STAGED_DATA_LOCATION, mode="overwrite")

print('--- cell 29 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 30 start ---', flush=True)

_cell_t0 = time.time()

data = spark.read.parquet(s.STAGED_DATA_LOCATION)

print('--- cell 30 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 31 start ---', flush=True)

_cell_t0 = time.time()

assert data.count() == data.select("transaction_id").distinct().count()

print('--- cell 31 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 32 start ---', flush=True)

_cell_t0 = time.time()

data.count(), cases_data.shape[0], cases_data["transaction_id"].nunique(), cases_data["id"].nunique()
# (179504480, 137936, 137933, 16467)

print('--- cell 32 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 33 start ---', flush=True)

_cell_t0 = time.time()

shutil.rmtree("temp-spark", ignore_errors=True)

print('--- cell 33 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)
