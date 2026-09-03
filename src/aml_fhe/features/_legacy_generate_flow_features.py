import sys, time

print('Running converted notebook:', __file__, flush=True)


print('--- cell 0 start ---', flush=True)

_cell_t0 = time.time()

import operator

from pyspark.sql import types as st
from pyspark.sql.window import Window

print('--- cell 0 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 1 start ---', flush=True)

_cell_t0 = time.time()

FLOWS_LOCATION = f"{location_main}{os.sep}flows_intermediate{os.sep}"
shutil.rmtree(FLOWS_LOCATION, ignore_errors=True)

print('--- cell 1 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 2 start ---', flush=True)

_cell_t0 = time.time()

def save_and_load_intermediate_data(data_to_load):
    location_intermediate = f"{FLOWS_LOCATION}{uuid.uuid4()}"
    data_to_load.write.parquet(location_intermediate, mode="overwrite")
    return spark.read.parquet(location_intermediate)

print('--- cell 2 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 3 start ---', flush=True)

_cell_t0 = time.time()

data_agg = (
    data_input.groupby(["source", "target"])
    .agg(
        sf.sum("amount").alias("amount")
    )
).repartition(os.cpu_count(), "source", "target")
data_agg = save_and_load_intermediate_data(data_agg)
# print("data_agg", data_agg.count())

print('--- cell 3 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 4 start ---', flush=True)

_cell_t0 = time.time()

totals_sent = data_agg.groupby("source").agg(
    sf.sum("amount").alias("amount")
).toPandas().set_index("source")["amount"].to_dict()
totals_received = data_agg.groupby("target").agg(
    sf.sum("amount").alias("amount")
).toPandas().set_index("target")["amount"].to_dict()

print('--- cell 4 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 5 start ---', flush=True)

_cell_t0 = time.time()

def get_communities(top_n, n_hops, data_input, pov, cp, totals, to_check_in):
    if not(0 < n_hops < 11):
        raise NotImplementedError
    if top_n < 1:
        raise ValueError
    
    to_check_in = pd.DataFrame(to_check_in, columns=["match_with"])
    to_check_in.loc[:, "total"] = to_check_in["match_with"].apply(lambda x: totals[x])
    to_check_in.to_parquet(f"{FLOWS_LOCATION}to_check_in.parquet")
    to_check_in = spark.read.parquet(
        f"{FLOWS_LOCATION}to_check_in.parquet"
    ).repartition(os.cpu_count(), "match_with")
    to_check_in = save_and_load_intermediate_data(to_check_in)

    data_input = data_input.where(sf.col("source") != sf.col("target")).join(
        to_check_in,
        sf.col(pov) == sf.col("match_with"), how="inner"
    ).drop("match_with").repartition(os.cpu_count(), pov)
    data_input = save_and_load_intermediate_data(data_input)
    
    window = Window.partitionBy(sf.col(pov)).orderBy(sf.col("amount").desc())
    
    level_1st = data_input.select(
        "*", sf.row_number().over(window).alias("row_number")
    ).where(sf.col("row_number") <= top_n).drop("row_number")
    level_1st = level_1st.withColumn(
        "amount", sf.least("amount", "total")
    ).drop("total").repartition(os.cpu_count(), "source")
    level_1st = save_and_load_intermediate_data(level_1st)

    level_1st_comms = level_1st.groupby(pov).agg(
        sf.collect_list(cp).alias("nodes"), sf.collect_list("amount").alias("amounts")
    )
    level_1st_comms = save_and_load_intermediate_data(level_1st_comms)

    # print(f"Processed hop #1 | {level_1st.count():,} | {level_1st_comms.count():,}")

    result = [level_1st_comms.select("*").alias("hop_1")]
    for n_hop in range(1, n_hops):
        if n_hop == 1:
            n_minus_1 = level_1st.select("*").alias(f"hop_{n_hop + 1}")
        else:
            n_minus_1 = level_nth.select("*").alias(f"hop_{n_hop + 1}")
        for column in n_minus_1.columns:
            n_minus_1 = n_minus_1.withColumnRenamed(column, f"{column}_left")
        level_nth = n_minus_1.join(
            level_1st, sf.col("target_left") == sf.col("source"), how="inner"
        ).select(
            sf.col("source_left").alias("source"), "target", 
            sf.least("amount_left", "amount").alias("amount")
        ).where(sf.col("source") != sf.col("target")).repartition(
            os.cpu_count(), "source", "target"
        )
        level_nth = save_and_load_intermediate_data(level_nth)
    
        level_nth = level_nth.groupby(["source", "target"]).agg(
            sf.sum("amount").alias("amount")
        ).repartition(
            os.cpu_count(), pov
        )
        level_nth = save_and_load_intermediate_data(level_nth)
            
        level_nth = level_nth.join(
            to_check_in,
            sf.col(pov) == sf.col("match_with"), how="inner"
        ).drop("match_with").repartition(
            os.cpu_count(), pov
        )
        level_nth = save_and_load_intermediate_data(level_nth)
        
        level_nth = level_nth.select(
            "*", sf.row_number().over(window).alias("row_number")
        ).where(sf.col("row_number") <= top_n).drop("row_number").withColumn(
            "amount", sf.least("amount", "total")
        ).drop("total").repartition(
            os.cpu_count(), "target"
        )
        level_nth = save_and_load_intermediate_data(level_nth)

        level_nth_comms = level_nth.groupby(pov).agg(
            sf.collect_list(cp).alias("nodes"), sf.collect_list("amount").alias("amounts")
        )
        level_nth_comms = save_and_load_intermediate_data(level_nth_comms)
        
        # print(f"Processed hop #{n_hop + 1} | {level_nth.count():,} | {level_nth_comms.count():,}")

        result.append(level_nth_comms.select("*").alias(f"hop_{n_hop + 1}"))

    return result

print('--- cell 5 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 6 start ---', flush=True)

_cell_t0 = time.time()

st_flows = time.time()

print('--- cell 6 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 7 start ---', flush=True)

_cell_t0 = time.time()

# print("\nProcessing comm_as_source\n")
comm_as_source = get_communities(
    TOP_N, NUM_HOPS, data_agg, "source", "target", 
    totals_sent, nodes_source
)

print('--- cell 7 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 8 start ---', flush=True)

_cell_t0 = time.time()

# print("\nProcessing comm_as_target\n")
comm_as_target = get_communities(
    TOP_N, NUM_HOPS, data_agg, "target", "source", 
    totals_received, nodes_target
)

print('--- cell 8 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 9 start ---', flush=True)

_cell_t0 = time.time()

# print("\nProcessing comm_as_passthrough\n")
comm_as_passthrough = get_communities(
    TOP_N, NUM_HOPS, data_agg, "source", "target", 
    totals_received, nodes_passthrough
)

print('--- cell 9 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 10 start ---', flush=True)

_cell_t0 = time.time()

# print("\nProcessing comm_as_passthrough_reverse\n")
comm_as_passthrough_reverse = get_communities(
    TOP_N, NUM_HOPS, data_agg, "target", "source", 
    totals_sent, nodes_passthrough
)
# print()

print('--- cell 10 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 11 start ---', flush=True)

_cell_t0 = time.time()

def std_array(array):
    return float(np.std(array))

def max_array(array):
    return max(array)

std_array = sf.udf(std_array, st.FloatType())
max_array = sf.udf(max_array, st.FloatType())

print('--- cell 11 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 12 start ---', flush=True)

_cell_t0 = time.time()

def construct_global_features(input_data, pov, totals):
    result = pd.DataFrame(totals.items(), columns=[pov, "total"])
    for index, hop_data in enumerate(input_data):
        index += 1
        flows_nth = hop_data.select(
            pov, 
            sf.aggregate("amounts", sf.lit(0.0), operator.add).alias(f"flow_hop_{index}_total"),
            sf.size("nodes").alias(f"flow_hop_{index}_number_of_nodes"),
            std_array("amounts").alias(f"flow_hop_{index}_std_amounts"),
            max_array("amounts").alias(f"flow_hop_{index}_max_amounts"),
        ).toPandas()
        result = result.set_index(pov).join(
            flows_nth.set_index(pov), how="left"
        ).reset_index()
        result.loc[:, f"flow_hop_{index}_rel_transferred"] = (
            result.loc[:, f"flow_hop_{index}_total"] / result.loc[:, "total"]
        )
    return result.rename(columns={pov: "key"})

print('--- cell 12 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 13 start ---', flush=True)

_cell_t0 = time.time()

# %%time

# print("\ncomm_as_source_features\n")
comm_as_source_features = construct_global_features(comm_as_source, "source", totals_sent)
del comm_as_source

print('--- cell 13 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 14 start ---', flush=True)

_cell_t0 = time.time()

# %%time

# print("\ncomm_as_target_features\n")
comm_as_target_features = construct_global_features(comm_as_target, "target", totals_received)
del comm_as_target

print('--- cell 14 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 15 start ---', flush=True)

_cell_t0 = time.time()

# %%time

# print("\ncomm_as_passthrough_features\n")
comm_as_passthrough_features = construct_global_features(
    comm_as_passthrough, "source", totals_received
)
del comm_as_passthrough

print('--- cell 15 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 16 start ---', flush=True)

_cell_t0 = time.time()

# %%time

# print("\ncomm_as_passthrough_features_reverse\n")
comm_as_passthrough_features_reverse = construct_global_features(
    comm_as_passthrough_reverse, "target", totals_sent
)
del comm_as_passthrough_reverse

print('--- cell 16 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 17 start ---', flush=True)

_cell_t0 = time.time()

print(f"FLOWS ET: {round(time.time() - st_flows)}")

print('--- cell 17 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 18 start ---', flush=True)

_cell_t0 = time.time()

print("\n")
comm_as_source_features.set_index("key", inplace=True)
comm_as_target_features.set_index("key", inplace=True)
comm_as_passthrough_features.set_index("key", inplace=True)
comm_as_passthrough_features_reverse.set_index("key", inplace=True)

print('--- cell 18 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 19 start ---', flush=True)

_cell_t0 = time.time()

comm_as_source_features.columns = [f"{s.G_GLOB_PREFIX}{x}" for x in comm_as_source_features.columns]
comm_as_target_features.columns = [f"{s.G_GLOB_PREFIX}{x}" for x in comm_as_target_features.columns]
comm_as_passthrough_features.columns = [f"{s.G_GLOB_PREFIX}{x}" for x in comm_as_passthrough_features.columns]
comm_as_passthrough_features_reverse.columns = [f"{s.G_GLOB_PREFIX}{x}" for x in comm_as_passthrough_features_reverse.columns]

print('--- cell 19 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)


print('--- cell 20 start ---', flush=True)

_cell_t0 = time.time()

shutil.rmtree(FLOWS_LOCATION, ignore_errors=True)

print('--- cell 20 done in ' + str(round(time.time() - _cell_t0, 1)) + 's ---', flush=True)
