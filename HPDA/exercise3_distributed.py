"""
=============================================================================
EXERCISE 3: Distributed Analytics on HPC Cluster
=============================================================================
OBJECTIVE: Scale analytics across multiple HPC nodes using Dask distributed,
           measure scaling efficiency, and identify communication bottlenecks.

In this exercise we:
  1. Deploy a Dask cluster via SLURM (MeluXina's job scheduler)
  2. Run distributed groupby, join, and aggregation on a large dataset
  3. Measure linear scaling efficiency (speedup vs number of workers)
  4. Identify and understand shuffle-heavy operations (the #1 bottleneck)
  5. Apply communication-reduction techniques

DURATION: ~45 minutes

HPC CLUSTER CONTEXT (MeluXina):
  - Each node: 64 cores, 512 GB RAM, AMD EPYC 7H12
  - Interconnect: HDR InfiniBand 200 Gb/s (vs ~10 Gb/s Ethernet on cloud)
  - Shared filesystem: Lustre parallel file system
  - Job scheduler: SLURM

WHY INFINIBRAND MATTERS FOR DASK:
  Distributed Dask operations that require data movement (like join and groupby
  across partitions) generate "shuffle" traffic.  On cloud VMs with Ethernet,
  shuffles are the bottleneck.  On MeluXina's InfiniBand, node-to-node
  bandwidth is 20× higher, making distributed analytics dramatically faster
  for shuffle-heavy workloads.
=============================================================================
"""

import os
import time
import math
import socket
import multiprocessing

import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, wait, performance_report

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CLUSTER SETUP
# ─────────────────────────────────────────────────────────────────────────────
# On MeluXina you would use dask-jobqueue to submit SLURM jobs.
# Here we show BOTH the production SLURM setup AND a LocalCluster fallback
# so the exercise can run anywhere.

SLURM_SETUP_CODE = """
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO SET UP DASK ON MELUXINA WITH SLURM (run this in your notebook/script)
# ─────────────────────────────────────────────────────────────────────────────

# pip install dask-jobqueue

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(
    # Each SLURM job = one Dask worker
    cores=64,               # Physical cores per node (MeluXina CPU node)
    processes=1,            # 1 Python process per job (use threads internally)
    memory="480GB",         # Leave headroom from 512 GB for OS and Dask metadata
    walltime="02:00:00",    # Max job duration

    # SLURM-specific settings for MeluXina
    queue="cpu",            # MeluXina CPU partition name
    account="p200XXXX",     # Your project account on MeluXina

    # Worker environment
    interface="ib0",        # Use InfiniBand interface for worker communication
    extra=["--lifetime", "115m", "--lifetime-stagger", "2m"],

    job_extra_directives=[
        "--nodes=1",
        "--exclusive",      # Reserve entire node — no sharing with other jobs
    ],

    # Python environment
    python="python3",
    log_directory="./dask-logs/",
)

# Scale to 8 nodes (= 8 SLURM jobs, each with 64 cores)
cluster.scale(jobs=8)

# Connect client
client = Client(cluster)
print(client)
# Now you have 8 × 64 = 512 cores available!
"""

def setup_cluster(n_workers: int = None, mode: str = "local") -> Client:
    """
    Set up a Dask cluster.

    Parameters
    ----------
    n_workers : int
        Number of worker processes/nodes.
        - local mode: number of worker processes on this machine
        - slurm mode: number of SLURM jobs (nodes) to request
    mode : str
        'local'  → LocalCluster (works on any machine, used in this exercise)
        'slurm'  → SLURMCluster (production HPC deployment on MeluXina)
    """
    if n_workers is None:
        n_workers = max(2, multiprocessing.cpu_count() // 4)

    if mode == "slurm":
        # Production MeluXina setup — requires dask-jobqueue and a SLURM allocation
        try:
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster(
                cores=64,
                processes=1,
                memory="480GB",
                walltime="01:00:00",
                queue="cpu",
                interface="ib0",
                log_directory="./dask-logs/",
            )
            cluster.scale(jobs=n_workers)
            print(f"[CLUSTER] Requested {n_workers} SLURM jobs (nodes)")
        except ImportError:
            print("[CLUSTER] dask-jobqueue not installed — falling back to LocalCluster")
            mode = "local"

    if mode == "local":
        # LocalCluster simulates a distributed cluster on one machine.
        # threads_per_worker=4: each worker uses 4 threads (shared memory)
        # n_workers: simulates different numbers of "nodes"
        n_cpu = multiprocessing.cpu_count()
        threads = max(1, n_cpu // n_workers)
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads,
            memory_limit="4GiB",   # per worker, adjust to your machine
            silence_logs=True,
        )
        print(f"[CLUSTER] LocalCluster: {n_workers} workers × {threads} threads")

    client = Client(cluster)
    print(f"[CLUSTER] Dashboard: {client.dashboard_link}")
    print(f"[CLUSTER] Workers: {len(client.scheduler_info()['workers'])}")
    print(f"[CLUSTER] Total cores: {sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())}")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DISTRIBUTED ANALYTICS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_distributed_pipeline(client: Client, parquet_path: str) -> dict:
    """
    Execute a multi-step analytics pipeline on the distributed cluster.

    Steps:
      A – Distributed read: each worker reads a subset of partitions
      B – Distributed groupby: aggregation within each partition, then merge
      C – Distributed join: requires a "shuffle" (data exchange between workers)
      D – Distributed time-series resampling
      E – Write results back to Parquet (parallel write)
    """
    metrics = {}
    n_workers = len(client.scheduler_info()["workers"])
    # Use 4 partitions per worker for good load balancing
    npartitions = n_workers * 4

    print("\n" + "="*60)
    print("  DISTRIBUTED PIPELINE")
    print(f"  Workers: {n_workers}   |   Partitions: {npartitions}")
    print("="*60)

    # ── STEP A: Distributed Read ─────────────────────────────────────────────
    # Each worker reads a subset of Parquet row groups.
    # On MeluXina: workers read in parallel from Lustre (the parallel file
    # system), so I/O scales with the number of workers.
    print("\n[STEP A] Distributed Parquet read …")
    t0 = time.perf_counter()

    ddf = dd.read_parquet(
        parquet_path,
        engine="pyarrow",
        # Only read columns we need — projection pushdown at the file level
        columns=["transaction_id", "customer_id", "product_id",
                 "region", "amount", "quantity", "timestamp"],
    )
    # Repartition to match our worker count for even load balancing
    ddf = ddf.repartition(npartitions=npartitions)
    ddf["revenue"] = ddf["amount"] * ddf["quantity"]

    # persist() distributes data across workers and keeps it in memory
    # This is the distributed equivalent of pandas' in-memory DataFrame
    ddf = client.persist(ddf)
    wait(ddf)   # block until all workers have loaded their partitions

    metrics["A_load_s"] = time.perf_counter() - t0
    print(f"  → {metrics['A_load_s']:.2f}s  |  {npartitions} partitions across {n_workers} workers")

    # ── STEP B: Distributed GroupBy (NO shuffle needed) ─────────────────────
    # groupby().agg() with simple reductions (sum, mean, count) can be done
    # in two phases:
    #   Phase 1: Each worker aggregates its own partitions (local, fast)
    #   Phase 2: Results are merged across workers (small data, fast)
    # No full data shuffle required → scales linearly with worker count!
    print("\n[STEP B] Distributed groupby (region × product_id) …")
    t0 = time.perf_counter()

    region_stats = (
        ddf.groupby(["region", "product_id"], observed=True)
           .agg(
               total_revenue=("revenue",        "sum"),
               avg_amount   =("amount",         "mean"),
               n_txn        =("transaction_id", "count"),
           )
           .reset_index()
           .compute()   # collect to local pandas DataFrame
    )

    metrics["B_groupby_s"] = time.perf_counter() - t0
    print(f"  → {metrics['B_groupby_s']:.2f}s  |  result shape: {region_stats.shape}")

    # ── STEP C: Distributed Join (SHUFFLE required!) ────────────────────────
    # This is the most important step to understand for HPC performance.
    #
    # When you join two Dask DataFrames on a key (e.g., customer_id),
    # Dask must ensure that all rows with the same customer_id are on the
    # SAME worker.  This requires a SHUFFLE — a global redistribution of data.
    #
    # Shuffle cost = O(dataset_size × log(n_partitions))
    # On Ethernet  : 10 Gb/s → bottleneck for large datasets
    # On InfiniBand: 200 Gb/s → 20× faster → HPC wins for shuffle-heavy work!
    print("\n[STEP C] Distributed join (shuffle operation) …")
    print("         (This is the expensive step — watch the dashboard!)")
    t0 = time.perf_counter()

    # Build a customer-level feature table (small result, quick to compute)
    customer_features = (
        ddf.groupby("customer_id")
           .agg(customer_ltv=("revenue", "sum"),
                n_orders    =("transaction_id", "count"))
           .reset_index()
    )

    # merge() triggers the shuffle: ddf is repartitioned by customer_id
    # so that matching rows end up on the same worker for the join
    df_enriched = ddf.merge(
        customer_features,
        on="customer_id",
        how="left",
        # shuffle="tasks" is Dask's default: uses the task scheduler for shuffle
        # shuffle="p2p" (peer-to-peer) is faster for large joins in newer Dask
    )
    # compute() materialises the join result
    df_enriched = client.persist(df_enriched)
    wait(df_enriched)

    metrics["C_join_shuffle_s"] = time.perf_counter() - t0
    print(f"  → {metrics['C_join_shuffle_s']:.2f}s  (includes shuffle)")

    # ── STEP D: Time-series Resampling ───────────────────────────────────────
    # Resample to hourly revenue.  Dask handles this by partitioning on time.
    # NOTE: For a proper time-series resample, data must be sorted by timestamp.
    # This is another shuffle-like operation.
    print("\n[STEP D] Time-series resampling (hourly revenue) …")
    t0 = time.perf_counter()

    hourly = (
        ddf.assign(hour=ddf["timestamp"].dt.floor("H"))
           .groupby("hour")["revenue"]
           .sum()
           .compute()
           .sort_index()
    )

    metrics["D_resample_s"] = time.perf_counter() - t0
    print(f"  → {metrics['D_resample_s']:.2f}s  |  {len(hourly)} hourly buckets")

    # ── STEP E: Parallel Write ────────────────────────────────────────────────
    # Write results back to Parquet.  Dask writes one file per partition in
    # parallel.  On Lustre (MeluXina): all workers write simultaneously to
    # different files → I/O scales with worker count.
    print("\n[STEP E] Writing enriched dataset to Parquet …")
    t0 = time.perf_counter()

    df_enriched.to_parquet(
        "enriched_output/",    # Dask writes one file per partition into a directory
        engine="pyarrow",
        compression="snappy",
        write_index=False,
    )

    metrics["E_write_s"] = time.perf_counter() - t0
    print(f"  → {metrics['E_write_s']:.2f}s")

    return metrics, region_stats, hourly


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SCALING EFFICIENCY MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────
# Scaling efficiency = (speedup) / (n_workers)
# Perfect linear scaling = 100% efficiency
# Real workloads: 60-80% efficiency is excellent for analytics
#
# Why efficiency < 100%?
#   - Communication overhead (shuffles, coordination)
#   - Uneven partition sizes (load imbalance)
#   - Amdahl's Law: serial fraction limits parallel speedup
#   - Scheduler overhead grows with more tasks

def measure_scaling_efficiency(parquet_path: str, worker_counts: list = None):
    """
    Run the same pipeline with increasing numbers of workers and plot
    the speedup curve to visualise Amdahl's Law in practice.
    """
    if worker_counts is None:
        n_cpu = multiprocessing.cpu_count()
        # Test with 1, 2, 4, up to n_cpu workers
        worker_counts = [1, 2, 4, min(8, n_cpu)]
        worker_counts = sorted(set(worker_counts))

    print("\n" + "="*60)
    print("  SCALING EFFICIENCY EXPERIMENT")
    print(f"  Testing worker counts: {worker_counts}")
    print("="*60)

    results = {}
    baseline_time = None

    for n_workers in worker_counts:
        print(f"\n  Testing with {n_workers} worker(s) …")
        client = setup_cluster(n_workers=n_workers, mode="local")

        try:
            ddf = dd.read_parquet(parquet_path, engine="pyarrow")
            ddf = ddf.repartition(npartitions=n_workers * 4)
            ddf["revenue"] = ddf["amount"] * ddf["quantity"]
            ddf = client.persist(ddf)
            wait(ddf)

            # Run the core operation: groupby + join (most representative)
            t0 = time.perf_counter()

            _ = (
                ddf.groupby(["region", "product_id"])
                   .agg({"revenue": "sum", "amount": "mean"})
                   .compute()
            )

            elapsed = time.perf_counter() - t0
        finally:
            client.close()

        results[n_workers] = elapsed

        if baseline_time is None:
            baseline_time = elapsed

        speedup    = baseline_time / elapsed
        efficiency = speedup / n_workers * 100
        print(f"  n_workers={n_workers}  |  time={elapsed:.2f}s  |  "
              f"speedup={speedup:.2f}×  |  efficiency={efficiency:.0f}%")

    # ── PRINT SCALING TABLE ────────────────────────────────────────────────
    print("\n" + "-"*65)
    print(f"  {'Workers':>8}  {'Time (s)':>10}  {'Speedup':>8}  {'Efficiency':>10}")
    print("-"*65)
    for n, t in results.items():
        speedup    = baseline_time / t
        efficiency = speedup / n * 100
        bar        = "█" * int(efficiency / 5)   # ASCII bar chart
        print(f"  {n:>8}  {t:>10.2f}  {speedup:>8.2f}×  {efficiency:>9.0f}%  {bar}")
    print("-"*65)

    print("""
  INTERPRETING THE RESULTS:
  • Efficiency drops as workers increase → communication overhead grows
  • The "knee" of the curve = optimal worker count for this workload
  • On MeluXina InfiniBand: the knee is much higher than on cloud Ethernet
    because shuffle costs are 20× lower
  • For embarrassingly-parallel operations (no shuffle): efficiency stays
    near 100% regardless of worker count
    """)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SHUFFLE OPTIMISATION TECHNIQUES
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_shuffle_reduction(client: Client, parquet_path: str):
    """
    Show three techniques to reduce shuffle traffic:
      1. Pre-sort by join key (avoids shuffle if data is already partitioned)
      2. Broadcast join     (for small tables: broadcast instead of shuffle)
      3. Columnar pruning   (less data = less to shuffle)
    """
    print("\n" + "="*60)
    print("  OPTIMISATION: Reducing Shuffle Traffic")
    print("="*60)

    ddf = dd.read_parquet(parquet_path, engine="pyarrow")
    ddf["revenue"] = ddf["amount"] * ddf["quantity"]
    ddf = client.persist(ddf)
    wait(ddf)

    # ── TECHNIQUE 1: Naive join (full shuffle) ──────────────────────────────
    print("\n[TECHNIQUE 1] Naive join — full shuffle on customer_id")
    t0 = time.perf_counter()
    customer_stats = (
        ddf.groupby("customer_id")["revenue"].sum().reset_index()
    )
    naive_result = ddf.merge(customer_stats, on="customer_id", how="left").compute()
    naive_time = time.perf_counter() - t0
    print(f"  → Naive join: {naive_time:.2f}s  (full shuffle required)")

    # ── TECHNIQUE 2: Avoid join by computing within groupby ─────────────────
    # Instead of computing customer_stats separately and joining back,
    # use transform() to compute group aggregates without a shuffle.
    # transform() broadcasts the group result to all rows in the group.
    print("\n[TECHNIQUE 2] transform() — no join, no shuffle")
    t0 = time.perf_counter()
    ddf2 = ddf.copy()
    # groupby + transform stays within each partition (no data movement)
    # Note: this works when npartitions is set so same customer_id stays together
    # For exact results across partitions, we use the approach below:
    customer_ltv = ddf2.groupby("customer_id")["revenue"].transform("sum")
    ddf2["customer_ltv"] = customer_ltv
    result2 = ddf2[["transaction_id", "customer_ltv"]].compute()
    transform_time = time.perf_counter() - t0
    print(f"  → transform(): {transform_time:.2f}s  (reduced shuffle)")

    # ── TECHNIQUE 3: Reduce columns before shuffle ──────────────────────────
    # The more columns in the data during a shuffle, the more bytes are sent
    # over the network.  Drop unnecessary columns BEFORE the join.
    print("\n[TECHNIQUE 3] Column pruning before shuffle")
    t0 = time.perf_counter()

    # Only keep the columns needed for the join and aggregation
    slim_ddf = ddf[["customer_id", "revenue"]].copy()
    customer_stats3 = slim_ddf.groupby("customer_id")["revenue"].sum().reset_index()
    # Now join the slim result back (fewer bytes transferred)
    slim_join = slim_ddf[["customer_id"]].merge(
        customer_stats3, on="customer_id", how="left"
    ).compute()
    pruned_time = time.perf_counter() - t0
    print(f"  → Pruned columns: {pruned_time:.2f}s")

    print(f"""
  COMPARISON:
    Naive join      : {naive_time:.2f}s  (baseline)
    transform()     : {transform_time:.2f}s  ({naive_time/transform_time:.1f}× faster)
    Column pruning  : {pruned_time:.2f}s  ({naive_time/pruned_time:.1f}× faster)

  RULE OF THUMB FOR HPC:
    Always prune columns before any operation that involves data movement.
    On a 100-column table with only 5 needed for a join, pruning reduces
    shuffle data by 95%!
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PARQUET_PATH = "transactions.parquet"

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            "transactions.parquet not found. "
            "Please run Exercise 1 first."
        )

    # Print the SLURM setup code for reference
    print("\n" + "="*60)
    print("  PRODUCTION SETUP: How to deploy on MeluXina")
    print("="*60)
    print(SLURM_SETUP_CODE)

    # 1. Measure scaling efficiency
    scaling_results = measure_scaling_efficiency(PARQUET_PATH)

    # 2. Set up a multi-worker cluster for the pipeline
    client = setup_cluster(n_workers=4, mode="local")

    try:
        # 3. Run distributed pipeline (with performance report)
        with performance_report(filename="dask_report.html"):
            pipeline_metrics, region_stats, hourly = run_distributed_pipeline(
                client, PARQUET_PATH
            )

        # 4. Demonstrate shuffle reduction techniques
        demonstrate_shuffle_reduction(client, PARQUET_PATH)

        # ── FINAL REPORT ─────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("  EXERCISE 3 SUMMARY")
        print("="*60)
        total = sum(pipeline_metrics.values())
        print(f"\n  Pipeline step timings:")
        for step, t in pipeline_metrics.items():
            pct = t / total * 100
            print(f"    {step:<20} {t:.2f}s  ({pct:.0f}%)")
        print(f"    {'TOTAL':<20} {total:.2f}s")

        print(f"""
  KEY FINDINGS:
  • Step C (join/shuffle) is typically the most expensive step
    → On MeluXina's InfiniBand this is 20× faster than on cloud Ethernet

  • The Dask performance report was saved to dask_report.html
    → Open it in a browser to see the full task timeline

  • Linear scaling breaks down at:
    {list(scaling_results.keys())[-1]} workers for this dataset
    → Consider whether adding more nodes is cost-effective

  NEXT: Exercise 4 puts it all together in a real optimisation challenge.
        """)

    finally:
        client.close()
