"""
=============================================================================
EXERCISE 2: Single-Node Parallelization with Dask
=============================================================================
OBJECTIVE: Scale analytics workloads on a single multi-core HPC node
           using Dask, without changing the pandas-style API.

In this exercise we:
  1. Learn Dask's three key abstractions (DataFrame, delayed, Futures)
  2. Choose the right scheduler for our workload (threads vs processes)
  3. Tune chunk sizes and observe their impact on performance
  4. Process a dataset larger than RAM (out-of-core computation)
  5. Use the Dask dashboard to visualise the task graph

PREREQUISITES: Exercise 1 completed (transactions.parquet must exist)
DURATION: ~45 minutes

KEY CONCEPT — Why Dask?
  pandas is inherently single-threaded.  On a 64-core MeluXina node you are
  using 1/64 of the available CPU power.  Dask divides the data into
  *partitions* (chunks) and processes them in parallel across all cores.
  The API stays almost identical to pandas, so the migration cost is low.
=============================================================================
"""

import os
import time
import multiprocessing
import pandas as pd
import numpy as np

# Dask imports
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, performance_report

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — UNDERSTANDING DASK FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────

def explain_dask_concepts():
    """
    Demonstrates Dask's lazy evaluation model.

    CRITICAL INSIGHT:  When you call dask operations, NOTHING is executed.
    Dask builds a *task graph* — a directed acyclic graph (DAG) that describes
    what needs to happen.  Computation only happens when you call .compute().

    This lazy model enables two optimisations:
      1. Dask can rearrange tasks for better parallelism
      2. Dask can fuse consecutive operations to reduce data movement
    """
    print("\n" + "="*60)
    print("  DASK CONCEPT: Lazy Evaluation & Task Graphs")
    print("="*60)

    # Create a tiny Dask DataFrame from a pandas one
    pdf = pd.DataFrame({"x": range(1000), "y": np.random.randn(1000)})

    # npartitions=4 splits the 1000-row DataFrame into 4 chunks of 250 rows
    # On a 64-core node you would use npartitions = 2 × n_cores as a starting point
    ddf = dd.from_pandas(pdf, npartitions=4)

    print(f"\n  pandas shape  : {pdf.shape}")
    print(f"  Dask partitions: {ddf.npartitions}")
    print(f"  Rows per partition (approx): {len(pdf) // ddf.npartitions}")

    # This looks like pandas but returns another Dask object – nothing executed yet
    result = ddf[ddf["x"] > 500].groupby("x")["y"].sum()

    print(f"\n  type(result) = {type(result)}")
    print("  result.compute() has NOT been called — no work done yet!")

    # Visualise the task graph (requires graphviz; skip if not installed)
    try:
        result.visualize(filename="task_graph.png", optimize_graph=True)
        print("  Task graph saved to task_graph.png")
    except Exception:
        print("  (graphviz not installed — skipping task graph visualisation)")

    # NOW trigger computation
    t0 = time.perf_counter()
    computed = result.compute()
    print(f"\n  .compute() completed in {time.perf_counter()-t0:.3f}s")
    print(f"  Result shape: {computed.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SCHEDULER SELECTION
# ─────────────────────────────────────────────────────────────────────────────
# Dask has three built-in schedulers.  Choosing the wrong one can HURT performance.
#
#  ┌──────────────────┬─────────────────────────────────────────────────────┐
#  │ Scheduler        │ Best for                                            │
#  ├──────────────────┼─────────────────────────────────────────────────────┤
#  │ synchronous      │ Debugging only. No parallelism.                     │
#  │ threads          │ I/O-bound work, numpy/pandas (release the GIL).     │
#  │                  │ Shared memory → zero copy overhead.                  │
#  │ processes        │ Pure-Python CPU-bound code that holds the GIL.      │
#  │                  │ Each worker is an independent Python process.        │
#  │ distributed      │ Multi-node clusters or complex DAGs with data        │
#  │   (dask.dist.)   │ locality requirements. Always use for HPC clusters.  │
#  └──────────────────┴─────────────────────────────────────────────────────┘
#
# For pandas / numpy analytics on a single node: threads scheduler is best.
# numpy releases the Python GIL during heavy computation, so threads truly run
# in parallel.  The processes scheduler copies data between processes (pickle
# overhead) which often makes pandas analytics SLOWER than threads.

def benchmark_schedulers(parquet_path: str) -> dict:
    """
    Run the same groupby aggregation with each Dask scheduler and compare.
    """
    print("\n" + "="*60)
    print("  BENCHMARK: Dask Schedulers")
    print("="*60)

    n_cores = multiprocessing.cpu_count()
    print(f"\n  Available CPU cores on this node: {n_cores}")

    # Load with Dask — this is instant (lazy), no data is read yet
    ddf = dd.read_parquet(parquet_path, engine="pyarrow")
    ddf["revenue"] = ddf["amount"] * ddf["quantity"]

    # We'll run the same operation three times with different schedulers
    def run_groupby(scheduler_name):
        t0 = time.perf_counter()
        result = (
            ddf.groupby("region")
               .agg({"revenue": "sum", "amount": "mean"})
               .compute(scheduler=scheduler_name)   # ← scheduler is chosen here
        )
        elapsed = time.perf_counter() - t0
        print(f"  [{scheduler_name:>12}]  {elapsed:.2f}s  |  result shape: {result.shape}")
        return elapsed

    results = {}
    for sched in ["synchronous", "threads", "processes"]:
        results[sched] = run_groupby(sched)

    # The winner tells us whether our workload is GIL-bound or not
    fastest = min(results, key=results.get)
    print(f"\n  ✓ Fastest scheduler for this workload: '{fastest}'")
    print(  "    (For pandas/numpy: threads wins because numpy releases the GIL)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CHUNK SIZE TUNING
# ─────────────────────────────────────────────────────────────────────────────
# Chunk size (npartitions) has a U-shaped effect on performance:
#
#  Too few partitions → cores sit idle (not enough work to parallelise)
#  Too many partitions → scheduler overhead dominates (thousands of tiny tasks)
#  Sweet spot          → each task takes ~100ms–1s; few more partitions than cores
#
# Rule of thumb for Parquet:
#   target_chunk_size_MB = 256–512 MB
#   npartitions = ceil(total_size_MB / target_chunk_size_MB)
#   npartitions >= 2 × n_cores  (to keep all cores busy during stragglers)

def benchmark_chunk_sizes(parquet_path: str) -> dict:
    """
    Run the same pipeline with different numbers of partitions.
    Observe how scheduler overhead vs. parallelism trade-off plays out.
    """
    print("\n" + "="*60)
    print("  BENCHMARK: Chunk Size (npartitions)")
    print("="*60)

    n_cores = multiprocessing.cpu_count()
    file_mb = os.path.getsize(parquet_path) / 1_048_576
    print(f"  File size: {file_mb:.0f} MB   |   Cores: {n_cores}")

    results = {}
    # Test partition counts from 1 (no parallelism) to 8 × n_cores
    for nparts in [1, n_cores // 2, n_cores, n_cores * 2, n_cores * 4]:
        nparts = max(1, nparts)  # ensure at least 1

        # repartition() redistributes data into exactly nparts partitions
        # In production: set npartitions when reading (more efficient)
        ddf = dd.read_parquet(parquet_path, engine="pyarrow")
        ddf = ddf.repartition(npartitions=nparts)
        ddf["revenue"] = ddf["amount"] * ddf["quantity"]

        t0 = time.perf_counter()
        _ = (
            ddf.groupby(["region", "product_id"])
               .agg({"revenue": "sum", "amount": "mean", "quantity": "sum"})
               .compute(scheduler="threads")
        )
        elapsed = time.perf_counter() - t0
        results[nparts] = elapsed
        print(f"  npartitions={nparts:>4}  →  {elapsed:.2f}s  "
              f"(~{file_mb/nparts:.0f} MB/partition)")

    optimal = min(results, key=results.get)
    print(f"\n  ✓ Optimal npartitions for this file: {optimal}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OUT-OF-CORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
# "Out-of-core" means processing data that does not fit in RAM.
# Dask achieves this by processing one partition at a time and spilling to disk
# when memory is tight.  pandas would crash with MemoryError.
#
# How it works:
#   1. Dask reads partition 1 from disk
#   2. Applies operations to partition 1 in RAM
#   3. Writes intermediate result to disk (or passes to next stage)
#   4. Releases partition 1's memory
#   5. Reads partition 2 … and so on
#
# Memory limit per worker = total_RAM / n_workers
# When a worker exceeds 60% of its limit → spills to disk
# When a worker exceeds 80% of its limit → pauses until memory freed

def demonstrate_out_of_core(parquet_path: str):
    """
    Simulate an out-of-core workflow by artificially restricting memory
    and showing that Dask still completes the job.
    In a real scenario: your file is 200 GB and your node has 128 GB RAM.
    """
    print("\n" + "="*60)
    print("  DEMO: Out-of-Core Computation with Memory Limit")
    print("="*60)

    # Set up a LocalCluster with a small memory limit to force spilling.
    # In a classroom: set memory_limit to about 50% of the file size
    # to see the spill-to-disk behaviour clearly.
    file_mb = os.path.getsize(parquet_path) / 1_048_576
    # Use a low memory limit to force spilling behaviour
    memory_limit_mb = max(256, int(file_mb * 0.5))

    print(f"\n  File size  : {file_mb:.0f} MB")
    print(f"  Memory limit per worker: {memory_limit_mb} MB  (forcing spill behaviour)")

    with LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        memory_limit=f"{memory_limit_mb}MiB",
        silence_logs=True,      # reduce noise in classroom
    ) as cluster:
        with Client(cluster) as client:
            print(f"\n  Dask dashboard: {client.dashboard_link}")
            print("  ↑ Open this URL in your browser to watch tasks execute in real-time!\n")

            # Use many small partitions so Dask can manage memory carefully
            ddf = dd.read_parquet(parquet_path, engine="pyarrow")
            n_parts = max(8, multiprocessing.cpu_count())
            ddf = ddf.repartition(npartitions=n_parts)
            ddf["revenue"] = ddf["amount"] * ddf["quantity"]

            print(f"  Processing {ddf.npartitions} partitions with 2 workers …")
            t0 = time.perf_counter()

            # persist() loads data into distributed memory (or spills if needed)
            # It's non-blocking — we can do other things while data loads
            ddf_persisted = client.persist(ddf)

            # compute() triggers actual execution and returns a pandas result
            result = (
                ddf_persisted
                .groupby("region")
                .agg({"revenue": "sum", "amount": "mean"})
                .compute()
            )

            elapsed = time.perf_counter() - t0
            print(f"\n  ✓ Completed in {elapsed:.2f}s")
            print(f"  Result:\n{result.to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PANDAS vs DASK COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def pandas_vs_dask(parquet_path: str) -> dict:
    """
    Run the same complex pipeline in pandas and Dask and compare wall-clock time.
    This is the core benchmark of Exercise 2.
    """
    print("\n" + "="*60)
    print("  BENCHMARK: pandas vs Dask (single node)")
    print("="*60)

    n_cores = multiprocessing.cpu_count()
    npartitions = n_cores * 2   # rule of thumb: 2 partitions per core

    # ── PANDAS ───────────────────────────────────────────────────────────────
    print("\n  [pandas] Running pipeline …")
    t0 = time.perf_counter()

    pdf = pd.read_parquet(parquet_path, engine="pyarrow")
    pdf["revenue"] = pdf["amount"] * pdf["quantity"]
    pdf_result = (
        pdf.groupby(["region", "product_id"])
           .agg(total_revenue=("revenue", "sum"),
                avg_amount=("amount", "mean"),
                n_txn=("transaction_id", "count"))
           .reset_index()
    )
    pandas_time = time.perf_counter() - t0
    print(f"  → {pandas_time:.2f}s  |  result shape: {pdf_result.shape}")

    # ── DASK (threads scheduler) ─────────────────────────────────────────────
    print(f"\n  [Dask threads, {npartitions} partitions] Running pipeline …")
    t0 = time.perf_counter()

    ddf = dd.read_parquet(parquet_path, engine="pyarrow")
    ddf = ddf.repartition(npartitions=npartitions)
    ddf["revenue"] = ddf["amount"] * ddf["quantity"]
    dask_result = (
        ddf.groupby(["region", "product_id"])
           .agg({"revenue": "sum", "amount": "mean", "transaction_id": "count"})
           .compute(scheduler="threads")
    )
    dask_time = time.perf_counter() - t0
    speedup = pandas_time / dask_time
    print(f"  → {dask_time:.2f}s  |  result shape: {dask_result.shape}")
    print(f"\n  ✓ Dask speedup: {speedup:.1f}×  (with {n_cores} cores)")
    print(f"    Theoretical max speedup (Amdahl's Law): ~{n_cores:.0f}×")
    print(f"    Overhead (serialisation, scheduling): accounts for the gap")

    return {
        "pandas_s": pandas_time,
        "dask_s":   dask_time,
        "speedup":  speedup,
        "n_cores":  n_cores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DASK DELAYED: PARALLELISING ARBITRARY PYTHON CODE
# ─────────────────────────────────────────────────────────────────────────────
# dd.DataFrame works for tabular pandas-like operations.
# For arbitrary Python functions (e.g., file preprocessing, custom algorithms)
# use dask.delayed.  It wraps any function so that calls return futures.

def demonstrate_dask_delayed():
    """
    Parallelise a list of file-processing tasks with dask.delayed.
    Pattern: embarrassingly-parallel jobs (no data sharing between tasks).
    """
    print("\n" + "="*60)
    print("  DEMO: dask.delayed — Parallelise Arbitrary Functions")
    print("="*60)

    def slow_compute(x: int) -> float:
        """Simulates expensive per-file computation (e.g., preprocessing)."""
        time.sleep(0.1)   # 100 ms per task
        return x ** 2

    inputs = list(range(20))  # 20 tasks × 100 ms = 2.0 s sequential

    # ── SEQUENTIAL ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sequential = [slow_compute(x) for x in inputs]
    seq_time = time.perf_counter() - t0
    print(f"\n  Sequential: {seq_time:.2f}s")

    # ── PARALLEL with dask.delayed ───────────────────────────────────────────
    # @dask.delayed turns slow_compute into a lazy function.
    # Calling it returns a Delayed object (like a promise/future).
    delayed_results = [dask.delayed(slow_compute)(x) for x in inputs]

    # dask.compute() triggers all delayed calls simultaneously across threads
    t0 = time.perf_counter()
    parallel = dask.compute(*delayed_results, scheduler="threads")
    par_time = time.perf_counter() - t0

    print(f"  Parallel   : {par_time:.2f}s  (speedup: {seq_time/par_time:.1f}×)")
    print(f"\n  Results match: {list(sequential) == list(parallel)}")
    print("\n  KEY INSIGHT: dask.delayed works on ANY Python function.")
    print("  Use it to parallelise: file reading, API calls, custom transforms.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PARQUET_PATH = "transactions.parquet"

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            "transactions.parquet not found. "
            "Please run Exercise 1 first to generate the data."
        )

    # 1. Understand lazy evaluation
    explain_dask_concepts()

    # 2. Compare schedulers
    scheduler_results = benchmark_schedulers(PARQUET_PATH)

    # 3. Find optimal chunk size
    chunk_results = benchmark_chunk_sizes(PARQUET_PATH)

    # 4. Demonstrate out-of-core processing
    demonstrate_out_of_core(PARQUET_PATH)

    # 5. pandas vs Dask head-to-head
    comparison = pandas_vs_dask(PARQUET_PATH)

    # 6. dask.delayed for arbitrary functions
    demonstrate_dask_delayed()

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXERCISE 2 SUMMARY")
    print("="*60)
    print(f"""
  On this {comparison['n_cores']}-core machine:

  pandas     : {comparison['pandas_s']:.2f}s  (1 core, all data in RAM)
  Dask        : {comparison['dask_s']:.2f}s  ({comparison['n_cores']} cores, chunked processing)
  Speedup     : {comparison['speedup']:.1f}×

  Scheduler winner   : threads (numpy releases the GIL)
  Optimal partitions : ~{comparison['n_cores'] * 2} ({comparison['n_cores']} cores × 2)

  IMPORTANT: Dask's speedup on a single node is real but limited.
  Exercise 3 shows how to get 10-100× more by distributing across
  the full MeluXina cluster (573 nodes × 64 cores = 36,672 cores).
""")
