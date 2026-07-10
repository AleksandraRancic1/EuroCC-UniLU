"""
=========================================================================
EXERCISE 1: Profiling and Baseline Performance
=============================================================================
OBJECTIVE: Understand performance bottlenecks in a data analytics pipeline.

In this exercise we:
  1. Generate a realistic synthetic dataset (~5GB in CSV)
  2. Run a pandas-based pipeline and measure its performance
  3. Profile CPU, memory, and I/O bottlenecks
  4. Convert data to Parquet and measure the improvement
  5. Compare and document all results in a performance report

==========================================================================
"""

import os
import time
import tracemalloc       # Standard library: tracks memory allocations
import cProfile          # Standard library: CPU profiler
import pstats            # Standard library: reads cProfile output
import io
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# SECTION 1 — SYNTHETIC DATA GENERATION
# 
# Why synthetic data?  On an HPC cluster you would use real data sitting on a
# Lustre/GPFS parallel file system.  For this exercise we generate a dataset
# that has the same statistical shape as a real-world transactions table so
# that the bottlenecks we observe are representative.

def generate_dataset(n_rows: int = 50_000_000, csv_path: str = "transactions.csv"):
    """
    Generate a synthetic transactions CSV (~500 MB-1 GB depending on n_rows).

    For a true 5 GB file set n_rows=50_000_000.  We keep it smaller here so
    the exercise finishes in reasonable classroom time.

    Schema:
      transaction_id   – unique integer identifier
      customer_id      – foreign-key to customers (high cardinality: ~100 k)
      product_id       – foreign-key to products  (lower cardinality: ~10 k)
      region           – categorical with 8 values (typical analytics dimension)
      amount           – float, right-skewed (realistic sales distribution)
      quantity         – integer 1-10
      timestamp        – datetime in 2023
    """
    print(f"\n[DATA GEN] Generating {n_rows:,} rows → {csv_path}")
    rng = np.random.default_rng(seed=42)   # reproducible results

    df = pd.DataFrame({
        "transaction_id": np.arange(n_rows),
        "customer_id":    rng.integers(0, 100_000, n_rows),
        "product_id":     rng.integers(0, 10_000,  n_rows),
        "region":         rng.choice(
            ["North", "South", "East", "West", "NE", "NW", "SE", "SW"],
            size=n_rows
        ),
        # Lognormal gives a realistic right-skewed price distribution
        "amount":   np.round(rng.lognormal(mean=4.0, sigma=1.2, size=n_rows), 2),
        "quantity": rng.integers(1, 11, n_rows),
        "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="1s"),
    })

    start = time.perf_counter()
    df.to_csv(csv_path, index=False)
    elapsed = time.perf_counter() - start

    size_mb = os.path.getsize(csv_path) / 1_048_576
    print(f"[DATA GEN] Written {size_mb:.1f} MB in {elapsed:.1f}s")
    return csv_path


# 
# SECTION 2 — THE ANALYTICS PIPELINE (pandas, single-threaded)
# 
# This is the kind of code that uses pandas defaults which are single-threaded and keep
# the entire dataset in RAM.

def run_analytics_pipeline_csv(csv_path: str) -> dict:
    """
    A representative analytics pipeline on the CSV:
      Step A – Load CSV into a pandas DataFrame
      Step B – Cast types (timestamp parsing is expensive)
      Step C – Compute derived columns (revenue = amount × quantity)
      Step D – GroupBy aggregation by region and product
      Step E – Join: attach an aggregated customer-level feature back to rows
      Step F – Filter for high-value transactions

    Returns a dict with timing and memory stats for each step.
    """
    metrics = {}

    # STEP A: Load CSV 
    # pd.read_csv is single-threaded.  For a 5 GB file this can take 60-120 s
    # on a fast SSD because CSV is uncompressed, untyped text that must be
    # parsed character-by-character.
    print("\n[PIPELINE] Step A – Loading CSV …")
    tracemalloc.start()
    t0 = time.perf_counter()

    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],   # forces datetime parsing – expensive!
        dtype={
            "transaction_id": "int32",
            "customer_id":    "int32",
            "product_id":     "int16",
            "region":         "category",  # category dtype saves RAM vs object
            "quantity":       "int8",
        }
    )

    metrics["A_load_csv_s"]        = time.perf_counter() - t0
    metrics["A_peak_mem_mb"]       = tracemalloc.get_traced_memory()[1] / 1_048_576
    tracemalloc.stop()
    print(f"  → {metrics['A_load_csv_s']:.2f}s  |  peak RAM {metrics['A_peak_mem_mb']:.0f} MB")
    print(f"  → DataFrame shape: {df.shape}  |  dtypes:\n{df.dtypes.to_string()}")

    #  STEP B: Derived column
    # Simple vectorised operation – very fast in pandas 
    print("\n[PIPELINE] Step B – Derived column (revenue) …")
    t0 = time.perf_counter()
    df["revenue"] = df["amount"] * df["quantity"]
    metrics["B_derived_col_s"] = time.perf_counter() - t0
    print(f"  → {metrics['B_derived_col_s']:.4f}s")

    # STEP C: GroupBy aggregation 
    # GroupBy forces a full scan + hash-grouping.  For many groups (high
    # cardinality) this is memory-intensive
    print("\n[PIPELINE] Step C – GroupBy region × product_id …")
    tracemalloc.start()
    t0 = time.perf_counter()

    region_summary = (
        df.groupby(["region", "product_id"], observed=True)
          .agg(
              total_revenue = ("revenue",  "sum"),
              avg_amount    = ("amount",   "mean"),
              n_txn         = ("transaction_id", "count"),
          )
          .reset_index()
    )

    metrics["C_groupby_s"]      = time.perf_counter() - t0
    metrics["C_peak_mem_mb"]    = tracemalloc.get_traced_memory()[1] / 1_048_576
    tracemalloc.stop()
    print(f"  → {metrics['C_groupby_s']:.2f}s  |  peak RAM {metrics['C_peak_mem_mb']:.0f} MB")
    print(f"  → Result shape: {region_summary.shape}")

    # STEP D: Join (merge) 
    # Compute a customer-level aggregate, then join it back.
    # This "self-join" pattern is common in feature engineering.
    # The merge triggers a sort or hash-join internally.
    print("\n[PIPELINE] Step D – Customer-level join …")
    tracemalloc.start()
    t0 = time.perf_counter()

    customer_stats = (
        df.groupby("customer_id")
          .agg(customer_lifetime_value=("revenue", "sum"))
          .reset_index()
    )
    df = df.merge(customer_stats, on="customer_id", how="left")

    metrics["D_join_s"]       = time.perf_counter() - t0
    metrics["D_peak_mem_mb"]  = tracemalloc.get_traced_memory()[1] / 1_048_576
    tracemalloc.stop()
    print(f"  → {metrics['D_join_s']:.2f}s  |  peak RAM {metrics['D_peak_mem_mb']:.0f} MB")

    # STEP E: Filter
    print("\n[PIPELINE] Step E – Filter high-value transactions …")
    t0 = time.perf_counter()
    threshold = df["revenue"].quantile(0.90)
    high_value = df[df["revenue"] > threshold]
    metrics["E_filter_s"] = time.perf_counter() - t0
    print(f"  → {metrics['E_filter_s']:.4f}s  |  {len(high_value):,} rows kept (top 10 %)")

    return metrics, df


# 
# SECTION 3 — CPU PROFILING with cProfile
#
# cProfile instruments every Python function call and records cumulative time.
# It answers "WHERE is the time being spent?" at the function level.
# Limitation: it cannot see inside C-extension calls (numpy internals).

def profile_pipeline(csv_path: str, profile_output: str = "pipeline.prof"):
    """
    Wrap run_analytics_pipeline_csv in cProfile and save the profile data.
    After the exercise, inspect it with:
        python -m snakeviz pipeline.prof
    or with pstats in a REPL:
        import pstats; p = pstats.Stats('pipeline.prof'); p.sort_stats('cumulative'); p.print_stats(20)
    """
    print("\n[PROFILE] Running cProfile …  (this adds ~10-15 % overhead)")
    profiler = cProfile.Profile()
    profiler.enable()
    metrics, df = run_analytics_pipeline_csv(csv_path)
    profiler.disable()

    profiler.dump_stats(profile_output)

    # Print top-20 hotspots sorted by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print("\n[PROFILE] Top-20 cumulative time hotspots:")
    print(stream.getvalue())

    return metrics, df


# 
# SECTION 4 — CONVERT TO PARQUET AND BENCHMARK
# 
# Parquet is a columnar binary format.  Key advantages for analytics:
#   • Columnar storage  → queries that touch only a few columns skip irrelevant
#     data entirely (predicate/projection pushdown)
#   • Snappy/ZSTD compression  → file is 5-10× smaller than CSV on disk
#   • Typed schema  → no parsing overhead; int32 is stored as int32 on disk
#   • Row-group statistics  → can skip entire blocks whose min/max
#     do not satisfy a filter predicate ("predicate pushdown")

def convert_to_parquet(df: pd.DataFrame, parquet_path: str = "transactions.parquet"):
    """
    Write the DataFrame to Parquet using the PyArrow engine with Snappy compression.
    """
    print(f"\n[PARQUET] Writing {parquet_path} …")
    t0 = time.perf_counter()

    # row_group_size controls how many rows are stored per block inside the
    # Parquet file.  Larger groups = better compression; smaller groups =
    # faster predicate pushdown for selective queries.  500k is a good default.
    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="snappy",    # fast compress/decompress; good for analytics
        index=False,
        row_group_size=500_000,
    )

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(parquet_path) / 1_048_576
    print(f"  → Written {size_mb:.1f} MB in {elapsed:.1f}s")
    return parquet_path


def run_analytics_pipeline_parquet(parquet_path: str) -> dict:
    """
    Identical analytics operations but reading from Parquet.
    We pass 'columns=' to demonstrate projection pushdown – only columns we
    actually need are read from disk, so I/O is dramatically reduced.
    """
    metrics = {}

    print("\n[PARQUET PIPELINE] Step A – Loading Parquet (all columns) …")
    tracemalloc.start()
    t0 = time.perf_counter()

    df = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
        # Demonstrating projection pushdown: only read columns we need
        columns=["transaction_id", "customer_id", "product_id",
                 "region", "amount", "quantity", "timestamp"],
    )

    metrics["A_load_parquet_s"]  = time.perf_counter() - t0
    metrics["A_peak_mem_mb"]     = tracemalloc.get_traced_memory()[1] / 1_048_576
    tracemalloc.stop()
    print(f"  → {metrics['A_load_parquet_s']:.2f}s  |  peak RAM {metrics['A_peak_mem_mb']:.0f} MB")

    # Remaining steps are identical to the CSV pipeline
    print("\n[PARQUET PIPELINE] Steps B-E (derived col, groupby, join, filter) …")
    t0 = time.perf_counter()
    df["revenue"] = df["amount"] * df["quantity"]
    customer_stats = (
        df.groupby("customer_id")
          .agg(customer_lifetime_value=("revenue", "sum"))
          .reset_index()
    )
    df = df.merge(customer_stats, on="customer_id", how="left")
    high_value = df[df["revenue"] > df["revenue"].quantile(0.90)]
    metrics["BCDE_s"] = time.perf_counter() - t0
    print(f"  → {metrics['BCDE_s']:.2f}s  |  {len(high_value):,} rows kept (top 10 %)")

    return metrics


# 
# SECTION 5 — REPORT
# 
def print_report(csv_metrics: dict, parquet_metrics: dict,
                 csv_path: str, parquet_path: str):
    csv_size_mb     = os.path.getsize(csv_path)     / 1_048_576
    parquet_size_mb = os.path.getsize(parquet_path) / 1_048_576

    csv_total  = sum(v for k, v in csv_metrics.items()     if k.endswith("_s"))
    parq_total = (parquet_metrics["A_load_parquet_s"] +
                  parquet_metrics["BCDE_s"])

    print("\n" + "="*60)
    print("  PERFORMANCE REPORT — Exercise 1")
    print("="*60)
    print(f"\n  File sizes:")
    print(f"    CSV     : {csv_size_mb:>8.1f} MB")
    print(f"    Parquet : {parquet_size_mb:>8.1f} MB  ({csv_size_mb/parquet_size_mb:.1f}× smaller)")

    print(f"\n  Pipeline timing (CSV):")
    for k, v in csv_metrics.items():
        if k.endswith("_s"):
            print(f"    {k:<25} {v:>7.2f} s")
    print(f"    {'TOTAL':<25} {csv_total:>7.2f} s")

    print(f"\n  Pipeline timing (Parquet):")
    for k, v in parquet_metrics.items():
        if k.endswith("_s"):
            print(f"    {k:<25} {v:>7.2f} s")
    print(f"    {'TOTAL':<25} {parq_total:>7.2f} s")

    print(f"\n  Speedup  (Parquet vs CSV): {csv_total/parq_total:.1f}×")
    print("="*60)

    # KEY TAKEAWAYS 
    print("""
  KEY TAKEAWAYS:
  1. The single biggest bottleneck is almost always I/O — not CPU.
     CSV forces the OS to read every byte sequentially; Parquet lets the
     reader skip columns and row-groups it doesn't need.

  2. Memory usage can spike 3-5× the file size during operations like merge.
     On a node with 512 GB RAM this is fine; on a laptop it causes swapping.

  3. cProfile reveals that pandas' C extensions dominate runtime.
     To go faster we need true parallelism (Exercise 2) not optimised loops.

  4. Switching to Parquet is the cheapest performance win available.
     It costs zero code changes in pandas and often gives 3-10× I/O speedup.
""")


# 
# MAIN
# 
if __name__ == "__main__":
    CSV_PATH     = "transactions.csv"
    PARQUET_PATH = "transactions.parquet"
    N_ROWS       = 50_000_000   # ← increase to 50_000_000 for a true 5 GB file

    # Step 1: Generate data (skip if the file already exists)
    if not os.path.exists(CSV_PATH):
        generate_dataset(n_rows=N_ROWS, csv_path=CSV_PATH)
    else:
        print(f"[DATA GEN] {CSV_PATH} already exists – skipping generation.")

    # Step 2: Profile the CSV pipeline
    csv_metrics, df = profile_pipeline(CSV_PATH)

    # Step 3: Convert to Parquet
    convert_to_parquet(df, PARQUET_PATH)

    # Step 4: Run the Parquet pipeline
    parquet_metrics = run_analytics_pipeline_parquet(PARQUET_PATH)

    # Step 5: Print the comparison report
    print_report(csv_metrics, parquet_metrics, CSV_PATH, PARQUET_PATH)
