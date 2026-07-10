"""
=============================================================================
EXERCISE 3: Running Your Analytics on a GPU with RAPIDS cuDF
=============================================================================

WHAT WE ARE DOING 
  In Exercise 1 you found the bottleneck was I/O, and fixed it with Parquet.
  In Exercise 2 you used Dask to spread the work across all the CPU cores
  instead of just one.

  In this exercise we take the SAME data and the SAME pipeline and run it on a
  GPU. A GPU has thousands of small cores and very fast memory, which makes it
  extremely good at exactly the operations analytics does all day: grouping,
  joining, and aggregating lots of rows.

you do NOT have to rewrite your code. A library called
 cuDF can run your existing pandas code on the GPU 


BEFORE YOU START
  - You must have completed Exercise 1, so that the file "transactions.parquet"
    already exists in this folder.
  - You must be on a GPU node. On iris-cluster, request a GPU node and load RAPIDS,
    for example:
        module load RAPIDS
    (or activate the provided conda environment: conda activate rapids)
  - Check the GPU is visible by running:  nvidia-smi
=============================================================================
"""

import os
import time
import pandas as pd          

PARQUET_PATH = "transactions.parquet"


# =============================================================================
# PART 1 — RUN EXERCISE 1 ON THE GPU (ZERO CODE CHANGES)
# =============================================================================
#
# You do not need to edit Exercise 1 at all. You just launch it a different way.
#
# STEP 1. Run Exercise 1 the normal way (on the CPU) and note the total time:
#
#         python exercise1.py
#
# STEP 2. Now run the SAME file on the GPU by adding "-m cudf.pandas":
#
#         python -m cudf.pandas exercise1.py
#
#         This tells cuDF: "whenever this code calls pandas, try to run it on
#         the GPU instead." If the GPU can do an operation, it does. If it
#         cannot, it quietly runs that part on the CPU as usual.
#
# STEP 3. Compare the two total times. Same code, same results — different
#         hardware.
#
# (If you are in a Jupyter notebook instead of a script, you get the same
#  behaviour by putting this line at the very top, BEFORE importing pandas:
#         %load_ext cudf.pandas
#  )


# =============================================================================
# PART 2 — BENCHMARK: THE SAME PIPELINE ON CPU vs GPU
# =============================================================================
#
# In Part 1 you ran the whole script twice. Here we measure each step so you
# can see WHERE the GPU helps most.
#
# Because cuDF copies the pandas API, the very same pipeline function runs on
# both. We simply hand it either "pandas" (CPU) or "cudf" (GPU).

def run_pipeline(frame_lib, parquet_path=PARQUET_PATH):
    """
    The analytics pipeline from earlier exercises.
    Pass frame_lib = pandas to run on the CPU, or cudf to run on the GPU.
    Returns the time taken by each step.
    """
    times = {}

    # Step A: read the data from the Parquet file
    t0 = time.perf_counter()
    df = frame_lib.read_parquet(parquet_path)
    times["A_load"] = time.perf_counter() - t0

    # Step B: make a new column, revenue = amount * quantity
    t0 = time.perf_counter()
    df["revenue"] = df["amount"] * df["quantity"]
    times["B_new_column"] = time.perf_counter() - t0

    # Step C: group by region and product, and summarise
    #         (this is heavy work — watch how the GPU does here)
    t0 = time.perf_counter()
    summary = (
        df.groupby(["region", "product_id"])
          .agg(total_revenue=("revenue", "sum"),
               avg_amount=("amount", "mean"),
               n_txn=("transaction_id", "count"))
          .reset_index()
    )
    times["C_groupby"] = time.perf_counter() - t0

    # Step D: build a per-customer total, then join it back onto every row
    #         (also heavy — joins are something GPUs are very good at)
    t0 = time.perf_counter()
    per_customer = (
        df.groupby("customer_id")
          .agg(customer_lifetime_value=("revenue", "sum"))
          .reset_index()
    )
    df = df.merge(per_customer, on="customer_id", how="left")
    times["D_join"] = time.perf_counter() - t0

    # Step E: keep only the top 10% highest-revenue transactions
    t0 = time.perf_counter()
    threshold = df["revenue"].quantile(0.90)
    high_value = df[df["revenue"] > threshold]
    times["E_filter"] = time.perf_counter() - t0

    times["_rows_kept"] = int(len(high_value))
    return times


def run_benchmark(parquet_path=PARQUET_PATH):
    print("\n" + "=" * 58)
    print("  BENCHMARK: same pipeline on CPU (pandas) and GPU (cuDF)")
    print("=" * 58)

    # --- CPU run ---
    print("\n  Running on the CPU with pandas ...")
    cpu = run_pipeline(pd, parquet_path)
    cpu_total = sum(v for k, v in cpu.items() if not k.startswith("_"))
    print(f"  CPU total: {cpu_total:.2f} s")

    # --- GPU run ---
    try:
        import cudf
    except ImportError:
        print("\n  Could not import cudf — are you on a GPU node with RAPIDS?")
        print("  Try:  module load RAPIDS   (then re-run this script)")
        return

    # The very first GPU operation has to start up the GPU, which takes a
    # moment. We do one tiny throwaway operation first so the timing below
    # measures real work, not the one-time startup. (This is a fair comparison.)
    print("\n  Warming up the GPU ...")
    _ = cudf.Series([0, 1, 2]).sum()

    print("  Running on the GPU with cuDF ...")
    gpu = run_pipeline(cudf, parquet_path)
    gpu_total = sum(v for k, v in gpu.items() if not k.startswith("_"))
    print(f"  GPU total: {gpu_total:.2f} s")

    # --- Results table ---
    print("\n  Time per step (seconds):")
    print(f"  {'step':<14}{'CPU':>10}{'GPU':>10}{'GPU speedup':>14}")
    print("  " + "-" * 48)
    for step in ["A_load", "B_new_column", "C_groupby", "D_join", "E_filter"]:
        c, g = cpu[step], gpu[step]
        speedup = c / g if g > 0 else float("inf")
        print(f"  {step:<14}{c:>10.3f}{g:>10.3f}{speedup:>12.1f}x")
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<14}{cpu_total:>10.3f}{gpu_total:>10.3f}"
          f"{cpu_total / gpu_total:>12.1f}x")

    print(f"\n  The GPU was about {cpu_total / gpu_total:.1f}x faster overall.")
    print("  Look at which steps gained the most — usually groupby and join,")
    print("  because those touch a lot of data and the GPU handles that well.")


# =============================================================================
# PART 3 — WHEN IS A GPU WORTH IT?  (think about this)
# =============================================================================
#
# A GPU is powerful, but it is not always the right choice. The same "does the
# speedup beat the overhead?" question you saw with Dask applies here too.
#
#   A GPU helps when:
#     - The data fits in the GPU's memory (an A100 has 40 or 80 GB).
#     - You do heavy work: big groupby, joins, aggregations over many rows.
#     - There is enough work to be worth sending the data to the GPU first.
#
#   A GPU does NOT help (and can be slower) when:
#     - The data is tiny. Sending it to the GPU costs more than you save.
#     - The data is far bigger than GPU memory (you would need multiple GPUs).
#     - You do lots of row-by-row Python logic instead of whole-column
#       operations.
#
#   The mental picture:
#     Getting your data onto the GPU costs a little time up front, but then
#     every operation is very cheap. So the GPU wins when there is enough work
#     to make that up-front cost worth it.
#
#   Where this goes next (you do not run this today):
#     - dask-cudf combines Dask AND cuDF to use SEVERAL GPUs, for data too big
#       for one GPU. That completes the picture:
#           1 CPU core  ->  many CPU cores (Dask)  ->  1 GPU (cuDF)  ->  many GPUs


# =============================================================================
# MAIN — run the benchmark
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            "transactions.parquet was not found.\n"
            "Please run Exercise 1 first — it creates this file."
        )

    run_benchmark(PARQUET_PATH)

    print("\n" + "=" * 58)
    print("  WHAT TO REMEMBER")
    print("=" * 58)
    print("""
  1. You ran your Exercise 1 code on a GPU with NO code changes, just:
         python -m cudf.pandas exercise1.py

  2. The GPU helped most on the heavy steps (groupby and join), because
     those move a lot of data — which is what GPUs are built for.

  3. A GPU is the right tool only when the data fits in GPU memory AND there
     is enough work to make it worthwhile. Otherwise the CPU is fine.

  4. The full progression across the three exercises:
         pandas (1 core)  ->  Dask (all cores)  ->  cuDF (GPU)
""")
