"""
================================================================================
EXERCISE 3: Bottleneck Identification in the Training Pipeline
================================================================================
GOAL:
    Systematically find WHERE time is being lost in the training pipeline by
    running controlled experiments and measuring each component independently.

    We isolate and measure:
      1. DataLoader throughput vs number of workers (CPU bottleneck)
      2. GPU utilisation under different data loading configs
      3. CPU→GPU transfer time (pin_memory vs non-pinned)
      4. Generator vs Discriminator forward/backward compute time
      5. The impact of drop_last, prefetching, and batch size on throughput

WHY THIS MATTERS:
    A GPU can only train as fast as it is fed data.
    An underutilised GPU is often caused by:
      - Too few DataLoader workers (CPU bottleneck)
      - Missing pin_memory (slow CPU→GPU transfer)
      - I/O latency (disk reads stalling data loading)
    Fixing these costs zero model changes and can give 2–5× speedups.

WHAT YOU WILL SEE:
    - A sweep over NUM_WORKERS ∈ {0, 1, 2, 4, 8} showing throughput
    - Timing of individual pipeline stages with CUDA events
    - "GPU starvation" visualised as idle time between batches
    - A waterfall breakdown: data load + H2D transfer + forward + backward

DATASET: MNIST
ARCHITECTURE: DCGAN from Exercise 1 (stable baseline for fair comparison)
================================================================================
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LATENT_DIM   = 100
CHANNELS     = 1
BATCH_SIZE   = 128
PROFILE_BATCHES = 100   # Number of batches to use for profiling experiments
                         # (keep small — we run many experiments in sequence)
OUTPUT_DIR   = "ex3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*70}")
print(f"  EXERCISE 3 — Bottleneck Identification in the Training Pipeline")
print(f"{'='*70}")
print(f"\n[DEVICE] {device}")
if device.type == "cuda":
    print(f"[DEVICE] GPU : {torch.cuda.get_device_name(0)}")
    print(f"[DEVICE] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MODEL (DCGAN — same as Exercise 1)
# ─────────────────────────────────────────────────────────────────────────────
# We reuse the baseline architecture. The point of this exercise is the
# pipeline, not the model. Using a fixed model ensures any performance
# differences are due to pipeline changes ONLY.

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(LATENT_DIM, 256*7*7), nn.ReLU(True))
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, CHANNELS, 3, 1, 1), nn.Tanh()
        )
    def forward(self, z):
        return self.conv_blocks(self.project(z).view(z.size(0), 256, 7, 7))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(CHANNELS, 64,  4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,  128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(256*4*4, 1), nn.Sigmoid()
        )
    def forward(self, img): return self.model(img)

def make_models():
    G = Generator().to(device)
    D = Discriminator().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return G, D, optimizer_G, optimizer_D

criterion = nn.BCELoss()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CUDA TIMING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
# CUDA operations are asynchronous — they are queued on the GPU stream.
# time.time() on the CPU will NOT accurately measure GPU operation duration.
# We must use CUDA Events, which are placed directly on the GPU stream
# and measure elapsed time on the device itself.

class CUDATimer:
    """
    Context manager that measures GPU execution time using CUDA events.
    Usage:
        with CUDATimer() as t:
            # GPU work here
        print(t.elapsed_ms)
    """
    def __init__(self, enabled=True):
        self.enabled = enabled and (device.type == "cuda")
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event   = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            torch.cuda.synchronize()   # Wait for GPU to finish
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._cpu_start) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EXPERIMENT A: NUM_WORKERS SWEEP
# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS: With NUM_WORKERS=0, the main process loads data sequentially.
#             GPU waits idle while CPU decodes, transforms, and batches images.
#             More workers → parallel data loading → GPU starved less.
#
# We measure: pure DataLoader throughput (images/sec) WITHOUT GPU training.
#             This isolates the data pipeline from compute.

print(f"\n{'─'*70}")
print(f"  EXPERIMENT A: DataLoader Throughput vs NUM_WORKERS")
print(f"  (Measures: how fast CPU can deliver batches to GPU)")
print(f"  (Isolates: data pipeline from model compute)")
print(f"{'─'*70}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

worker_results = {}
worker_counts  = [0, 1, 2, 4]
# Try 8 workers if CPU has enough cores
import multiprocessing
max_workers = min(8, multiprocessing.cpu_count())
if max_workers > 4:
    worker_counts.append(max_workers)

for nw in worker_counts:
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    # Warm-up: first few batches are slower (worker process startup)
    warmup_batches = min(5, len(loader))
    loader_iter    = iter(loader)
    for _ in range(warmup_batches):
        _ = next(loader_iter)

    # Measure PROFILE_BATCHES batches of pure data loading
    t_start  = time.perf_counter()
    n_images = 0
    for i, (imgs, _) in enumerate(loader):
        if i >= PROFILE_BATCHES:
            break
        # Simulate H2D transfer (the operation that follows data loading)
        imgs = imgs.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()  # wait for transfer to complete
        n_images += imgs.size(0)

    elapsed        = time.perf_counter() - t_start
    images_per_sec = n_images / elapsed

    worker_results[nw] = images_per_sec
    print(f"  workers={nw:2d}  |  {images_per_sec:>10,.0f} img/s  |  "
          f"{elapsed:.2f}s for {PROFILE_BATCHES} batches")

# Find optimal worker count
best_workers = max(worker_results, key=worker_results.get)
print(f"\n  RESULT: Optimal NUM_WORKERS = {best_workers} "
      f"({worker_results[best_workers]:,.0f} img/s)")
speedup = worker_results[best_workers] / max(worker_results[0], 1)
print(f"  Speedup vs workers=0 : {speedup:.2f}×")
print(f"  → Use NUM_WORKERS={best_workers} for the rest of training.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EXPERIMENT B: pin_memory IMPACT
# ─────────────────────────────────────────────────────────────────────────────
# pin_memory=True: tensors are allocated in page-locked (pinned) RAM.
# Pinned memory enables DMA (Direct Memory Access) transfers to GPU,
# bypassing the CPU entirely → faster H2D transfer, especially at scale.
#
# We measure: H2D (Host to Device) transfer time with and without pin_memory.

if device.type == "cuda":
    print(f"\n{'─'*70}")
    print(f"  EXPERIMENT B: pin_memory Impact on CPU→GPU Transfer Time")
    print(f"  (Measures: H2D transfer latency for a single batch)")
    print(f"{'─'*70}")

    for pinned in [False, True]:
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=best_workers,
            pin_memory=pinned,
            drop_last=True
        )

        h2d_times = []
        for i, (imgs, _) in enumerate(loader):
            if i >= PROFILE_BATCHES:
                break
            with CUDATimer() as t:
                imgs_gpu = imgs.to(device, non_blocking=True)
                torch.cuda.synchronize()
            h2d_times.append(t.elapsed_ms)

        avg_h2d = np.mean(h2d_times)
        print(f"  pin_memory={str(pinned):<5}  |  "
              f"avg H2D transfer: {avg_h2d:.3f} ms/batch  |  "
              f"{BATCH_SIZE/avg_h2d*1000:,.0f} img/s transfer rate")

    print(f"  → pin_memory=True reduces transfer latency via DMA.")
    print(f"  → Effect is larger with more workers and larger batches.")
else:
    print(f"\n  [SKIP] Experiment B skipped — no GPU available.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EXPERIMENT C: PIPELINE STAGE TIMING (WATERFALL)
# ─────────────────────────────────────────────────────────────────────────────
# Break one training iteration into its component stages and time each:
#   Stage 1: Data load + H2D transfer  (data pipeline)
#   Stage 2: D forward pass            (compute)
#   Stage 3: D backward pass           (compute)
#   Stage 4: G forward pass            (compute)
#   Stage 5: G backward pass           (compute)
#
# This gives a "waterfall" view of where each second goes.

print(f"\n{'─'*70}")
print(f"  EXPERIMENT C: Pipeline Stage Waterfall")
print(f"  (Measures: time per stage per training iteration)")
print(f"{'─'*70}")

G, D, opt_G, opt_D = make_models()
G.train(); D.train()

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=best_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=True
)

stage_times = {
    "data_load_h2d": [],
    "D_forward":     [],
    "D_backward":    [],
    "G_forward":     [],
    "G_backward":    [],
    "total_iter":    []
}

real_labels = torch.full((BATCH_SIZE, 1), 0.9, device=device)
fake_labels = torch.zeros(BATCH_SIZE, 1, device=device)

progress_bar = tqdm(loader, desc="  Waterfall profiling", leave=True,
                    ncols=80, total=PROFILE_BATCHES)

for batch_idx, (real_imgs, _) in enumerate(progress_bar):
    if batch_idx >= PROFILE_BATCHES:
        break

    iter_start = time.perf_counter()

    # Stage 1: Data loading is implicit in the DataLoader iteration above.
    # We measure H2D transfer here.
    with CUDATimer() as t_h2d:
        real_imgs = real_imgs.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
    stage_times["data_load_h2d"].append(t_h2d.elapsed_ms)

    z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    fake_imgs = G(z).detach()

    # Stage 2: D forward
    opt_D.zero_grad()
    with CUDATimer() as t_df:
        real_preds = D(real_imgs)
        fake_preds = D(fake_imgs)
        loss_D = (criterion(real_preds, real_labels) +
                  criterion(fake_preds, fake_labels)) / 2
    stage_times["D_forward"].append(t_df.elapsed_ms)

    # Stage 3: D backward
    with CUDATimer() as t_db:
        loss_D.backward()
        opt_D.step()
    stage_times["D_backward"].append(t_db.elapsed_ms)

    # Stage 4: G forward
    opt_G.zero_grad()
    z2 = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    with CUDATimer() as t_gf:
        fake_imgs2 = G(z2)
        gen_preds  = D(fake_imgs2)
        loss_G     = criterion(gen_preds, real_labels)
    stage_times["G_forward"].append(t_gf.elapsed_ms)

    # Stage 5: G backward
    with CUDATimer() as t_gb:
        loss_G.backward()
        opt_G.step()
    stage_times["G_backward"].append(t_gb.elapsed_ms)

    if device.type == "cuda":
        torch.cuda.synchronize()
    iter_ms = (time.perf_counter() - iter_start) * 1000
    stage_times["total_iter"].append(iter_ms)

# ── Report waterfall ──────────────────────────────────────────────────────────
print(f"\n  WATERFALL BREAKDOWN (averaged over {PROFILE_BATCHES} batches)")
print(f"  {'Stage':<25} {'Avg (ms)':<12} {'% of iter':<12} {'Min':<10} {'Max'}")
print(f"  {'-'*65}")

total_avg = np.mean(stage_times["total_iter"])
for stage, times in stage_times.items():
    if stage == "total_iter":
        continue
    avg  = np.mean(times)
    pct  = avg / total_avg * 100
    mn   = np.min(times)
    mx   = np.max(times)
    flag = " ◄ BOTTLENECK" if pct > 30 else ""
    print(f"  {stage:<25} {avg:<12.3f} {pct:<12.1f} {mn:<10.3f} {mx:.3f}{flag}")

print(f"  {'─'*65}")
print(f"  {'Total iteration':<25} {total_avg:<12.3f} {'100.0':<12}")

bottleneck = max(
    [(s, np.mean(t)) for s, t in stage_times.items() if s != "total_iter"],
    key=lambda x: x[1]
)
print(f"\n  IDENTIFIED BOTTLENECK: '{bottleneck[0]}' ({bottleneck[1]:.3f} ms avg)")
print(f"  → This is the stage to optimise first.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EXPERIMENT D: GPU STARVATION VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
# GPU "starvation" = time the GPU sits idle waiting for the next batch.
# We measure the gap between end of one batch's GPU work and the moment
# the next batch's data is ready on GPU.
# With NUM_WORKERS=0 (baseline), this gap is large.
# With NUM_WORKERS=best_workers + pin_memory, this gap shrinks.

print(f"\n{'─'*70}")
print(f"  EXPERIMENT D: GPU Starvation — workers=0 vs workers={best_workers}")
print(f"  (Measures: idle GPU time between batches)")
print(f"{'─'*70}")

for test_workers in [0, best_workers]:
    test_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=test_workers,
        pin_memory=(device.type == "cuda" and test_workers > 0),
        drop_last=True
    )

    starvation_times = []
    prev_end = None

    for i, (imgs, _) in enumerate(test_loader):
        if i >= PROFILE_BATCHES:
            break

        # Record when GPU work starts for this batch
        now = time.perf_counter()
        if prev_end is not None:
            # Gap between GPU finishing last batch and being ready for this one
            starvation_ms = (now - prev_end) * 1000
            starvation_times.append(max(starvation_ms, 0))

        # Simulate one GPU operation
        imgs_gpu = imgs.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        prev_end = time.perf_counter()

    avg_starvation = np.mean(starvation_times) if starvation_times else 0
    print(f"  workers={test_workers:2d}  |  avg GPU starvation: {avg_starvation:.2f} ms/batch  |  "
          f"total idle: {avg_starvation * PROFILE_BATCHES / 1000:.2f}s over {PROFILE_BATCHES} batches")

print(f"  → Reducing starvation directly increases GPU utilisation.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FULL TRAINING RUN (OPTIMISED PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────
# Apply the findings from Experiments A–D: best workers + pin_memory.
# Run 3 epochs and compare throughput to Exercise 1.

print(f"\n{'─'*70}")
print(f"  FULL TRAINING: Optimised Pipeline (applying Ex3 findings)")
print(f"  workers={best_workers}, pin_memory=True, drop_last=True")
print(f"{'─'*70}\n")

NUM_EPOCHS  = 3
LR          = 2e-4
G, D, opt_G, opt_D = make_models()
G.train(); D.train()

optimised_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=best_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
    prefetch_factor=2 if best_workers > 0 else None,
    # prefetch_factor=2 means each worker pre-fetches 2 batches ahead
    # → next batch is ready before current batch processing finishes
    persistent_workers=(best_workers > 0)
    # persistent_workers=True: workers are kept alive between epochs
    # → avoids worker process restart cost at start of each epoch
)

history = {"epoch_time": [], "images_per_sec": [], "loss_D": [], "loss_G": []}

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    run_D, run_G = 0.0, 0.0
    n = 0

    pbar = tqdm(
        optimised_loader,
        desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]",
        leave=True,
        ncols=110
    )

    for real_imgs, _ in pbar:
        bs        = real_imgs.size(0)
        real_imgs = real_imgs.to(device, non_blocking=True)

        real_lbl = torch.full((bs, 1), 0.9, device=device)
        fake_lbl = torch.zeros(bs, 1, device=device)

        # Train D
        opt_D.zero_grad()
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake = G(z).detach()
        loss_D = (criterion(D(real_imgs), real_lbl) + criterion(D(fake), fake_lbl)) / 2
        loss_D.backward(); opt_D.step()

        # Train G
        opt_G.zero_grad()
        z = torch.randn(bs, LATENT_DIM, device=device)
        loss_G = criterion(D(G(z)), real_lbl)
        loss_G.backward(); opt_G.step()

        run_D += loss_D.item()
        run_G += loss_G.item()
        n     += 1

        pbar.set_postfix({
            "D_loss": f"{loss_D.item():.4f}",
            "G_loss": f"{loss_G.item():.4f}",
        })

    epoch_time = time.time() - epoch_start
    imgs_sec   = len(dataset) / epoch_time
    history["epoch_time"].append(epoch_time)
    history["images_per_sec"].append(imgs_sec)
    history["loss_D"].append(run_D / n)
    history["loss_G"].append(run_G / n)

    print(f"\n  Epoch {epoch}/{NUM_EPOCHS}: {epoch_time:.2f}s | "
          f"{imgs_sec:,.0f} img/s | D_loss={run_D/n:.4f} | G_loss={run_G/n:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  EXERCISE 3 — BOTTLENECK IDENTIFICATION REPORT")
print(f"{'='*70}")
print(f"\n  Data Pipeline Optimisation Results:")
print(f"  Best NUM_WORKERS   : {best_workers}")
print(f"  Throughput improvement vs workers=0 : {speedup:.2f}×")
print(f"\n  Stage Breakdown (top bottleneck):")
print(f"  → {bottleneck[0]} takes {bottleneck[1]:.3f} ms ({bottleneck[1]/total_avg*100:.1f}% of iteration)")
print(f"\n  Optimised Pipeline Performance:")
print(f"  Avg time/epoch  : {np.mean(history['epoch_time']):.2f}s")
print(f"  Avg throughput  : {np.mean(history['images_per_sec']):,.0f} img/s")
print(f"\n  KEY LESSONS:")
print(f"  1. NUM_WORKERS matters enormously — even 0→2 gives major gains")
print(f"  2. pin_memory reduces H2D transfer latency via DMA")
print(f"  3. persistent_workers avoids process restart cost per epoch")
print(f"  4. prefetch_factor overlaps data loading with GPU compute")
print(f"  5. Always profile BEFORE optimising — surprises are common")
print(f"\n  NEXT STEP → Exercise 4: Apply AMP (Mixed Precision) to reduce")
print(f"  compute time per batch (the compute stages in your waterfall).")
print(f"{'='*70}\n")
