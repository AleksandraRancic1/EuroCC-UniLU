"""
================================================================================
EXERCISE 1: Baseline GAN — Profiling and Measurement
================================================================================
GOAL:
    Build a working DCGAN-style baseline on MNIST and establish a performance
    baseline BEFORE any optimization. We measure:
      - Time per epoch
      - Images per second (throughput)
      - GPU utilization and memory footprint
      - Generator and Discriminator loss curves

WHY THIS MATTERS:
    "You cannot optimize what you have not measured."
    Every optimization decision in the rest of the training will be justified
    by comparing against this baseline. Speed without a reference is meaningless.

WHAT YOU WILL OBSERVE:
    - How long a single epoch takes on 1 GPU (or CPU)
    - Whether the GPU is actually busy (utilization), or starved (I/O bottleneck)
    - How much VRAM the two networks consume
    - Whether generator and discriminator losses are in reasonable balance

ARCHITECTURE CHOICE (DCGAN on MNIST):
    Generator:   Latent vector z (100-d) → Conv-Transpose layers → 1×28×28 image
    Discriminator: 1×28×28 image → Conv layers → scalar probability
    This is competitive enough to generate sharp MNIST digits.

DATASET: MNIST (60,000 training images, 28×28 grayscale)
================================================================================
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm          # Progress bars — essential for monitoring
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# Everything is centralised here so trainees can experiment easily.

LATENT_DIM   = 100        # Dimension of the noise vector z fed to the generator
IMAGE_SIZE   = 28         # MNIST images are 28×28
CHANNELS     = 1          # Grayscale
BATCH_SIZE   = 128        # Larger batch → more stable gradients; costs more VRAM
NUM_EPOCHS   = 5          # Enough to see learning; keep short for a live demo
LR_G         = 2e-4       # Generator learning rate (Adam)
LR_D         = 2e-4       # Discriminator learning rate (Adam)
BETA1        = 0.5        # Adam β₁ — lower than default; standard for GANs
BETA2        = 0.999      # Adam β₂
NUM_WORKERS  = 4          # DataLoader worker processes (CPU data loading threads)
SAMPLE_EVERY = 1          # Save generated images every N epochs
OUTPUT_DIR   = "ex1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*70}")
print(f"  EXERCISE 1 — Baseline GAN Profiling")
print(f"{'='*70}")
print(f"\n[DEVICE] Using: {device}")
if device.type == "cuda":
    print(f"[DEVICE] GPU name       : {torch.cuda.get_device_name(0)}")
    print(f"[DEVICE] Total VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[DEVICE] WARNING — No GPU found. Training will be significantly slower.")
    print("[DEVICE] This is your baseline; with a GPU the speedup will be dramatic.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATASET AND DATALOADER
# ─────────────────────────────────────────────────────────────────────────────
# We normalize pixel values from [0,255] → [-1,1] because the generator's final
# activation is Tanh (output range [-1,1]).  Matching normalizations is critical.

print(f"\n[DATA] Loading MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),                         # PIL image → Tensor [0,1]
    transforms.Normalize([0.5], [0.5])             # [0,1] → [-1,1]  (mean=0.5, std=0.5)
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# num_workers: how many CPU processes pre-load batches in parallel.
# If num_workers=0, the main process does all loading → GPU waits → low utilization.
# We will experiment with this in Exercise 3.
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),   # pin_memory: keeps tensors in page-locked
                                          # RAM → faster CPU→GPU transfer (DMA)
    drop_last=True                        # drop_last: ensures every batch is exactly
                                          # BATCH_SIZE (avoids batch-norm issues)
)

print(f"[DATA] Dataset size     : {len(dataset):,} images")
print(f"[DATA] Batch size       : {BATCH_SIZE}")
print(f"[DATA] Batches per epoch: {len(dataloader)}")
print(f"[DATA] DataLoader workers: {NUM_WORKERS}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GENERATOR ARCHITECTURE (DCGAN-style)
# ─────────────────────────────────────────────────────────────────────────────
# The Generator maps a random latent vector z ∈ R^100 → image ∈ R^(1×28×28).
#
# ConvTranspose2d (transposed convolution / "deconvolution"):
#   Learned upsampling — the inverse spatial operation of Conv2d.
#   Doubles spatial dimensions at each step.
#
# BatchNorm2d:
#   Stabilises training by normalising layer inputs.
#   Used after every ConvTranspose except the output layer.
#
# ReLU (hidden layers) + Tanh (output):
#   Tanh maps outputs to [-1,1], matching our normalised data.

class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Project and reshape: (N, latent_dim) → (N, 256, 7, 7)
        # 7×7 is our starting spatial resolution; we will upsample twice.
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        self.conv_blocks = nn.Sequential(
            # Block 1: (N, 256, 7, 7) → (N, 128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2: (N, 128, 14, 14) → (N, 64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer: (N, 64, 28, 28) → (N, 1, 28, 28)
            nn.Conv2d(64, CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()          # Output in [-1, 1] — must match data normalisation
        )

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        x = self.project(z)           # → (batch, 256*7*7)
        x = x.view(x.size(0), 256, 7, 7)  # → (batch, 256, 7, 7)
        return self.conv_blocks(x)    # → (batch, 1, 28, 28)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DISCRIMINATOR ARCHITECTURE (DCGAN-style)
# ─────────────────────────────────────────────────────────────────────────────
# The Discriminator maps an image ∈ R^(1×28×28) → scalar ∈ [0,1].
# 0 = "probably fake", 1 = "probably real".
#
# LeakyReLU (slope=0.2):
#   Preferred over ReLU in discriminators — avoids dying neurons, passes
#   small gradients even for negative activations. Keeps discriminator healthy.
#
# No BatchNorm in first layer: standard DCGAN practice.
# Spectral normalisation would be the next upgrade (see Exercise 2).

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Block 1: (N, 1, 28, 28) → (N, 64, 14, 14)
            nn.Conv2d(CHANNELS, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),   # No BatchNorm in first layer

            # Block 2: (N, 64, 14, 14) → (N, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (N, 128, 7, 7) → (N, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Flatten and classify
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()     # Sigmoid squashes to (0,1) — interpreted as P(real)
        )

    def forward(self, img):
        features = self.model(img)
        return self.classifier(features)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — WEIGHT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
# DCGAN paper recommends initialising Conv and BatchNorm weights from
# N(0, 0.02). This prevents exploding/vanishing gradients at step 0.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)   # γ initialised near 1
        nn.init.constant_(m.bias.data, 0)             # β initialised at 0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MODEL INSTANTIATION
# ─────────────────────────────────────────────────────────────────────────────

G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

# Count parameters — important for understanding memory cost.
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[MODEL] Generator parameters    : {count_params(G):,}")
print(f"[MODEL] Discriminator parameters: {count_params(D):,}")
print(f"[MODEL] Total parameters        : {count_params(G) + count_params(D):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — LOSS AND OPTIMISERS
# ─────────────────────────────────────────────────────────────────────────────
# Binary Cross-Entropy loss: standard GAN objective.
#   D tries to output 1 for real, 0 for fake.
#   G tries to make D output 1 for its fakes (i.e., fool D).

criterion = nn.BCELoss()

# Separate optimisers for G and D — they update independently!
optimizer_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
optimizer_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

# Fixed noise: reuse the same z across epochs to visualise G's progress.
fixed_noise = torch.randn(64, LATENT_DIM, device=device)

print(f"\n[OPTIM] Loss function : BCELoss")
print(f"[OPTIM] Optimizer G   : Adam  lr={LR_G}  β=({BETA1},{BETA2})")
print(f"[OPTIM] Optimizer D   : Adam  lr={LR_D}  β=({BETA1},{BETA2})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PROFILING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_gpu_memory_mb():
    """Return current GPU memory allocated in MB (0 if no GPU)."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e6
    return 0.0

def get_gpu_memory_reserved_mb():
    """Return GPU memory reserved by PyTorch allocator in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device) / 1e6
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
# This is the core GAN training loop. For each batch:
#   Step A — Train Discriminator:
#     1. Run real images → D → real_loss  (want D(real) → 1)
#     2. Run fake images → D → fake_loss  (want D(G(z)) → 0)
#     3. Total D loss = (real_loss + fake_loss) / 2
#   Step B — Train Generator:
#     1. Run fake images → D → G_loss     (want D(G(z)) → 1 to fool D)
#
# We track D(x) and D(G(z)) as accuracy proxies:
#   D(x)   ~ 1.0 means D is correctly identifying real images
#   D(G(z)) ~ 0.5 means training is balanced (equilibrium)

print(f"\n{'='*70}")
print(f"  STARTING TRAINING — {NUM_EPOCHS} epochs, {len(dataloader)} batches/epoch")
print(f"{'='*70}\n")

training_start = time.time()

# History for end-of-training summary
history = {
    "epoch": [], "epoch_time": [], "images_per_sec": [],
    "loss_D": [], "loss_G": [], "D_real": [], "D_fake": [],
    "gpu_mem_mb": []
}

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_start = time.time()
    G.train()
    D.train()

    # Accumulators reset each epoch
    running_loss_D = 0.0
    running_loss_G = 0.0
    running_D_real = 0.0   # Average D(x): should stay near 0.5-0.8
    running_D_fake = 0.0   # Average D(G(z)): should stay near 0.3-0.5

    # tqdm wraps the dataloader and prints a live progress bar.
    # desc= sets the prefix label; leave=False so bars don't stack up.
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]",
        leave=True,
        ncols=110
    )

    for batch_idx, (real_imgs, _) in enumerate(progress_bar):
        batch_size_actual = real_imgs.size(0)

        # Move real images to GPU (non-blocking for async transfer)
        real_imgs = real_imgs.to(device, non_blocking=True)

        # ── Labels ──────────────────────────────────────────────────────────
        # Soft labels (0.9 instead of 1.0) slightly stabilise training by
        # preventing the discriminator from becoming overconfident too quickly.
        real_labels = torch.full((batch_size_actual, 1), 0.9, device=device)
        fake_labels = torch.zeros(batch_size_actual, 1, device=device)

        # ══════════════════════════════════════════════════════════════════
        # STEP A — TRAIN DISCRIMINATOR
        # Goal: maximise  E[log D(x)] + E[log(1 - D(G(z)))]
        # ══════════════════════════════════════════════════════════════════
        optimizer_D.zero_grad()

        # A1. Real images through D
        real_preds = D(real_imgs)                 # D(x)
        loss_D_real = criterion(real_preds, real_labels)

        # A2. Generate fake images (detach so G is not updated here)
        z = torch.randn(batch_size_actual, LATENT_DIM, device=device)
        fake_imgs = G(z).detach()   # .detach() stops gradients flowing into G

        fake_preds = D(fake_imgs)                 # D(G(z))
        loss_D_fake = criterion(fake_preds, fake_labels)

        # A3. Total D loss and backward
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # ══════════════════════════════════════════════════════════════════
        # STEP B — TRAIN GENERATOR
        # Goal: maximise  E[log D(G(z))]  (fool D into calling fakes real)
        # ══════════════════════════════════════════════════════════════════
        optimizer_G.zero_grad()

        z = torch.randn(batch_size_actual, LATENT_DIM, device=device)
        fake_imgs = G(z)                          # fresh fakes (no detach)
        gen_preds = D(fake_imgs)                  # D evaluates fresh fakes

        # G wants D to output 1 for its fakes → compare to real_labels
        loss_G = criterion(gen_preds, real_labels)
        loss_G.backward()
        optimizer_G.step()

        # ── Accumulate metrics ───────────────────────────────────────────
        running_loss_D += loss_D.item()
        running_loss_G += loss_G.item()
        running_D_real += real_preds.mean().item()   # D(x): wants ~ 0.9
        running_D_fake += gen_preds.mean().item()    # D(G(z)): wants ~ 0.5 at equilib.

        # Update tqdm bar with live metrics every batch
        progress_bar.set_postfix({
            "D_loss": f"{loss_D.item():.4f}",
            "G_loss": f"{loss_G.item():.4f}",
            "D(x)":   f"{real_preds.mean().item():.3f}",
            "D(Gz)":  f"{gen_preds.mean().item():.3f}",
        })

    # ── End of epoch: compute averages ───────────────────────────────────
    n_batches     = len(dataloader)
    epoch_time    = time.time() - epoch_start
    images_per_sec = len(dataset) / epoch_time

    avg_loss_D = running_loss_D / n_batches
    avg_loss_G = running_loss_G / n_batches
    avg_D_real = running_D_real / n_batches
    avg_D_fake = running_D_fake / n_batches
    gpu_mem    = get_gpu_memory_mb()

    # ── Print epoch summary ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Epoch {epoch}/{NUM_EPOCHS} Summary")
    print(f"{'─'*70}")
    print(f"  Time            : {epoch_time:.2f}s")
    print(f"  Throughput      : {images_per_sec:,.0f} images/sec")
    print(f"  Loss D          : {avg_loss_D:.4f}   "
          f"← should stay near 0.5 (balanced)")
    print(f"  Loss G          : {avg_loss_G:.4f}   "
          f"← should decrease as G improves")
    print(f"  D(real)         : {avg_D_real:.3f}    "
          f"← D's confidence on real images (want ~0.7–0.9)")
    print(f"  D(G(z))         : {avg_D_fake:.3f}    "
          f"← D(fake after G update) — want ~0.4–0.6 at equilib.")
    if device.type == "cuda":
        print(f"  GPU mem alloc   : {gpu_mem:.1f} MB")
        print(f"  GPU mem reserved: {get_gpu_memory_reserved_mb():.1f} MB")
    print(f"{'─'*70}\n")

    # ── Log to history ────────────────────────────────────────────────────
    history["epoch"].append(epoch)
    history["epoch_time"].append(epoch_time)
    history["images_per_sec"].append(images_per_sec)
    history["loss_D"].append(avg_loss_D)
    history["loss_G"].append(avg_loss_G)
    history["D_real"].append(avg_D_real)
    history["D_fake"].append(avg_D_fake)
    history["gpu_mem_mb"].append(gpu_mem)

    # ── Save sample images ────────────────────────────────────────────────
    if epoch % SAMPLE_EVERY == 0:
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
            # Rescale from [-1,1] → [0,1] for saving
            samples = (samples + 1) / 2
            path = os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch:02d}.png")
            save_image(samples, path, nrow=8)
        print(f"  [SAMPLE] Saved 64 generated images → {path}")
        G.train()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FINAL PERFORMANCE REPORT (YOUR BASELINE)
# ─────────────────────────────────────────────────────────────────────────────
total_time = time.time() - training_start

print(f"\n{'='*70}")
print(f"  EXERCISE 1 — BASELINE PERFORMANCE REPORT")
print(f"  (This is your reference. Every future optimisation is measured")
print(f"   against these numbers.)")
print(f"{'='*70}")
print(f"\n  Total training time   : {total_time:.2f}s  ({total_time/60:.2f} min)")
print(f"  Average time/epoch    : {np.mean(history['epoch_time']):.2f}s")
print(f"  Average throughput    : {np.mean(history['images_per_sec']):,.0f} images/sec")
print(f"  Peak throughput       : {np.max(history['images_per_sec']):,.0f} images/sec")
print(f"\n  Loss Trajectory:")
print(f"  {'Epoch':<8} {'D Loss':<12} {'G Loss':<12} {'D(real)':<12} {'D(fake)':<12} {'Time(s)'}")
print(f"  {'-'*68}")
for i in range(NUM_EPOCHS):
    print(f"  {history['epoch'][i]:<8} "
          f"{history['loss_D'][i]:<12.4f} "
          f"{history['loss_G'][i]:<12.4f} "
          f"{history['D_real'][i]:<12.3f} "
          f"{history['D_fake'][i]:<12.3f} "
          f"{history['epoch_time'][i]:.2f}")

print(f"\n  GPU Memory (allocated):")
if device.type == "cuda":
    print(f"    Peak: {max(history['gpu_mem_mb']):.1f} MB")
    print(f"    Avg : {np.mean(history['gpu_mem_mb']):.1f} MB")
else:
    print(f"    N/A (CPU training)")

print(f"\n  STABILITY INDICATORS:")
d_loss_stable = all(0.1 < l < 1.5 for l in history['loss_D'])
g_loss_stable = all(0.1 < l < 5.0 for l in history['loss_G'])
print(f"  D loss in healthy range (0.1–1.5)  : {'✓ YES' if d_loss_stable else '✗ CHECK'}")
print(f"  G loss in healthy range (0.1–5.0)  : {'✓ YES' if g_loss_stable else '✗ CHECK'}")
d_real_ok = all(0.4 < v < 0.99 for v in history['D_real'])
print(f"  D(real) reasonable (0.4–0.99)      : {'✓ YES' if d_real_ok else '✗ CHECK'}")

print(f"\n{'='*70}")
print(f"  WHAT TO NOTE FOR THE NEXT EXERCISES:")
print(f"  → Time/epoch   = {np.mean(history['epoch_time']):.2f}s  ← target to reduce")
print(f"  → Throughput   = {np.mean(history['images_per_sec']):,.0f} img/s ← target to increase")
print(f"  → GPU memory   = {max(history['gpu_mem_mb']):.1f} MB    ← watch for AMP savings in Ex4")
print(f"  → D Loss       = {history['loss_D'][-1]:.4f}  ← stability reference")
print(f"  → G Loss       = {history['loss_G'][-1]:.4f}  ← stability reference")
print(f"{'='*70}\n")

# Save models for use in later exercises
torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, "generator_baseline.pth"))
torch.save(D.state_dict(), os.path.join(OUTPUT_DIR, "discriminator_baseline.pth"))
print(f"[CHECKPOINT] Models saved to {OUTPUT_DIR}/")
print(f"[DONE] Exercise 1 complete. Use the numbers above as your baseline.\n")
