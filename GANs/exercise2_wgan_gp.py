"""
================================================================================
EXERCISE 2: Architectural Modifications — WGAN-GP + Performance Analysis
================================================================================
GOAL:
    Replace the vanilla DCGAN (BCE loss) baseline with a WGAN-GP (Wasserstein GAN
    with Gradient Penalty). Measure:
      - How the architecture change affects training stability
      - Whether the Critic/Generator loss is more interpretable
      - How the gradient penalty adds compute overhead (we will measure it)
      - Side-by-side comparison of generated image quality vs Exercise 1

WHY WGAN-GP?
    Vanilla GAN suffers from:
      - Mode collapse (generator repeats the same output)
      - Vanishing gradients when discriminator is too strong
      - BCE loss that doesn't correlate with visual quality
    WGAN-GP fixes these by:
      - Using Wasserstein distance (Earth Mover's Distance) as the objective
      - Enforcing the 1-Lipschitz constraint via gradient penalty (GP)
        instead of weight clipping (original WGAN)
      - Providing a loss that actually correlates with image quality

WHAT IS NEW ARCHITECTURALLY:
    1. Discriminator → renamed "Critic" (no Sigmoid; outputs unbounded scalar)
    2. Gradient Penalty: computed on interpolated real/fake pairs
    3. Critic updates more than Generator (5 critic steps per 1 G step)
    4. Spectral Normalisation added to Critic for extra stability
    5. Layer Normalisation in Generator (BatchNorm breaks with GP)

DATASET: MNIST (same as Exercise 1 for fair comparison)
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
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

LATENT_DIM    = 100
IMAGE_SIZE    = 28
CHANNELS      = 1
BATCH_SIZE    = 64        # Smaller batch for GP stability (GP is computed per-batch)
NUM_EPOCHS    = 5
LR_G          = 1e-4      # WGAN-GP uses lower LR than DCGAN
LR_C          = 1e-4      # Critic LR
BETA1         = 0.0       # WGAN-GP paper: β₁=0 (no momentum) for stability
BETA2         = 0.9
N_CRITIC      = 5         # Critic steps per generator step (key WGAN parameter)
LAMBDA_GP     = 10        # Gradient penalty weight (from the original paper)
NUM_WORKERS   = 4
SAMPLE_EVERY  = 1
OUTPUT_DIR    = "ex2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DEVICE
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*70}")
print(f"  EXERCISE 2 — WGAN-GP: Architectural Modifications + Analysis")
print(f"{'='*70}")
print(f"\n[DEVICE] Using: {device}")
if device.type == "cuda":
    print(f"[DEVICE] GPU  : {torch.cuda.get_device_name(0)}")
    print(f"[DEVICE] VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[DATA] Loading MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])    # → [-1, 1]
])
dataset    = datasets.MNIST("./data", train=True, download=True, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    drop_last=True
)
print(f"[DATA] {len(dataset):,} images | batch={BATCH_SIZE} | {len(dataloader)} batches/epoch")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
# KEY CHANGE vs Exercise 1:
#   - BatchNorm2d → replaced with GroupNorm / InstanceNorm
#   - Why? Gradient Penalty is computed per-sample. BatchNorm mixes samples
#     within a batch, which corrupts the per-sample gradient norm used in GP.
#   - GroupNorm normalises within a group of channels per sample → compatible.

class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        self.conv_blocks = nn.Sequential(
            # Block 1: upsample 7→14
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),          # GroupNorm: 8 groups, 128 channels
            nn.ReLU(inplace=True),

            # Block 2: upsample 14→28
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            # Refine at full 28×28 resolution
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),

            # Output
            nn.Conv2d(32, CHANNELS, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z).view(z.size(0), 256, 7, 7)
        return self.conv_blocks(x)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CRITIC (formerly Discriminator)
# ─────────────────────────────────────────────────────────────────────────────
# KEY CHANGES vs Exercise 1:
#   1. No Sigmoid at output → Critic outputs unbounded real value (Wasserstein score)
#      Real images → high scores; Fake images → low scores
#   2. Spectral Normalisation on all Conv layers
#      → Constrains weight matrices to have spectral norm ≤ 1
#      → Enforces Lipschitz continuity (complementary to GP)
#   3. No BatchNorm (same reason as Generator — GP incompatibility)
#   4. LeakyReLU throughout with slope=0.2

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.utils.spectral_norm wraps any layer and normalises its weights
        self.model = nn.Sequential(
            # Block 1: (1,28,28) → (64,14,14)
            nn.utils.spectral_norm(nn.Conv2d(CHANNELS, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (64,14,14) → (128,7,7)
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (128,7,7) → (256,4,4)
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: (256,4,4) → (512,2,2)
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(512 * 2 * 2, 1))
            # NO Sigmoid — Wasserstein score is unbounded
        )

    def forward(self, img):
        return self.classifier(self.model(img))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GRADIENT PENALTY
# ─────────────────────────────────────────────────────────────────────────────
# The Gradient Penalty enforces the 1-Lipschitz constraint on the Critic.
# HOW IT WORKS:
#   1. Create interpolated samples between real and fake (random linear blend)
#   2. Pass interpolated samples through Critic
#   3. Compute gradient of Critic output w.r.t. interpolated samples
#   4. Penalise deviation of gradient norm from 1
#
# COMPUTE COST: GP requires an extra forward + backward through the Critic
#   per discriminator step. This is ~33% overhead per D update.
#   We will measure this overhead explicitly.

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    Computes the Wasserstein Gradient Penalty.
    Args:
        critic:       the Critic network
        real_samples: batch of real images  (N, C, H, W)
        fake_samples: batch of fake images  (N, C, H, W)
        device:       torch device
    Returns:
        gradient_penalty: scalar tensor
    """
    # Random interpolation coefficient ε ~ Uniform(0,1), one per sample
    epsilon = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_samples)   # broadcast to image shape

    # Interpolated images: x̂ = ε·x_real + (1-ε)·x_fake
    interpolated = (epsilon * real_samples + (1 - epsilon) * fake_samples)
    interpolated.requires_grad_(True)           # we need grad w.r.t. this

    # Forward pass through Critic on interpolated samples
    critic_interpolated = critic(interpolated)

    # Compute gradients of Critic output w.r.t. interpolated inputs
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,    # needed to differentiate through this gradient
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten gradients → compute L2 norm per sample
    gradients = gradients.view(gradients.size(0), -1)    # (N, C*H*W)
    gradient_norm = gradients.norm(2, dim=1)              # L2 norm per sample

    # Penalty = E[(||∇f(x̂)||₂ - 1)²]
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — WEIGHT INIT AND MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname and not hasattr(m, 'weight_orig'):
        # Don't re-init spectral-norm wrapped layers (they have 'weight_orig')
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "Linear" in classname and not hasattr(m, 'weight_orig'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "GroupNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G = Generator(LATENT_DIM).to(device)
C = Critic().to(device)
G.apply(weights_init)
C.apply(weights_init)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[MODEL] Generator parameters : {count_params(G):,}")
print(f"[MODEL] Critic parameters    : {count_params(C):,}")
print(f"[MODEL] Total parameters     : {count_params(G)+count_params(C):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — OPTIMISERS
# ─────────────────────────────────────────────────────────────────────────────
# WGAN-GP requires β₁=0 — momentum-based optimisers can destabilise training
# in the Wasserstein setting. The paper recommends RMSProp or Adam with β₁=0.

optimizer_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
optimizer_C = optim.Adam(C.parameters(), lr=LR_C, betas=(BETA1, BETA2))

fixed_noise = torch.randn(64, LATENT_DIM, device=device)

print(f"\n[OPTIM] Optimizer G : Adam  lr={LR_G}  β=({BETA1},{BETA2})")
print(f"[OPTIM] Optimizer C : Adam  lr={LR_C}  β=({BETA1},{BETA2})")
print(f"[OPTIM] N_CRITIC    : {N_CRITIC}  (critic updates per generator update)")
print(f"[OPTIM] LAMBDA_GP   : {LAMBDA_GP}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
# WGAN-GP update schedule:
#   For every batch:
#     1. Run N_CRITIC steps of Critic training
#     2. Run 1 step of Generator training
#
# Critic Loss = W-distance estimate:
#   L_C = E[C(fake)] - E[C(real)] + λ·GP
#   (We want C(real) high and C(fake) low → minimising L_C achieves this)
#
# Generator Loss:
#   L_G = -E[C(G(z))]
#   (We want C(fake) high → minimising L_G pushes fake scores up)
#
# Wasserstein Distance Estimate (logged):
#   W_dist = E[C(real)] - E[C(fake)]   (before GP)
#   Interpretation: higher = bigger distribution gap; approaches 0 at convergence

print(f"\n{'='*70}")
print(f"  STARTING WGAN-GP TRAINING — {NUM_EPOCHS} epochs")
print(f"  {N_CRITIC} critic steps per generator step")
print(f"{'='*70}\n")

training_start = time.time()

history = {
    "epoch": [], "epoch_time": [], "images_per_sec": [],
    "loss_C": [], "loss_G": [], "W_dist": [], "gp_time_ms": [],
    "gpu_mem_mb": []
}

# Data iterator — we need to cycle manually because N_CRITIC steps consume
# more batches per "iteration" than the dataloader naturally provides.
def infinite_loader(loader):
    """Cycle through dataloader indefinitely."""
    while True:
        for batch in loader:
            yield batch

data_iter = infinite_loader(dataloader)

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_start = time.time()
    G.train()
    C.train()

    running_loss_C  = 0.0
    running_loss_G  = 0.0
    running_W_dist  = 0.0
    running_gp_time = 0.0
    n_g_steps       = 0

    # Number of generator steps per epoch
    steps_per_epoch = len(dataloader) // N_CRITIC

    progress_bar = tqdm(
        range(steps_per_epoch),
        desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]",
        leave=True,
        ncols=115
    )

    for step in progress_bar:

        # ══════════════════════════════════════════════════════════════════
        # N_CRITIC STEPS OF CRITIC TRAINING
        # ══════════════════════════════════════════════════════════════════
        critic_losses_this_step = []
        gp_times_this_step = []

        for _ in range(N_CRITIC):
            real_imgs, _ = next(data_iter)
            real_imgs    = real_imgs.to(device, non_blocking=True)
            bs           = real_imgs.size(0)

            z         = torch.randn(bs, LATENT_DIM, device=device)
            fake_imgs = G(z).detach()     # detach: no G update during C steps

            optimizer_C.zero_grad()

            # Wasserstein scores
            score_real = C(real_imgs).mean()
            score_fake = C(fake_imgs).mean()

            # Gradient penalty (time it!)
            gp_start = time.perf_counter()
            gp = compute_gradient_penalty(C, real_imgs, fake_imgs.detach(), device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gp_time_ms = (time.perf_counter() - gp_start) * 1000

            # WGAN-GP Critic loss
            loss_C = score_fake - score_real + LAMBDA_GP * gp

            loss_C.backward()
            optimizer_C.step()

            critic_losses_this_step.append(loss_C.item())
            gp_times_this_step.append(gp_time_ms)

        # ══════════════════════════════════════════════════════════════════
        # 1 STEP OF GENERATOR TRAINING
        # ══════════════════════════════════════════════════════════════════
        real_imgs, _ = next(data_iter)
        real_imgs    = real_imgs.to(device, non_blocking=True)
        bs           = real_imgs.size(0)

        optimizer_G.zero_grad()
        z         = torch.randn(bs, LATENT_DIM, device=device)
        fake_imgs = G(z)
        loss_G    = -C(fake_imgs).mean()   # Maximise Critic score for fakes
        loss_G.backward()
        optimizer_G.step()

        # Wasserstein distance estimate for logging
        with torch.no_grad():
            w_dist = C(real_imgs).mean() - C(fake_imgs.detach()).mean()

        # Accumulate
        avg_c_this_step = np.mean(critic_losses_this_step)
        running_loss_C  += avg_c_this_step
        running_loss_G  += loss_G.item()
        running_W_dist  += w_dist.item()
        running_gp_time += np.mean(gp_times_this_step)
        n_g_steps       += 1

        progress_bar.set_postfix({
            "C_loss": f"{avg_c_this_step:.4f}",
            "G_loss": f"{loss_G.item():.4f}",
            "W_dist": f"{w_dist.item():.4f}",
            "GP_ms":  f"{np.mean(gp_times_this_step):.1f}",
        })

    # ── Epoch summary ─────────────────────────────────────────────────────
    epoch_time     = time.time() - epoch_start
    images_per_sec = len(dataset) / epoch_time

    avg_loss_C  = running_loss_C  / n_g_steps
    avg_loss_G  = running_loss_G  / n_g_steps
    avg_W_dist  = running_W_dist  / n_g_steps
    avg_gp_time = running_gp_time / n_g_steps
    gpu_mem     = torch.cuda.memory_allocated(device)/1e6 if device.type=="cuda" else 0

    print(f"\n{'─'*70}")
    print(f"  Epoch {epoch}/{NUM_EPOCHS} Summary  (WGAN-GP)")
    print(f"{'─'*70}")
    print(f"  Time              : {epoch_time:.2f}s")
    print(f"  Throughput        : {images_per_sec:,.0f} images/sec")
    print(f"  Critic Loss (avg) : {avg_loss_C:.4f}   ← includes GP penalty")
    print(f"  Generator Loss    : {avg_loss_G:.4f}   ← should decrease over time")
    print(f"  Wasserstein Dist  : {avg_W_dist:.4f}   ← interpretable: lower = more similar")
    print(f"  GP overhead (avg) : {avg_gp_time:.2f} ms/critic_step")
    print(f"  GP total/epoch    : {avg_gp_time * n_g_steps * N_CRITIC / 1000:.2f}s overhead")
    if device.type == "cuda":
        print(f"  GPU mem alloc     : {gpu_mem:.1f} MB")
    print(f"{'─'*70}\n")

    history["epoch"].append(epoch)
    history["epoch_time"].append(epoch_time)
    history["images_per_sec"].append(images_per_sec)
    history["loss_C"].append(avg_loss_C)
    history["loss_G"].append(avg_loss_G)
    history["W_dist"].append(avg_W_dist)
    history["gp_time_ms"].append(avg_gp_time)
    history["gpu_mem_mb"].append(gpu_mem)

    # Save samples
    if epoch % SAMPLE_EVERY == 0:
        G.eval()
        with torch.no_grad():
            samples = (G(fixed_noise) + 1) / 2
            path = os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch:02d}.png")
            save_image(samples, path, nrow=8)
        print(f"  [SAMPLE] → {path}")
        G.train()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — COMPARATIVE ANALYSIS REPORT
# ─────────────────────────────────────────────────────────────────────────────
total_time = time.time() - training_start

print(f"\n{'='*70}")
print(f"  EXERCISE 2 — ARCHITECTURAL ANALYSIS REPORT")
print(f"{'='*70}")
print(f"\n  Training Summary:")
print(f"  Total time      : {total_time:.2f}s  ({total_time/60:.2f} min)")
print(f"  Avg time/epoch  : {np.mean(history['epoch_time']):.2f}s")
print(f"  Avg throughput  : {np.mean(history['images_per_sec']):,.0f} img/s")

print(f"\n  Loss Trajectory (WGAN-GP):")
print(f"  {'Epoch':<8} {'C Loss':<12} {'G Loss':<12} {'W_dist':<12} {'GP ms/step':<12} {'Time(s)'}")
print(f"  {'-'*70}")
for i in range(NUM_EPOCHS):
    print(f"  {history['epoch'][i]:<8} "
          f"{history['loss_C'][i]:<12.4f} "
          f"{history['loss_G'][i]:<12.4f} "
          f"{history['W_dist'][i]:<12.4f} "
          f"{history['gp_time_ms'][i]:<12.2f} "
          f"{history['epoch_time'][i]:.2f}")

print(f"\n  GRADIENT PENALTY OVERHEAD ANALYSIS:")
avg_gp = np.mean(history['gp_time_ms'])
total_gp_s = avg_gp * sum(history['epoch_time']) / np.mean(history['epoch_time']) * \
             (len(dataloader) // N_CRITIC) * N_CRITIC / 1000
print(f"  Avg GP computation time : {avg_gp:.2f} ms per critic step")
print(f"  Estimated total GP time : {total_gp_s:.1f}s over {NUM_EPOCHS} epochs")
print(f"  GP fraction of training : {total_gp_s/total_time*100:.1f}%")
print(f"  → This is the cost of training stability with GP.")

print(f"\n  ARCHITECTURAL CHANGES AND THEIR IMPACT:")
print(f"  Change                         | Reason")
print(f"  {'-'*60}")
print(f"  BatchNorm → GroupNorm          | GP requires per-sample grad; BN mixes batch")
print(f"  Sigmoid removed from output    | Wasserstein needs unbounded critic scores")
print(f"  Spectral norm on critic layers | Lipschitz enforcement, complements GP")
print(f"  β₁=0 in Adam                   | Prevents momentum from destabilising W-train")
print(f"  N_CRITIC=5                     | Critic needs to converge faster than G")
print(f"  Deeper critic (4 conv blocks)  | Richer feature space for W-distance estimate")

print(f"\n  STABILITY INDICATORS (WGAN-GP specific):")
w_decreasing = history['W_dist'][-1] < history['W_dist'][0]
print(f"  W-distance decreasing over training : {'✓' if w_decreasing else '? check'}")
print(f"  Final W-distance : {history['W_dist'][-1]:.4f}  ← lower = G closer to data dist.")
print(f"  Critic loss range: {min(history['loss_C']):.4f} – {max(history['loss_C']):.4f}")
print(f"  Generator loss range: {min(history['loss_G']):.4f} – {max(history['loss_G']):.4f}")

print(f"\n  COMPARISON WITH EXERCISE 1 (load ex1 numbers to fill in):")
print(f"  Metric              | Ex1 (DCGAN/BCE) | Ex2 (WGAN-GP)")
print(f"  {'-'*50}")
print(f"  Avg throughput      | ??? img/s       | {np.mean(history['images_per_sec']):,.0f} img/s")
print(f"  Loss interpretable? | No (BCE)        | Yes (W-distance)")
print(f"  Mode collapse risk  | High            | Low")
print(f"  GP overhead         | None            | {np.mean(history['gp_time_ms']):.1f} ms/step")
print(f"  Architecture cost   | Lower           | Higher (4 vs 3 C blocks)")
print(f"\n  → Record the Exercise 1 numbers and compare manually!")
print(f"{'='*70}\n")

torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, "generator_wgangp.pth"))
torch.save(C.state_dict(), os.path.join(OUTPUT_DIR, "critic_wgangp.pth"))
print(f"[CHECKPOINT] Saved to {OUTPUT_DIR}/")
print(f"[DONE] Exercise 2 complete.\n")
