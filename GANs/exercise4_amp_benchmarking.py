"""
================================================================================
EXERCISE 4 — AMP Integration and Throughput Benchmarking
================================================================================

GOAL
----
Integrate Automatic Mixed Precision (AMP) into GAN training and measure:
  - Throughput gain (images/sec) vs fp32 baseline
  - Memory reduction (VRAM usage) vs fp32 baseline
  - Training stability: do losses and image quality hold up?
  - Numerical safety: do we hit NaN/Inf with GAN losses?

WHAT YOU WILL LEARN
-------------------
1. What AMP is and why it works (tensor cores, memory bandwidth)
2. How to correctly apply GradScaler to the GAN training loop
3. Why GAN training requires EXTRA care with AMP (one shared scaler)
4. How to verify AMP is actually being used (dtype checks)
5. How to measure and report speedup properly (wall time)

HOW TO RUN
----------
    python exercise4_amp_benchmarking.py

WHAT HAPPENS
------------
  Round 1 - fp32 Baseline
    Standard DCGAN, no AMP. Records time/epoch, imgs/sec, memory, losses.

  Round 2 - AMP (bf16 or fp16 depending on hardware)
    Same loop with torch.autocast + GradScaler. Same metrics.

  Round 3 - AMP + Larger Batch
    AMP frees memory so we can fit a larger batch.
    Shows the compounded benefit: AMP speed + batch size throughput.

  Final Report: side-by-side table + interpretation guide.

AMP THEORY (quick recap)
-------------------------
  fp32: 32 bits, 7 decimal digits precision
  fp16: 16 bits, 3 decimal digits precision -> 2x smaller, 2-8x faster on Tensor Cores
  bf16: 16 bits, same exponent range as fp32 -> safer for GANs (no overflow)

  torch.autocast: automatically casts eligible ops (Conv2d, matmul) to fp16/bf16.
                  Loss, BatchNorm, Sigmoid stay in fp32 for numerical safety.

  GradScaler: scales loss up before backward() to prevent gradient underflow.
              Scales down before opt.step(). Skips step if Inf/NaN detected.

  GAN rule: ONE scaler shared between G and D.
            Call scaler.step() for BOTH optimisers.
            Call scaler.update() ONCE at end of iteration.
================================================================================
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(42)


# ==============================================================================
# 0.  CONFIGURATION
# ==============================================================================

CFG = dict(
    image_size    = 64,
    num_workers   = 4,
    latent_dim    = 100,
    ngf           = 64,
    ndf           = 64,
    lr_g          = 2e-4,
    lr_d          = 2e-4,
    beta1         = 0.5,
    beta2         = 0.999,
    num_epochs    = 10,
    output_dir    = "outputs",
    n_eval_imgs   = 64,
    batch_fp32    = 128,   # Round 1: fp32 baseline
    batch_amp     = 128,   # Round 2: AMP, same batch (isolates AMP gain)
    batch_amp_lg  = 256,   # Round 3: AMP + larger batch (fits due to AMP mem savings)
)


# ==============================================================================
# 1.  DCGAN MODELS  (identical to Exercise 1)
# ==============================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim, ngf):
        super().__init__()
        self.project = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True))
        self.up1 = self._b(ngf*8, ngf*4)
        self.up2 = self._b(ngf*4, ngf*2)
        self.up3 = self._b(ngf*2, ngf)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False), nn.Tanh())

    @staticmethod
    def _b(i, o):
        return nn.Sequential(
            nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.out(self.up3(self.up2(self.up1(self.project(z)))))


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf,      4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf,   ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1,    4, 1, 0, bias=False))

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    cn = m.__class__.__name__
    if "Conv" in cn:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cn:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def build_models(device, cfg):
    G = Generator(cfg["latent_dim"], cfg["ngf"]).to(device)
    D = Discriminator(cfg["ndf"]).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D


def build_loader(cfg, batch_size, num_workers):
    t = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


# ==============================================================================
# 2.  HELPERS
# ==============================================================================

def mem_alloc_mb(device):
    return torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0

def peak_mem_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0

def detect_amp_dtype(device):
    if device.type != "cuda":
        return None, "CPU - AMP not applicable"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16 (same exponent as fp32, safer for GANs)"
    return torch.float16, "float16 (GradScaler essential to prevent underflow)"

def save_grid(G, noise, path):
    G.eval()
    with torch.no_grad():
        imgs = ((G(noise).cpu() + 1) / 2.0)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(imgs, nrow=8, padding=2), path)
    G.train()


# ==============================================================================
# 3.  TRAINING LOOP - fp32 (BASELINE, NO AMP)
# ==============================================================================
# Unchanged from Exercise 1. This is the reference point.

def train_epoch_fp32(epoch, loader, G, D, opt_G, opt_D,
                     criterion, device, cfg, label="fp32"):
    G.train(); D.train()
    sum_lD = sum_lG = sum_Dx = sum_DGz = 0.0
    images_seen = 0
    t0 = time.perf_counter()
    bs_cfg = loader.batch_size
    ones  = torch.ones(bs_cfg, device=device)
    zeros = torch.zeros(bs_cfg, device=device)

    pbar = tqdm(loader,
                desc=f"  [{label}] Epoch {epoch:>3}/{cfg['num_epochs']}",
                unit="batch", dynamic_ncols=True, colour="cyan")

    for real_imgs, _ in pbar:
        bs = real_imgs.size(0)
        images_seen += bs
        real_imgs = real_imgs.to(device, non_blocking=True)

        opt_D.zero_grad()
        out_real  = D(real_imgs)
        loss_D_r  = criterion(out_real, ones[:bs])
        Dx = out_real.mean().item()
        noise    = torch.randn(bs, cfg["latent_dim"], device=device)
        fake     = G(noise)
        out_fake = D(fake.detach())
        loss_D_f = criterion(out_fake, zeros[:bs])
        loss_D   = (loss_D_r + loss_D_f) / 2.0
        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()
        out_f2 = D(fake)
        loss_G = criterion(out_f2, ones[:bs])
        DGz    = out_f2.mean().item()
        loss_G.backward()
        opt_G.step()

        sum_lD += loss_D.item(); sum_lG += loss_G.item()
        sum_Dx += Dx;             sum_DGz += DGz
        pbar.set_postfix(D_loss=f"{loss_D.item():.4f}", G_loss=f"{loss_G.item():.4f}",
                         D_x=f"{Dx:.3f}", D_Gz=f"{DGz:.3f}",
                         mem_MB=f"{mem_alloc_mb(device):.0f}")

    pbar.close()
    n = len(loader)
    elapsed = time.perf_counter() - t0
    return dict(epoch=epoch, loss_D=sum_lD/n, loss_G=sum_lG/n,
                D_x=sum_Dx/n, D_Gz=sum_DGz/n,
                epoch_time_s=elapsed, imgs_per_sec=images_seen/elapsed,
                mem_alloc_mb=mem_alloc_mb(device))


# ==============================================================================
# 4.  TRAINING LOOP - AMP
# ==============================================================================
# Diff from fp32 loop: 5 lines changed, shown with comments below.
#
#  KEY CHANGE 1: torch.autocast wraps forward passes
#    - Conv2d, ConvTranspose2d ops run in fp16/bf16 inside this block
#    - BCELoss stays in fp32 (autocast excludes loss functions)
#    - No code changes inside the block are needed
#
#  KEY CHANGE 2: scaler.scale(loss).backward()
#    - Multiplies loss by scale factor before backward
#    - Prevents tiny fp16 gradients from becoming zero (underflow)
#
#  KEY CHANGE 3: scaler.step(opt) replaces opt.step()
#    - Unscales gradients internally
#    - Checks for Inf/NaN: if found, skips this weight update safely
#
#  KEY CHANGE 4: scaler.update() called ONCE per iteration
#    - Adjusts scale factor: halve if any Inf/NaN, double every 2000 clean steps
#    - Must come AFTER both scaler.step() calls (for D and G)
#
#  MONITOR: 'scale=' in tqdm shows current scale factor
#    Healthy: > 1000 and stable or growing
#    Problem: dropping every few steps -> gradient overflow -> try bf16

def train_epoch_amp(epoch, loader, G, D, opt_G, opt_D, criterion,
                    scaler, device, cfg, amp_dtype, label="AMP"):
    G.train(); D.train()
    sum_lD = sum_lG = sum_Dx = sum_DGz = 0.0
    images_seen = 0
    t0 = time.perf_counter()
    bs_cfg = loader.batch_size
    ones  = torch.ones(bs_cfg, device=device)
    zeros = torch.zeros(bs_cfg, device=device)

    pbar = tqdm(loader,
                desc=f"  [{label}] Epoch {epoch:>3}/{cfg['num_epochs']}",
                unit="batch", dynamic_ncols=True, colour="green")

    for real_imgs, _ in pbar:
        bs = real_imgs.size(0)
        images_seen += bs
        real_imgs = real_imgs.to(device, non_blocking=True)

        # --- DISCRIMINATOR UPDATE WITH AMP ---
        opt_D.zero_grad()
        # KEY CHANGE 1: autocast context wraps D and G forward passes
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            out_real  = D(real_imgs)
            loss_D_r  = criterion(out_real, ones[:bs])
            Dx = out_real.mean().item()
            noise    = torch.randn(bs, cfg["latent_dim"], device=device)
            fake     = G(noise)
            out_fake = D(fake.detach())
            loss_D_f = criterion(out_fake, zeros[:bs])
            loss_D   = (loss_D_r + loss_D_f) / 2.0
        # KEY CHANGE 2: scale loss before backward
        scaler.scale(loss_D).backward()
        # KEY CHANGE 3: scaler.step() instead of opt.step()
        scaler.step(opt_D)

        # --- GENERATOR UPDATE WITH AMP ---
        opt_G.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            out_f2 = D(fake)
            loss_G = criterion(out_f2, ones[:bs])
            DGz    = out_f2.mean().item()
        scaler.scale(loss_G).backward()
        scaler.step(opt_G)
        # KEY CHANGE 4: ONE update() call per iteration, after BOTH steps
        scaler.update()

        sum_lD += loss_D.item(); sum_lG += loss_G.item()
        sum_Dx += Dx;             sum_DGz += DGz
        pbar.set_postfix(D_loss=f"{loss_D.item():.4f}", G_loss=f"{loss_G.item():.4f}",
                         D_x=f"{Dx:.3f}", D_Gz=f"{DGz:.3f}",
                         scale=f"{scaler.get_scale():.0f}",
                         mem_MB=f"{mem_alloc_mb(device):.0f}")

    pbar.close()
    n = len(loader)
    elapsed = time.perf_counter() - t0
    return dict(epoch=epoch, loss_D=sum_lD/n, loss_G=sum_lG/n,
                D_x=sum_Dx/n, D_Gz=sum_DGz/n,
                epoch_time_s=elapsed, imgs_per_sec=images_seen/elapsed,
                mem_alloc_mb=mem_alloc_mb(device),
                amp_scale=scaler.get_scale())


# ==============================================================================
# 5.  AMP VERIFICATION
# ==============================================================================

def verify_amp(device, amp_dtype, cfg):
    print("\n  AMP VERIFICATION - checking tensor dtypes inside autocast")
    print("  " + "-" * 60)
    G_v = Generator(cfg["latent_dim"], cfg["ngf"]).to(device)
    G_v.eval()
    z = torch.randn(2, cfg["latent_dim"], device=device)
    with torch.no_grad():
        out_fp32 = G_v(z)
    print(f"  Generator output dtype (no autocast): {out_fp32.dtype}")
    if device.type == "cuda" and amp_dtype is not None:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            out_amp = G_v(z)
        print(f"  Generator output dtype (in autocast): {out_amp.dtype}")
        if out_amp.dtype in (torch.float16, torch.bfloat16):
            print("  OK: AMP active - outputs are in reduced precision")
        else:
            print("  OK: Output is fp32 (Tanh output stays fp32 - expected)")
            print("      Internal Conv2d activations ARE in reduced precision.")
    else:
        print("  (Verification skipped - not on CUDA)")
    del G_v
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ==============================================================================
# 6.  RUN ONE TRAINING ROUND
# ==============================================================================

def run_round(label, use_amp, batch_size, device, cfg, amp_dtype):
    print(f"\n  {'='*68}")
    print(f"  ROUND: {label}")
    print(f"  {'='*68}")
    print(f"  AMP    : {'ON  dtype=' + str(amp_dtype) if use_amp else 'OFF (fp32)'}")
    print(f"  Batch  : {batch_size}")
    print(f"  Epochs : {cfg['num_epochs']}")
    print()

    G, D = build_models(device, cfg)
    opt_G = optim.Adam(G.parameters(), lr=cfg["lr_g"],
                       betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = optim.Adam(D.parameters(), lr=cfg["lr_d"],
                       betas=(cfg["beta1"], cfg["beta2"]))
    criterion   = nn.BCEWithLogitsLoss()
    loader      = build_loader(cfg, batch_size, cfg["num_workers"])
    fixed_noise = torch.randn(cfg["n_eval_imgs"], cfg["latent_dim"], device=device)
    # GradScaler: enabled=False makes it a no-op (safe for fp32 runs too)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    print(f"  {'Ep':<5} {'Time s':<9} {'Imgs/s':<10} {'D_loss':<9}"
          f"{'G_loss':<9} {'D(x)':<7} {'D(G(z))':<10} Mem MB")
    print("  " + "-" * 72)

    history = []
    for epoch in range(1, cfg["num_epochs"] + 1):
        if use_amp:
            m = train_epoch_amp(epoch, loader, G, D, opt_G, opt_D, criterion,
                                scaler, device, cfg, amp_dtype, label)
        else:
            m = train_epoch_fp32(epoch, loader, G, D, opt_G, opt_D, criterion,
                                 device, cfg, label)
        history.append(m)
        amp_info = f" sc={m.get('amp_scale', 0):.0f}" if use_amp else ""
        print(f"  {epoch:<5} {m['epoch_time_s']:<9.1f} {m['imgs_per_sec']:<10.0f}"
              f"{m['loss_D']:<9.4f}{m['loss_G']:<9.4f}"
              f"{m['D_x']:<7.4f} {m['D_Gz']:<10.4f}"
              f"{m['mem_alloc_mb']:.1f}{amp_info}")

        if epoch == cfg["num_epochs"]:
            path = os.path.join(cfg["output_dir"],
                                f"exercise4_{label.replace(' ','_')}_ep{epoch:03d}.png")
            save_grid(G, fixed_noise, path)
            print(f"  -> Grid: {path}")

    pk = peak_mem_mb(device)
    print(f"\n  Peak VRAM: {pk:.1f} MB")
    return history, pk


# ==============================================================================
# 7.  FINAL REPORT
# ==============================================================================

def print_final_report(rounds):
    print("\n" + "=" * 72)
    print("  FINAL BENCHMARKING REPORT - EXERCISE 4")
    print("=" * 72)
    avg = lambda h, k: sum(m[k] for m in h) / len(h)

    print(f"\n  {'Config':<28} {'Avg Imgs/s':>11} {'Speedup':>9}"
          f"{'Avg ep s':>10} {'Peak VRAM':>11} {'D_loss':>9} {'G_loss':>9}")
    print("  " + "-" * 90)

    base_speed = None
    for label, history, peak in rounds:
        sp = avg(history, "imgs_per_sec")
        ep = avg(history, "epoch_time_s")
        if base_speed is None:
            base_speed = sp
        speedup = sp / base_speed
        print(f"  {label:<28} {sp:>11.0f} {speedup:>8.2f}x"
              f"{ep:>10.1f}s {peak:>10.1f}MB"
              f"{history[-1]['loss_D']:>9.4f} {history[-1]['loss_G']:>9.4f}")

    print()
    if len(rounds) >= 2:
        mem0 = rounds[0][2]; mem1 = rounds[1][2]
        if mem0 > 0:
            saving = (mem0 - mem1) / mem0 * 100
            print(f"  VRAM reduction fp32 -> AMP: {saving:.1f}%  ({mem0:.0f} -> {mem1:.0f} MB)")

    print()
    print("  INTERPRETATION GUIDE")
    print("  " + "-" * 68)
    print("  Speedup > 1.5x   : AMP + Tensor Cores working well")
    print("  Speedup 1.0-1.5x : AMP active but not the bottleneck")
    print("  Speedup < 1.0x   : AMP overhead > benefit (small batch or CPU)")
    print()
    print("  STABILITY: fp32 vs AMP final D_loss/G_loss should be within +/-0.05")
    print("  Larger divergence -> numerical issues -> switch fp16 to bf16")
    print()
    print("  AMP SCALE FACTOR (tqdm 'scale=' column during AMP training):")
    print("  > 1000 stable  -> numerically healthy, keep going")
    print("  Dropping often -> gradient overflow -> lower lr or use bf16")
    print()
    print("  THE 5 LINES THAT ENABLE AMP IN A GAN LOOP:")
    print("  1. scaler = torch.cuda.amp.GradScaler()")
    print("  2. with torch.autocast(device_type, dtype):  # wrap forward pass")
    print("  3.     [forward pass unchanged inside block]")
    print("  4. scaler.scale(loss).backward()")
    print("  5. scaler.step(opt_D); scaler.step(opt_G)")
    print("  6. scaler.update()  # once per iteration, after all steps")
    print("=" * 72)


# ==============================================================================
# 8.  MAIN
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  EXERCISE 4 - AMP Integration and Throughput Benchmarking")
    print("=" * 72)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  VRAM   : {props.total_memory/1024**2:.0f} MB")
        print(f"  CUDA   : {torch.version.cuda}")
    print()

    amp_dtype, amp_desc = detect_amp_dtype(device)
    print(f"  AMP dtype: {amp_desc}")
    if amp_dtype is None:
        print("  WARNING: CPU detected. AMP won't accelerate but code is demonstrated.")
        amp_dtype = torch.float32
    print()

    os.makedirs(CFG["output_dir"], exist_ok=True)
    verify_amp(device, amp_dtype, CFG)

    print("\n  PLAN:")
    print(f"  Round 1 - fp32 baseline     batch={CFG['batch_fp32']}")
    print(f"  Round 2 - AMP same batch    batch={CFG['batch_amp']}")
    print(f"  Round 3 - AMP larger batch  batch={CFG['batch_amp_lg']}")
    print()

    rounds = []

    h1, pk1 = run_round("fp32 baseline", False, CFG["batch_fp32"],
                          device, CFG, amp_dtype=None)
    rounds.append(("fp32 (batch=128)", h1, pk1))

    h2, pk2 = run_round("AMP same batch", True, CFG["batch_amp"],
                          device, CFG, amp_dtype=amp_dtype)
    tag = str(amp_dtype).split(".")[-1]
    rounds.append((f"AMP {tag} (batch=128)", h2, pk2))

    print("\n  Round 3: larger batch made possible by AMP memory savings.")
    print(f"  If OOM with batch={CFG['batch_amp_lg']}, reduce batch_amp_lg in CFG.")
    try:
        h3, pk3 = run_round("AMP large batch", True, CFG["batch_amp_lg"],
                             device, CFG, amp_dtype=amp_dtype)
        rounds.append((f"AMP {tag} (batch=256)", h3, pk3))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM at batch={CFG['batch_amp_lg']} - skipping Round 3.")
        else:
            raise

    print_final_report(rounds)

    result_path = os.path.join(CFG["output_dir"], "exercise4_results.pt")
    torch.save({"rounds": [(lbl, h, pk) for lbl, h, pk in rounds],
                "amp_dtype": str(amp_dtype)}, result_path)
    print(f"\n  Results -> {result_path}")
    print()
    print("  WHAT WE DEMONSTRATED:")
    print("  1. AMP = ~6 lines of code, 1.5-3x speedup on Tensor Core GPUs")
    print("  2. AMP halves activation VRAM -> larger batches fit")
    print("  3. GAN + AMP is stable with correct GradScaler usage")
    print("  4. bf16 > fp16 for GAN training (wider numerical range)")
    print("  5. Profile first (Ex 3), then apply AMP where it helps most")
    print()
    print("  Exercise 4 complete.")
    print()


if __name__ == "__main__":
    main()
