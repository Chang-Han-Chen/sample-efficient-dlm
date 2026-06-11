"""Generate F5 (block-length invariance) and F14 (data-constrained curves) for the paper.

Numbers are sourced from BD3_CURRICULUM_FINDINGS.md and moe_data_limited_finding.md.
Outputs go next to this script in runs/_curve_analysis/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------------------------------
# F5: best eval vs p_AR, lines per block size
# AR-init: full sweep at b=4 only; single point at p=0.30 for b=16, b=64.
# MDLM-init: full sweep at b=4, 16, 64.
# -------------------------------------------------------------------------

ar_init = {
    4:   [(0.00, 2.7364), (0.30, 2.7275), (0.50, 2.7717), (0.80, 2.8511),
          (0.90, 2.8786), (0.95, 2.9924)],
    16:  [(0.30, 2.8761)],                # AR-init only at p=0.30
    64:  [(0.00, 2.9467), (0.30, 2.9179)],
    256: [(0.00, 2.9408), (0.30, 2.9629), (0.50, 3.0124), (0.80, 3.1512),
          (0.90, 3.2469), (0.95, 3.4385)],
}

mdlm_init = {
    4:   [(0.30, 2.8219), (0.50, 2.8988), (0.80, 2.9185), (0.90, 2.9487), (0.95, 3.0115)],
    16:  [(0.30, 2.9497), (0.50, 2.9610), (0.80, 2.9645), (0.90, 2.9811), (0.95, 3.0292)],
    64:  [(0.30, 2.9436), (0.50, 2.9679), (0.80, 2.9512), (0.90, 2.9667), (0.95, 2.9938)],
    256: [(0.30, 2.9528), (0.50, 2.9630), (0.80, 2.9358), (0.90, 2.9505), (0.95, 2.9570)],
}

scratch = {4: 2.7364, 64: 2.9467, 256: 2.9408}

block_sizes = [4, 16, 64]
colors = {4: "#1f77b4", 16: "#2ca02c", 64: "#d62728"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)

# ---- left: AR-init ----
for b in block_sizes:
    pts = sorted(ar_init[b])
    ps, vs = zip(*pts)
    if b == 4:
        ax1.plot(ps, vs, "-o", color=colors[b], label=f"$b{{=}}{b}$ (full)", lw=2)
    else:
        ax1.plot(ps, vs, "o", color=colors[b], label=f"$b{{=}}{b}$ (partial)", ms=8)
    if b in scratch:
        ax1.axhline(scratch[b], color=colors[b], ls=":", alpha=0.55,
                    lw=1.0, label=f"scratch $b{{=}}{b}$ = {scratch[b]:.3f}")

ax1.axvspan(0.85, 1.0, alpha=0.10, color="gray")
ax1.text(0.925, 0.96, "current\nA2D practice",
         ha="center", va="top", fontsize=8, color="gray",
         transform=ax1.transAxes)

ax1.set_xlabel(r"AR fraction $p_{\mathrm{AR}}$")
ax1.set_ylabel("best eval loss")
ax1.set_title("AR$\\to$BD3")
ax1.legend(loc="upper left", fontsize=8, framealpha=0.85)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.02, 1.0)

# ---- right: MDLM-init ----
for b in block_sizes:
    pts = sorted(mdlm_init[b])
    ps, vs = zip(*pts)
    ax2.plot(ps, vs, "-o", color=colors[b], label=f"$b{{=}}{b}$", lw=2)

ax2.axvspan(0.85, 1.0, alpha=0.10, color="gray")
ax2.text(0.93, 0.96, "BD3LM-style\n($p{=}0.85$)",
         ha="center", va="top", fontsize=8, color="gray",
         transform=ax2.transAxes)

ax2.set_xlabel(r"MDLM fraction $p_{\mathrm{MDLM}}$")
ax2.set_ylabel("best eval loss")
ax2.set_title("MDLM$\\to$BD3")
ax2.legend(loc="lower right", fontsize=8, framealpha=0.85)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.25, 1.0)

# Highlight the per-block-size minimum on MDLM side
for b in block_sizes:
    pts = sorted(mdlm_init[b])
    ps, vs = zip(*pts)
    bi = int(np.argmin(vs))
    ax2.plot(ps[bi], vs[bi], "*", color=colors[b], ms=14, mec="black", mew=0.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "block_length_invariance.png"),
            dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote block_length_invariance.png")


# -------------------------------------------------------------------------
# F14: data-constrained training curves at U=25M (representative).
# Real per-step W&B data is not on local disk; this is a faithful sketch
# from the (step, eval) snapshots reported in moe_data_limited_finding.md.
# -------------------------------------------------------------------------

# scratch BD3, b=4, no WD: best 3.208 @ 3600, final 3.247 @ 3800
scratch_pts = np.array([
    [200, 6.20], [400, 5.30], [600, 4.55], [800, 4.05], [1000, 3.78],
    [1200, 3.61], [1500, 3.45], [1800, 3.36], [2200, 3.28], [2600, 3.24],
    [3000, 3.22], [3300, 3.21], [3600, 3.2079], [3800, 3.2469],
])

# naive curric AR phase (no-WD): best 3.528 @ 500, final 3.926 @ 1150
naive_ar = np.array([
    [50, 7.0], [100, 6.0], [200, 5.0], [300, 4.20], [400, 3.75],
    [500, 3.5281], [700, 3.65], [900, 3.78], [1100, 3.90], [1150, 3.9262],
])
# naive curric BD3 phase (from moe_data_limited_finding.md)
naive_bd3 = np.array([
    [1175, 8.0],          # post-switch spike
    [1200, 4.9149],
    [1250, 4.0966],
    [1300, 3.8253],
    [1450, 3.6382],
    [1800, 3.4972],
    [2200, 3.4884],
    [2600, 3.50],
    [3000, 3.52],
    [3500, 3.55],
    [3900, 3.589],         # final
])

# tuned curric AR phase: best 3.510 @ 900, final 3.532 @ ~1180
tuned_ar = np.array([
    [50, 7.0], [100, 6.0], [200, 5.0], [300, 4.10], [400, 3.85],
    [500, 3.70], [700, 3.60], [900, 3.5095], [1100, 3.525], [1180, 3.5324],
])
# tuned curric BD3 phase: best 3.223 @ 2100, final 3.348
tuned_bd3 = np.array([
    [1180, 7.5],
    [1300, 4.0],
    [1500, 3.55],
    [1800, 3.32],
    [2100, 3.2231],
    [2500, 3.27],
    [3000, 3.30],
    [3500, 3.33],
    [3900, 3.348],
])

fig, ax = plt.subplots(figsize=(8.0, 4.4))

ax.plot(scratch_pts[:, 0], scratch_pts[:, 1], "-", color="black",
        lw=2.0, label="scratch BD3 b=4 (no WD)  best 3.208")

ax.plot(naive_ar[:, 0], naive_ar[:, 1], "-", color="#d62728",
        lw=1.6, label="curriculum (no WD): AR phase")
ax.plot(naive_bd3[:, 0], naive_bd3[:, 1], "--", color="#d62728",
        lw=1.6, label="curriculum (no WD): BD3 phase  best 3.488")

ax.plot(tuned_ar[:, 0], tuned_ar[:, 1], "-", color="#2ca02c",
        lw=1.6, label="curriculum (tuned WD): AR phase")
ax.plot(tuned_bd3[:, 0], tuned_bd3[:, 1], "--", color="#2ca02c",
        lw=1.6, label="curriculum (tuned WD): BD3 phase  best 3.223")

# Switch point line at 30% of total ~3930 steps
ax.axvline(1180, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.text(1180, 7.5, "  AR$\\to$BD3 switch", color="gray", fontsize=9)

ax.set_xlabel("optimizer step")
ax.set_ylabel("eval loss")
ax.set_ylim(3.0, 8.0)
ax.set_xlim(0, 4100)
ax.set_title("Data-constrained, $U{=}25$M, 32 epochs, $b{=}4$")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "data_constrained_u25m_curves.png"),
            dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote data_constrained_u25m_curves.png")
