"""Test-set report for a trained checkpoint. Run: python report.py best_gnn.pt [seed]"""
import sys, os, glob, random, csv                                    # args, files, split, CSV likhne ke liye
import numpy as np                                                  # maths
import torch                                                        # model load + inference
import matplotlib                                                   # plots
matplotlib.use("Agg")                                               # headless: window nahi khulegi
import matplotlib.pyplot as plt
from train import Net, to_data                                      # wahi model aur graph converter

ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_gnn.pt"           # checkpoint path
seed = int(sys.argv[2]) if len(sys.argv) > 2 else (                  # split ka seed
    int(os.path.basename(ckpt).split("_s")[-1].split(".")[0]) if "_s" in os.path.basename(ckpt) else 0)
graph = "mlp" not in os.path.basename(ckpt)                         # mlp_s*.pt = no-graph ablation

fs = sorted(glob.glob("graphs/*.npz"))                               # train.py jaisi same file list
assert fs, "graphs/ khali hai, pehle prep.py chalao"
random.Random(seed).shuffle(fs)                                     # train.py jaisa hi shuffle
te = fs[int(.85 * len(fs)):]                                        # wahi 15% test split
m = Net(graph=graph)                                                # model
m.load_state_dict(torch.load(ckpt, map_location="cpu"))             # trained weights
m.eval()                                                            # dropout band (deterministic)

rows = []                                                           # har scan ka ek row
with torch.no_grad():
    for f in te:
        d = to_data(dict(np.load(f)))
        p = torch.sigmoid(m(d))                                     # (N,3) predicted fraction
        y, w, vox = d.y, d.cnt[:, None], float(d.vox)               # label, node voxels, voxel size
        b = (p > 0.5).float()                                       # train.py jaisa 0.5 threshold
        inter = (b * y * w).sum(0)                                  # intersection
        pred = (b * w).sum(0); true = (y * w).sum(0)                # predicted / true positive
        dice = (2 * inter + 1e-6) / (pred + true + 1e-6)            # Dice = F1
        prec = (inter + 1e-6) / (pred + 1e-6)                       # precision
        rec = (inter + 1e-6) / (true + 1e-6)                        # recall (sensitivity)
        pv = (p * w).sum(0) * vox / 1000                            # predicted volume cm3 (w = cnt[:,None])
        tv = (y * w).sum(0) * vox / 1000                            # true volume cm3
        rows.append(dict(name=os.path.basename(f), dice=dice.numpy(), prec=prec.numpy(),
                         rec=rec.numpy(), pv=pv.numpy(), tv=tv.numpy()))

D = np.stack([r["dice"] for r in rows]); P = np.stack([r["prec"] for r in rows])
R = np.stack([r["rec"] for r in rows]); PV = np.stack([r["pv"] for r in rows])
TV = np.stack([r["tv"] for r in rows])                              # (scans, 3)
diff = PV - TV
MAE = np.abs(diff).mean(0); RMSE = np.sqrt((diff ** 2).mean(0))     # volume error
sst = ((TV - TV.mean(0)) ** 2).sum(0)
R2 = 1 - ((diff ** 2).sum(0) / np.where(sst > 0, sst, np.nan))      # R^2 (constant → nan)
corr = [np.corrcoef(TV[:, i], D[:, i])[0, 1] if TV[:, i].std() > 0 and D[:, i].std() > 0 else np.nan
        for i in range(3)]                                          # size vs Dice correlation

print(f"Checkpoint: {ckpt} | split seed {seed} | {len(rows)} test scans", flush=True)
for i, n in enumerate(["WT", "TC", "ET"]):
    print(f"{n}: Dice {D[:, i].mean():.3f} | Precision {P[:, i].mean():.3f} | Recall {R[:, i].mean():.3f}"
          f" | VolMAE {MAE[i]:.1f} | RMSE {RMSE[i]:.1f} | R2 {R2[i]:.3f} | corr(size,Dice) {corr[i]:+.3f}",
          flush=True)
print(f"Mean Dice {D.mean():.3f}", flush=True)
print("Worst 5 scans (by mean Dice):", flush=True)
for r in sorted(rows, key=lambda z: z["dice"].mean())[:5]:
    print(f"  {r['name']}: mean {r['dice'].mean():.3f} | WT/TC/ET {np.round(r['dice'], 3).tolist()}"
          f" | true WT vol {r['tv'][0]:.1f} cm3", flush=True)

os.makedirs("report_out", exist_ok=True)                            # plots + CSV yahin jayenge
fig, axs = plt.subplots(1, 3, figsize=(13, 4))                      # predicted vs true volume
for i, (ax, n) in enumerate(zip(axs, ["WT", "TC", "ET"])):
    ax.scatter(TV[:, i], PV[:, i], s=18)
    lim = [0, max(TV[:, i].max(), PV[:, i].max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1); ax.set(xlim=lim, ylim=lim, title=n,
                                          xlabel="true cm3", ylabel="predicted cm3")
fig.tight_layout(); fig.savefig("report_out/volume_scatter.png", dpi=120)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))                      # tumor size vs Dice
for i, (ax, n) in enumerate(zip(axs, ["WT", "TC", "ET"])):
    ax.scatter(TV[:, i], D[:, i], s=18); ax.set(title=n, xlabel="true cm3", ylabel="Dice", ylim=(0, 1))
fig.tight_layout(); fig.savefig("report_out/size_vs_dice.png", dpi=120)

with open("report_out/per_scan.csv", "w", newline="") as fh:        # README table ke liye raw data
    wr = csv.writer(fh); wr.writerow(["scan", "region", "dice", "precision", "recall", "pred_vol", "true_vol"])
    for r in rows:
        for i, n in enumerate(["WT", "TC", "ET"]):
            wr.writerow([r["name"], n, *np.round([r["dice"][i], r["prec"][i], r["rec"][i],
                                                  r["pv"][i], r["tv"][i]], 4)])
print("Plots + CSV: report_out/", flush=True)
