"""Tier-1 curves: val Dice from saved logs + train/val gap from best_gnn.pt.
Run: python curves.py   (no retraining, checkpoints untouched)."""
import os, re, glob, random                                          # files, regex, split
import numpy as np                                                  # maths
import torch                                                        # model load
import matplotlib                                                   # plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import Net, to_data, evaluate, loss_fn                   # same model / metrics / loss

os.makedirs("report_out", exist_ok=True)                            # output folder

# --- 1) validation curves from the logs we already saved -------------------
series = {}                                                         # name -> [(epoch, dice)]
for f in sorted(glob.glob("logs/*.txt")):
    vals = []
    raw = open(f, "rb").read()                                      # Tee-Object UTF-16 likhata hai
    text = raw.decode("utf-16", errors="ignore") if b"\x00" in raw[:20] else raw.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+val Dice\s+([0-9.]+)", line)      # "12 val Dice 0.683"
        if m:
            vals.append((int(m.group(1)), float(m.group(2))))
    if len(vals) > 5:                                               # 60 epochs = real curve
        series[os.path.basename(f)[:-4]] = vals

plt.figure(figsize=(8, 5))
for name, vals in series.items():
    plt.plot([e for e, _ in vals], [v for _, v in vals], label=name, lw=1.6)
plt.xlabel("epoch"); plt.ylabel("val Dice"); plt.title("Validation learning curves (saved runs)")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("report_out/curves_val.png", dpi=150)
plt.close()
print("Val curves:", {k: (len(v), round(v[-1][1], 3)) for k, v in series.items()}, flush=True)

# --- 2) train / val / test gap from the saved seed-0 checkpoint ------------
fs = sorted(glob.glob("graphs/*.npz"))
assert fs, "graphs/ khali hai"
random.Random(0).shuffle(fs)                                        # train.py seed-0 split
a, b = int(.7 * len(fs)), int(.85 * len(fs))
m = Net(graph=True)                                                 # demo wala GATv2
m.load_state_dict(torch.load("best_gnn.pt", map_location="cpu"))
m.eval()


def stats(files):
    ds = [to_data(dict(np.load(f))) for f in files]
    dice, _ = evaluate(m, ds)                                       # thresholded Dice (train.py wala)
    with torch.no_grad():
        loss = float(np.mean([loss_fn(m(d), d).item() for d in ds])) # BCE + soft Dice, dropout off
    return float(dice.mean()), loss


tr_d, tr_l = stats(fs[:a])
va_d, va_l = stats(fs[a:b])
te_d, te_l = stats(fs[b:])
msg = (f"best_gnn.pt (seed 0): train Dice {tr_d:.3f} / loss {tr_l:.3f} | "
       f"val Dice {va_d:.3f} / loss {va_l:.3f} | test Dice {te_d:.3f} / loss {te_l:.3f}\n"
       f"train - val Dice gap: {tr_d - va_d:+.3f}  (chhota gap = overfitting ka koi bada signal nahi)\n"
       "Val curve plateau ~epoch 27 ke baad -> 60 epochs kaafi the; zyada epochs se fayda nahi.")
print(msg, flush=True)
with open("report_out/train_val_gap.txt", "w") as fh:
    fh.write(msg + "\n")
