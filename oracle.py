"""Supervoxel ceiling: model perfect ho to bhi Dice kitna mil sakta hai. Run: python oracle.py"""
import glob                                                         # graphs/ ki files dhundne ke liye
import numpy as np                                                  # numpy maths

D = []                                                              # har scan ka Dice
for f in sorted(glob.glob("graphs/*.npz")):                         # saare banaye hue graphs
    z = np.load(f); y, w = z["y"], z["cnt"][:, None]                # y = asli tumor fraction, w = node ke voxels
    b = (y > 0.5).astype(np.float32)                                # perfect model: fraction 0.5 se upar = tumor
    D.append((2 * (b * y * w).sum(0) + 1e-6) / ((b * w).sum(0) + (y * w).sum(0) + 1e-6))  # voxel-level Dice (train.py jaisa formula)
print(len(D), "scans | oracle Dice WT/TC/ET:", np.round(np.mean(D, 0), 3))  # ye is approach ki upper limit hai