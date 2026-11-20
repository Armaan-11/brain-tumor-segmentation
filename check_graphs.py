"""graphs/ ki saari .npz check karo, adhuri ya kharab file delete karo. Run: python check_graphs.py"""
import glob, os                                                     # files dhundne aur delete karne ke liye
import numpy as np                                                  # npz kholne ke liye

bad = 0                                                             # kharab files ki ginti
for f in sorted(glob.glob("graphs/*.npz")):                         # har graph file
    try:
        z = np.load(f)                                              # file kholo
        [z[k] for k in ("x", "ei", "cnt", "vox", "y")]              # har array padho (adhuri file yahan fail hogi)
    except Exception:                                               # koi bhi error = file kharab
        print("kharab, delete:", f); os.remove(f); bad += 1         # delete, prep.py isse dobara banayega
print(len(glob.glob("graphs/*.npz")), "sahi graphs |", bad, "kharab delete hue")  # summary
