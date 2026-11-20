"""graphs/ ko pehle N scans par freeze karo, baaki graphs_extra/ me. Run: python freeze_graphs.py 150"""
import glob, os, shutil, sys                                        # files dhundne aur move karne ke liye

N = int(sys.argv[1])                                                # kitne graphs rakhne hain
fs = sorted(glob.glob("graphs/*.npz"))                              # prep.py jaisi sorted order (BRATS_001, 002 ...)
os.makedirs("graphs_extra", exist_ok=True)                          # extra graphs ka folder
for f in fs[N:]:                                                    # N ke baad wale
    shutil.move(f, os.path.join("graphs_extra", os.path.basename(f)))  # move (delete nahi, baad me kaam aa sakte hain)
print(len(glob.glob("graphs/*.npz")), "graphs graphs/ me bache |", len(glob.glob("graphs_extra/*.npz")), "graphs_extra me")
