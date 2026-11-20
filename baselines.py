"""Classical ML baselines (same node features, same split as train.py seed 0). Run: python baselines.py"""
import glob, random                                                 # glob: graph files | random: train.py jaisa shuffle
import numpy as np                                                  # numpy maths
from sklearn.ensemble import RandomForestRegressor                  # Random Forest (trees ka jungle)
from sklearn.svm import SVR                                         # SVM ka regression version (fraction predict karna hai)
from sklearn.multioutput import MultiOutputRegressor                # SVR ek output deta hai, 3 outputs (WT, TC, ET) ke liye wrapper
from sklearn.pipeline import make_pipeline                          # scaling + model ek saath
from sklearn.preprocessing import StandardScaler                    # features ko same scale pe laata hai (SVM ke liye zaroori)

fs = sorted(glob.glob("graphs/*.npz")); random.Random(0).shuffle(fs)  # train.py (seed 0) jaisa hi shuffle, to split same rahega
a, b = int(.7 * len(fs)), int(.85 * len(fs))                        # 70 / 15 / 15 split
tr = [dict(np.load(f)) for f in fs[:a]]; te = [dict(np.load(f)) for f in fs[b:]]  # train aur test graphs (val yahan use nahi hota)
X = np.concatenate([g["x"] for g in tr]); Y = np.concatenate([g["y"] for g in tr])  # saare train nodes ek badi table me (node = ek row)
i = np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)  # shortcut: SVR slow hai, isliye 20k nodes ka sample

models = {
    "RandomForest": (RandomForestRegressor(100, min_samples_leaf=5, n_jobs=-1, random_state=0), slice(None)),  # RF poore train data par
    "SVM (RBF SVR)": (make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=1.0))), i),  # SVR sirf sample par
}
for name, (m, rows) in models.items():
    m.fit(X[rows], Y[rows])                                         # train
    D, E = [], []                                                   # har test scan ka Dice aur volume error
    for g in te:
        p = np.clip(m.predict(g["x"]), 0, 1); w = g["cnt"][:, None]  # predicted fraction (0..1 me), node ke voxels
        bn = (p > 0.5).astype(np.float32); y = g["y"]               # 0.5 se upar = tumor
        D.append((2 * (bn * y * w).sum(0) + 1e-6) / ((bn * w).sum(0) + (y * w).sum(0) + 1e-6))  # Dice (train.py wala formula)
        E.append(np.abs(((p - y) * w).sum(0)) * float(g["vox"]) / 1000)  # volume error cm3
    print(f"{name} TEST Dice WT/TC/ET:", [round(float(v), 3) for v in np.mean(D, 0)],
          "| volume MAE cm3:", [round(float(v), 1) for v in np.mean(E, 0)], flush=True)