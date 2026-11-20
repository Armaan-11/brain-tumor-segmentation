"""Dataset ke bina sanity check: python test_smoke.py"""
import numpy as np                                                  # synthetic data banane ke liye
from prep import build_graph                                        # test karna hai


def fake():                                                         # nakli brain: gola + andar nested tumor
    r = np.random.default_rng(0)
    g = np.indices((64, 64, 48)).transpose(1, 2, 3, 0) - np.array([32, 32, 24])  # center se offset
    d = np.sqrt((g ** 2).sum(-1))                                   # center se doori
    y = np.zeros(d.shape, np.int16)                                 # label volume
    t = np.sqrt(((g - [8, 4, 0]) ** 2).sum(-1))                     # tumor center se doori
    y[t < 10] = 1; y[t < 6] = 2; y[t < 3] = 3                       # edema > non-enh > enh (nested)
    x = (r.normal(0, 1, d.shape + (4,)) + (y > 0)[..., None] * 3) * (d < 28)[..., None]  # tumor chamakdaar, brain ke bahar 0
    return x.astype(np.float32) + 0.01 * (d < 28)[..., None], y


def test_prep():
    x, y = fake()
    g, sv, crop = build_graph(x, y, vox=1.0, n_seg=100)
    assert g["x"].shape[1] == 12 and g["y"].shape == (len(g["x"]), 3)  # 12 features, 3 targets
    assert abs(g["cnt"].sum() - sv.astype(bool).sum()) < 1          # node sizes = brain voxels
    assert abs(g["y"][:, 0] @ g["cnt"] - (y[crop] > 0).sum()) < 1   # tumor voxels bachte hain (kuch kho nahi)
    e = set(map(tuple, g["ei"].T)); assert all((b, a) in e for a, b in e)  # graph undirected hai
    return g


def test_model():
    import torch; from train import Net, to_data, loss_fn           # torch yahin import (prep bina torch ke chalta hai)
    d = to_data(test_prep())
    for graph in (True, False):                                     # GNN aur no-graph dono
        out = Net(graph=graph)(d); assert out.shape == (len(d.x), 3)
    loss_fn(out, d).backward()                                      # loss + backward chalta hai


if __name__ == "__main__":
    test_prep(); print("prep OK")
    test_model(); print("model OK")
