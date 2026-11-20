"""GATv2 on supervoxel graph. Run: python train.py   (ablation: python train.py --no_graph)"""
import sys, glob, random                                            # sys: command-line flag | glob: files | random: split
import numpy as np                                                  # npz files load karne ke liye
import torch, torch.nn as nn, torch.nn.functional as F              # PyTorch basics
from torch_geometric.data import Data                               # graph ka container
from torch_geometric.nn import GATv2Conv                            # attention wala graph layer
from torch_geometric.loader import DataLoader                       # graphs ko batch me jodta hai


def to_data(g):                                                     # prep.py ka numpy dict -> PyG Data
    t = {k: torch.as_tensor(v) for k, v in g.items()}               # sab arrays ko tensor banao
    return Data(x=t["x"], edge_index=t["ei"], y=t.get("y"), cnt=t["cnt"], vox=t["vox"])  # y nahi ho to None (app me)


class Net(nn.Module):
    def __init__(self, h=64, graph=True):
        super().__init__()
        self.graph = graph                                          # False = ablation: edges hata do, graph ka fayda naapne ke liye
        self.inp = nn.Linear(12, h)                                 # 12 node features -> 64
        self.convs = nn.ModuleList(GATv2Conv(h, h // 4, heads=4) for _ in range(3))  # 3 attention layers (4 heads x 16 = 64)
        self.out = nn.Linear(h, 3)                                  # 3 outputs: WT, TC, ET fraction

    def forward(self, d):
        ei = d.edge_index if self.graph else d.edge_index[:, :0]    # ablation me khali edge list (sirf self-loop bachte hain)
        x = F.elu(self.inp(d.x))                                    # pehle features ko hidden size me laao
        for c in self.convs:                                        # har layer pe padosi supervoxels se info lo
            x = x + F.elu(c(F.dropout(x, 0.2, self.training), ei))  # residual connection + dropout
        return self.out(F.dropout(x, 0.2, self.training))           # logits (sigmoid baad me)


def loss_fn(logit, d):
    p, w = torch.sigmoid(logit), d.cnt[:, None]                     # p = predicted fraction, w = node ke voxels (bade node ka zyada weight)
    dice = 1 - (2 * (p * d.y * w).sum(0) + 1) / ((p * w).sum(0) + (d.y * w).sum(0) + 1)  # soft Dice loss, har region ka
    return F.binary_cross_entropy_with_logits(logit, d.y) + dice.mean()  # BCE (fraction regression) + Dice (overlap + volume)


@torch.no_grad()
def evaluate(m, ds):                                                # har patient ke liye Dice aur volume error
    m.eval()                                                        # dropout band
    D, E = [], []
    for d in ds:
        p, w = torch.sigmoid(m(d)), d.cnt[:, None]                  # prediction aur node sizes
        b = (p > 0.5).float()                                       # node ko tumor / non-tumor banao
        D.append((2 * (b * d.y * w).sum(0) + 1e-6) / ((b * w).sum(0) + (d.y * w).sum(0) + 1e-6))  # voxel-level Dice (node ko voxels pe paint karne ke barabar)
        E.append(((p - d.y) * w).sum(0).abs() * d.vox / 1000)       # |predicted - asli tumor volume| in cm3
    return torch.stack(D).mean(0), torch.stack(E).mean(0)


@torch.no_grad()
def mc_predict(m, d, T=20):                                         # MC-Dropout: uncertainty ke liye
    m.train()                                                       # dropout ON rakho, har pass thoda alag aayega
    return torch.stack([torch.sigmoid(m(d)) for _ in range(T)])     # (T, N, 3)


if __name__ == "__main__":
    graph = "--no_graph" not in sys.argv                            # flag se ablation chalao
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0  # command se seed lo, nahi diya to 0
    torch.manual_seed(seed)                                         # model ki shuruaati weights is seed se fix
    name = ("gnn" if graph else "mlp") + (f"_s{seed}" if seed else "")  # seed 0 ka naam pehle jaisa, baaki me _s1, _s2                                # checkpoint ka naam
    ds = [to_data(dict(np.load(f))) for f in sorted(glob.glob("graphs/*.npz"))]  # saare graphs load
    assert len(ds) >= 5, f"graphs/ me sirf {len(ds)} files hain, pehle prep.py chalao"  # khali train set se pehle saaf error
    random.Random(seed).shuffle(ds)                                 # split bhi seed ke hisab se badlega (har case ek alag patient hai)                                   # fixed seed shuffle (har case ek alag patient hai)
    a, b = int(.7 * len(ds)), int(.85 * len(ds))                    # 70 / 15 / 15 split
    tr, va, te = ds[:a], ds[a:b], ds[b:]                            # train / val / test
    m = Net(graph=graph)                                            # model
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)  # optimizer
    best = 0                                                        # best val Dice
    for ep in range(60):                                            # shortcut: 60 epochs fixed, early stopping nahi
        m.train()
        for d in DataLoader(tr, batch_size=4, shuffle=True):        # 4 graphs ek saath
            opt.zero_grad(); loss_fn(m(d), d).backward(); opt.step()  # standard training step
        dice, _ = evaluate(m, va)                                   # validation Dice
        if dice.mean() > best:                                      # behtar mila to save
            best = dice.mean(); torch.save(m.state_dict(), f"best_{name}.pt")
        print(ep, f"val Dice {dice.mean():.3f}", flush=True)
    m.load_state_dict(torch.load(f"best_{name}.pt"))                # best checkpoint wapas
    dice, err = evaluate(m, te)                                     # final test (sirf ek baar)
    print(f"{name} TEST Dice WT/TC/ET:", [round(v, 3) for v in dice.tolist()],
          "| volume MAE cm3:", [round(v, 1) for v in err.tolist()])
