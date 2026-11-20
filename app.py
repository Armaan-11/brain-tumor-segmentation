"""Demo: streamlit run app.py   (pehle train.py chala ke best_gnn.pt bana lo)"""
import os, io, glob, tempfile                                       # upload/sample/download ke liye
import numpy as np, nibabel as nib, torch, streamlit as st          # basic libraries + UI
import matplotlib.pyplot as plt                                     # slice dikhane ke liye
from prep import build_graph                                        # MRI -> graph
from train import Net, to_data, mc_predict                          # model + helpers


@st.cache_resource                                                  # model ek hi baar load hoga
def load():
    m = Net(); m.load_state_dict(torch.load("best_gnn.pt")); return m


@st.cache_data(show_spinner="Supervoxel graph ban raha hai (~1 min)...")  # cache: slider hilane pe dobara SLIC na chale
def run(raw: bytes):
    p = os.path.join(tempfile.mkdtemp(), "scan.nii.gz")             # temp file (nibabel ko path chahiye)
    open(p, "wb").write(raw)                                        # upload ke bytes likho
    img = nib.load(p)                                               # MRI load
    x = img.get_fdata(dtype=np.float32)                             # (X,Y,Z,4) array
    vox = float(np.prod(img.header.get_zooms()[:3]))                # 1 voxel ka volume mm3
    g, sv, crop = build_graph(x, None, vox)                         # graph banao
    ps = mc_predict(load(), to_data(g)).numpy()                     # 20 stochastic predictions
    return x[crop][..., 0], sv, ps, g["cnt"], vox                   # FLAIR slice stack, supervoxels, predictions


st.title("Brain Tumor Segmentation + Area Affected (Supervoxel GNN)")
st.caption("Educational project, not a medical device.")
st.sidebar.markdown("**Model:** best_gnn.pt (GATv2, seed 0)  \n"
                    "**3-seed mean Dice:** 0.688 ± 0.038  \n"
                    "**Oracle ceiling:** 0.756")
f = st.file_uploader("MRI: 4-modality .nii.gz (MSD Task01 format)", type=["gz"])
samples = sorted(glob.glob("data/samples/*.nii.gz"))                # demo ke liye ready test scans
pick = st.selectbox("Ya koi sample test scan chuno", ["(none)"] + [os.path.basename(s) for s in samples])
if f:
    raw, label = f.getvalue(), f.name
elif pick != "(none)":
    raw = open(os.path.join("data/samples", pick), "rb").read(); label = pick
else:
    raw = None
if raw is not None:
    st.caption(f"Scan: {label}")
    flair, sv, ps, cnt, vox = run(raw)                              # heavy kaam (cached)
    vols = (ps * cnt[None, :, None]).sum(1) * vox / 1000            # (T,3) tumor volume cm3 har pass ke liye
    brain = cnt.sum() * vox / 1000                                  # poore brain ka volume cm3
    for c, name, i in zip(st.columns(3), ["Whole tumor", "Tumor core", "Enhancing"], range(3)):
        c.metric(name, f"{vols[:, i].mean():.1f} cm³",              # mean volume
                 f"±{vols[:, i].std():.1f} | {100 * vols[:, i].mean() / brain:.1f}% of brain", delta_color="off")  # uncertainty + % area
    z = st.slider("Slice", 0, sv.shape[2] - 1, sv.shape[2] // 2)    # kaunsi slice dekhni hai
    fig, ax = plt.subplots()
    ax.imshow(flair[:, :, z].T, cmap="gray", origin="lower"); ax.axis("off")  # FLAIR slice
    for i, col in enumerate(["red", "orange", "yellow"]):           # WT, TC, ET ke contour
        seg = np.r_[False, ps.mean(0)[:, i] > 0.5][sv[:, :, z]]     # node prediction ko voxels pe wapas paint
        if seg.any():                                               # khali slice pe contour nahi
            ax.contour(seg.T.astype(float), levels=[0.5], colors=col)
    st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150)     # screenshot/backup ke liye
    st.download_button("Overlay PNG save karo", buf.getvalue(), file_name="overlay.png", mime="image/png")
