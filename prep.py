"""MRI volume -> supervoxel graph. Run: python prep.py Task01_BrainTumour 150"""
import os, sys, glob                                                # os/sys/glob: files aur command-line args ke liye
import numpy as np                                                  # numpy: saara array maths
import nibabel as nib                                               # nibabel: .nii.gz MRI files padhne ke liye
from skimage.segmentation import slic                               # slic: brain ko chhote 3D tukdon (supervoxels) me todta hai


def build_graph(x, y=None, vox=1.0, n_seg=1500):                    # x: (X,Y,Z,4) MRI | y: tumor label (optional) | vox: 1 voxel ka volume mm3
    brain = x.any(-1)                                               # jahan koi bhi modality non-zero ho, wahi brain hai
    idx = np.argwhere(brain)                                        # brain wale voxels ke coordinates
    lo, hi = idx.min(0), idx.max(0) + 1                             # brain ka bounding box (khali jagah hatane ke liye)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))               # crop ke slices (app me wapas use honge)
    full = np.array(brain.shape)                                    # original size (centroid ko 0..1 me normalize karne ke liye)
    x, brain = x[crop].copy(), brain[crop]                          # sirf brain wala hissa rakho
    for c in range(4):                                              # 4 modalities: FLAIR, T1, T1ce, T2
        v = x[..., c][brain]                                        # is modality ke brain voxels
        x[..., c] = np.where(brain, (x[..., c] - v.mean()) / (v.std() + 1e-6), 0)  # z-score normalise (sirf brain ke andar)
    sv = slic(x, n_segments=n_seg, compactness=0.1, channel_axis=-1, mask=brain, start_label=1)  # 3D supervoxels, 0 = bahar
    ids = np.unique(sv[sv > 0])                                     # jo labels sach me bache
    remap = np.zeros(sv.max() + 1, np.int32)                        # purane label -> naya label table
    remap[ids] = np.arange(1, len(ids) + 1)                         # naye labels 1..N bina gap ke
    sv = remap[sv]                                                  # volume ko naye labels se likh do
    n = len(ids)                                                    # N = graph ke nodes
    f = sv.ravel()                                                  # flat labels (bincount ke liye)
    cnt = np.bincount(f, minlength=n + 1)[1:].astype(np.float32)    # har supervoxel me kitne voxels hain

    feats = []                                                      # node features ki list
    for c in range(4):                                              # har modality ke liye
        v = x[..., c].ravel()                                       # us modality ke flat intensities
        m = np.bincount(f, v, n + 1)[1:] / cnt                      # supervoxel ka mean intensity
        s = np.sqrt(np.maximum(np.bincount(f, v * v, n + 1)[1:] / cnt - m ** 2, 0))  # supervoxel ka std (variance ka root)
        feats += [m, s]                                             # 2 features per modality = 8
    for a, g in enumerate(np.indices(sv.shape)):                    # x, y, z coordinate grids
        feats.append((np.bincount(f, g.ravel(), n + 1)[1:] / cnt + lo[a]) / full[a])  # centroid position 0..1 (brain me kahan hai)
    feats.append(np.log(cnt))                                       # supervoxel ka size (log scale me)
    xn = np.stack(feats, 1).astype(np.float32)                      # (N, 12) feature matrix

    pairs = []                                                      # edges ki list
    for a in range(3):                                              # x, y, z teeno direction me
        s = np.moveaxis(sv, a, 0)                                   # is axis ko aage laao
        p, q = s[:-1].ravel(), s[1:].ravel()                        # padosi voxels ke jode (p, q)
        k = (p != q) & (p > 0) & (q > 0)                            # alag supervoxel aur dono brain ke andar = edge
        pairs.append(np.minimum(p, q)[k].astype(np.int64) * (n + 1) + np.maximum(p, q)[k])  # (chhota,bada) ko ek number me encode
    code = np.unique(np.concatenate(pairs))                         # duplicate edges hata do
    u, w = code // (n + 1) - 1, code % (n + 1) - 1                  # wapas decode, 0-based index
    ei = np.stack([np.r_[u, w], np.r_[w, u]])                       # dono direction = undirected graph

    g = dict(x=xn, ei=ei, cnt=cnt, vox=np.float32(vox))             # graph ka saara data
    if y is not None:                                               # training ke time label bhi chahiye
        y = y[crop]                                                 # label ko bhi usi tarah crop karo
        regions = [y > 0, (y == 2) | (y == 3), y == 3]              # WT=poora tumor, TC=tumor core, ET=enhancing (MSD: 1 edema, 2 non-enh, 3 enh)
        g["y"] = np.stack([np.bincount(f, r.ravel(), n + 1)[1:] / cnt for r in regions], 1).astype(np.float32)  # har node ka tumor fraction 0..1
    return g, sv, crop                                              # sv aur crop app me overlay ke kaam aate hain


if __name__ == "__main__":
    root = sys.argv[1]                                              # dataset folder
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 150              # shortcut: 150 cases (SLIC slow hai), pure 484 ke liye N badhao
    os.makedirs("graphs", exist_ok=True)                            # output folder
    files = sorted(glob.glob(f"{root}/imagesTr/BRATS_*.nii.gz"))[:N]   # BRATS_* se macOS ki ._ files skip ho jaati hain
    assert files, f"{root}/imagesTr me BRATS_*.nii.gz nahi mili, path check karo"  # khali match pe chup mat raho
    for fn in files:                                                    # har scan pe loop
        out = f"graphs/{os.path.basename(fn)[:-7]}.npz"             # output file ka naam
        if os.path.exists(out):                                     # pehle ban chuka hai to skip (resume support)
            continue
        img = nib.load(fn)                                          # MRI load
        vox = float(np.prod(img.header.get_zooms()[:3]))            # 1 voxel ka volume mm3 (header se)
        y = np.asarray(nib.load(fn.replace("imagesTr", "labelsTr")).dataobj).astype(np.int16)  # tumor label
        g, _, _ = build_graph(img.get_fdata(dtype=np.float32), y, vox)  # graph banao
        np.savez_compressed(out, **g)                               # disk pe save
        print(fn, g["x"].shape[0], "nodes", g["ei"].shape[1] // 2, "edges", flush=True)
