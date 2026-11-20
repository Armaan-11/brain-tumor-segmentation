# Brain Tumor Segmentation + Tumor Volume with a Supervoxel Graph Neural Network

Brain MRI -> supervoxel graph -> GATv2 -> per-node tumor fraction for 3 nested regions
(whole tumor, tumor core, enhancing) -> segmentation **and** tumor volume (cm³, % of brain)
with MC-Dropout uncertainty.

## Data 
Medical Segmentation Decathlon, Task01_BrainTumour (484 labelled BraTS 2016/17 cases, CC-BY-SA 4.0).
```
curl -L -O https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xf Task01_BrainTumour.tar
```

## Run
```
pip install -r requirements.txt
python test_smoke.py                       # sanity check, no dataset needed
python prep.py Task01_BrainTumour 150      # build graphs (150 of 484 cases used)
python train.py                            # GNN
python train.py --no_graph                 # ablation: same model, edges removed
python oracle.py                           # supervoxel ceiling
python baselines.py                        # RandomForest + SVM
python report.py best_gnn.pt               # metrics, plots, worst scans -> report_out/
```
Demo (Windows): double-click `run_demo.bat`, or
`C:\Python314\python.exe -m streamlit run app.py` (same Python that has torch/nibabel).
Then pick a sample scan in the dropdown, move the slice slider, save the overlay PNG.

## Results
150 MSD cases, patient-level 70/15/15 split (~23 test scans).
Three seeds; values are mean ± std over seeds (seed 0 checkpoint used by the demo).

| Model | Dice WT | Dice TC | Dice ET | Mean Dice | Vol MAE WT/TC/ET (cm³) |
|---|---|---|---|---|---|
| No graph (MLP) | 0.740 ± 0.027 | 0.613 ± 0.058 | 0.522 ± 0.079 | 0.625 ± 0.054 | 22.1 / 13.0 / 4.8 |
| Supervoxel GATv2 | **0.805 ± 0.027** | **0.669 ± 0.067** | **0.589 ± 0.031** | **0.688 ± 0.038** | **15.9 / 12.1 / 5.2** |
| RandomForest (seed 0) | 0.730 | 0.515 | 0.475 | 0.573 | 23.0 / 12.0 / 4.2 |
| SVM-RBF SVR (seed 0, 20k nodes) | 0.711 | 0.477 | 0.404 | 0.531 | 28.2 / 27.8 / 24.8 |
| Oracle ceiling (seed 0) | 0.859 | 0.775 | 0.634 | 0.756 | — |

Graph improves mean Dice by **+0.063 ± 0.022**, positive in all 3 paired seeds.
Oracle analysis: WT is near the supervoxel ceiling; TC has the largest learnable gap;
ET is inherently limited by the supervoxel partition. Full 484-case training and
external validation are future work. `logs/`, `report_out/` and `data/samples/`
contain training logs, metrics/plots and demo scans. Educational project, not a medical device.

## Related work
- Saueressig et al., *Exploring Graph-Based Neural Networks for Automatic Brain Tumor Segmentation* (BraTS 2019)
- *A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation* (BraTS 2021, arXiv 2109.05580)
- Protani et al., *Decoder-Free Supervoxel GNN for Accurate Brain-Tumor Localization* (arXiv 2601.14055)
- Antonelli et al., *The Medical Segmentation Decathlon* (dataset)

Educational project, not a medical device.
