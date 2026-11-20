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
python test_smoke.py                       # 1. sanity check, no dataset needed
python prep.py Task01_BrainTumour 150      # 2. build graphs (~16 s/case on 1 CPU core)
python train.py                            # 3. GNN
python train.py --no_graph                 # 4. ablation: same model without edges
streamlit run app.py                       # 5. demo (upload a 4-channel .nii.gz)
```

## Results (fill in after your run)
| Model | Dice WT | Dice TC | Dice ET | Volume MAE (cm³) |
|---|---|---|---|---|
| No graph (ablation) | | | | |
| Supervoxel GATv2 | | | | |

## Related work
- Saueressig et al., *Exploring Graph-Based Neural Networks for Automatic Brain Tumor Segmentation* (BraTS 2019)
- *A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation* (BraTS 2021, arXiv 2109.05580)
- Protani et al., *Decoder-Free Supervoxel GNN for Accurate Brain-Tumor Localization* (arXiv 2601.14055)
- Antonelli et al., *The Medical Segmentation Decathlon* (dataset)

Educational project, not a medical device.
