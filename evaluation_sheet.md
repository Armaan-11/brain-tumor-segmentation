# Evaluation sheet (fill weights/marks from the actual sheet, then paste this text to me)

| FAQ Q | Criterion | What panel checks | Our evidence in project | Weight % | Self score /10 |
|---|---|---|---|---|---|
| Q3/Q4 | PPT + timing | 5–8 slides, 8 min demo + 4 min viva | 8-slide outline ready; `run_demo.bat`, sample scans | | |
| Q8/Q24 | Full story | dataset, split, architecture, training, metrics, baseline, errors, honest interpretation | README final table, oracle, ablation, limitations | | |
| Q9 | Maths | loss, gradient, backprop, optimizer, activation | BCE + soft Dice, AdamW, ELU, attention | | |
| Q10 | Architecture | why suitable, strengths, limits | SLIC graph → GATv2; neighbour context; boundary limit | | |
| Q11/Q19 | Understanding + AI disclosure | every critical part explained; AI help declared | Hinglish comments, viva prep, disclosure slide | | |
| Q12 | Scope discipline | complexity for its own sake not rewarded | 150 cases, 6 models; 484/extra datasets = future work | | |
| Q13 | Baseline/comparison | justified comparison | no-graph + RF + SVM + oracle (+ optional GCN/SAGE) | | |
| Q14 | Metrics | explain, not just values | Dice=F1, precision, recall, vol MAE/RMSE/R² | | |
| Q15 | Error analysis | evidence + reasons for poor/overfit | `report_out/`: worst-5 scans, scatter, size-vs-Dice | | |
| Q16 | Data hygiene | source, split, leakage | MSD Task01, patient-level 70/15/15, no overlap | | |
| Q18 | Unseen input | live demo on new scan | `app.py` sample dropdown + upload | | |
| Q21/Q22 | Artefacts + backup | weights, logs, backup | `*.pt`, `logs/`, screenshots, `overlay.png` | | |
| Q23 | General DL basics | neuron, backprop, overfitting, CNN idea | viva practice | | |

**Isme weight/marks bharne ke liye college ki actual sheet chahiye.** Usse PDF/image ke bajaye **text** me paste karo — main is model me PDF ya image nahi padh pa raha.
