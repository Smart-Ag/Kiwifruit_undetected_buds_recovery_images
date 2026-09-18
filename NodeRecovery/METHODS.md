# Included methods and scope

| Stage | Script(s) | Main role |
| --- | --- | --- |
| FC-level split | `split_field_data.py` | Keep all measurements from one cane in one partition |
| Synthetic masking | `get_ML_trainData.py` | Generate basal, internal, and apical gap labels/features |
| MGC selection | `get_ML_training5fold.py` | Five-fold grouped CV and held-out reports for four classifiers |
| Feature analysis | `evaluate_feature_importance.py` | Grouped ablation and held-out permutation importance |
| RGB-D projection | `maskMend.py`, `mask2pcd_projection.py`, `pixel2pcd_projection.py` | Mend cane mask and back-project cane, FPV, and detected-bud pixels |
| 3D reconstruction | `pcd_preprocessing.py`, `clean_branch_radial.py`, `l1_skeleton.py`, `denoise_depth.py` | Downsample, optionally clean, smooth depth, and create ordered skeleton |
| Geometry and GPLA | `get_bud_data.py`, `detect_buds_bulge.py` | Arc projection, diameter measurement, protrusion extraction |
| PNM/reconciliation | `recover_buds.py` | MGC flagging, Gaussian node count, GPLA reconciliation, output tables |
| Stage-wise evaluation | `evaluate_recovery_pipeline.py` | Visibility-defined gap counts and 30-mm node matching |

