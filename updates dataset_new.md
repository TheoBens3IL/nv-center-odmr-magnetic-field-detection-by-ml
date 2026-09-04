`dataset_new_1` :
- new setup (new coil, new antenna)
- 2109 samples unhomogenous
- current labels between [-2.2 ; 2.2] A

`dataset_new_2`: 
- decreasing current labels range between [-1.1 ; 1.1] A

`dataset_new_3` :
- adjusting current to remove magnetic field offset
- laser beam more convergent (more light on the diamond)

`dataset_new_4` :
- reducing mw intensities to have finest peaks (frequency of each peak more precise)
- current labels between [-1.25 ; 1.25] A

`dataset_new_5` :
- tilting the antenna (22/22°) to align well with NV axis and better observe them
- adjusting MW configurations according to best mw configs found with **find_best_mw_configs.py**

`dataset_new_6` :
- increasing the number of frequency points to 401 (instead of 201)

`dataset_new_7` :
- increasing the number of samples (4169 configs)


| Step / Dataset | Changes | MAE mean (A) | MAE Ax (A) | MAE Ay (A) | MAE Az (A) | Best model |
|---|---|---:|---:|---:|---:|---|
| **1** / `dataset_new_1` | new setup (new coil, new antenna); 2109 samples unhomogenous; current labels between [-2.2 ; 2.2] A | 0.0410 | 0.0462 | 0.0411 | 0.0357 | ZoneAwareTwoStageJoint |
| **2** / `dataset_new_2` | decreasing current labels range between [-1.1 ; 1.1] A | 0.0196 | 0.0218 | 0.0209 | 0.0162 | ZoneAwareTwoStageJointDeep |
| **3** / `dataset_new_3` | adjusting current to remove magnetic field offset; laser beam more convergent (more light on the diamond) | 0.1065 | 0.1564 | 0.1399 | 0.0233 | ZoneAwareTwoStageJointDeep |
| **4** / `dataset_new_4` | reducing mw intensities to have finest peaks (frequency of each peak more precise); current labels between [-1.25 ; 1.25] A | 0.0399 | 0.0522 | 0.0474 | 0.0201 | ZoneAwareTwoStageJoint |
| **5** / `dataset_new_5 (dataset_new_5_IQ_0,8)` | tilting the antenna (22/22°) to align well with NV axis and better observe them; adjusting MW configurations according to best mw configs found with **find_best_mw_configs.py** | 0.0133 | 0.0129 | 0.0143 | 0.0127 | ZoneAwareTwoStageJointDeep |
| **6** / `dataset_new_6` | increasing the number of frequency points to 401 (instead of 201) | 0.0136 | 0.0134 | 0.0124 | 0.0151 | ZoneAwareTwoStageJointDeep |
| **7** / `dataset_new_7` | increasing the number of samples (4169 configs) | 0.0127 | 0.0123 | 0.0122 | 0.0134 | ZoneAwareTwoStageJointDeep |

