# **DPA-MSCL**
Because the Daqing Oilfield dataset is part of an ongoing project and has not yet been completed, the full dataset cannot be publicly released. Therefore, only a sample dataset is provided for demonstration and reproducibility purposes, while the experimental code related to the Daqing Oilfield dataset is retained.
- **Comparison of Confusion Matrices for Different Model Predictions on the Hugoton and Panoma fields**
- - **Lithologic visualization**

<div align="center">
  <img src="datasave/img/profile_W_A.png" width="450" />
  <img src="datasave/img/profile_W_B.png" width="450" />
</div>

- - **Comparison of Confusion Matrices**
<div align="center">
  <img src="datasave/img/Confusion_SGAN.png" width="450" />
  <img src="datasave/img/Confusion_DPA.png" width="450" />
</div>

- - **3D t-SNE visualization of the feature space on the Hugoton-Panoma dataset before and after representation learning**
<div align="center">
  <img src="datasave/img/T-SNE.png" width="800" />
</div>



## The operating environment of the project
-	Python == 3.9.21
- conda == 24.9.2
-	pandas == 2.0.3
-	numpy == 1.23.5
-	matplotlib == 3.7.1
-	seaborn == 0.13.2
-	scikit-learn == 1.5.0
- pickleshare == 0.7.5

## How to use this project？
The root directory contains scripts for lithology identification and cross-well lithology identification experiments on two datasets. Specifically, Experiments 1–4 correspond to: (1) lithology identification on the U.S. Hugoton–Panoma dataset, (2) cross-well lithology identification on the Hugoton–Panoma dataset, (3) lithology identification on the Daqing dataset, and (4) cross-well lithology identification on the Daqing dataset.
