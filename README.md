# 🫁 Tri-Sectional Lung Mask Generator (Rule-Based + Deep Learning)

<p align="center">
  <img src="./MI2RL_logo.png" width="440" height="150">
</p>

---

- **Code Management**: Youngjae Kim (provbs36@gmail.com)  
- **Affiliation**: [MI2RL](https://www.mi2rl.co/) @ Asan Medical Center, South Korea  

---

## 📌 Overview

This project combines rule-based and deep learning approaches to generate **tri-sectional lung masks** from 3D Chest CT images.

### 🧠 Workflow Summary

<p align="center">
  <img width="100%" alt="workflow" src="https://github.com/user-attachments/assets/a8322f9f-ee23-454f-ad09-de97a69332d7" />
</p>

- The pipeline consists of:
  - **(a)** Rule-based mask generation
  - **(b)** Deep learning-based mask training

- In **Step 1**, 3D tri-sectional masks are generated using a rule-based method.  
- In **Step 2**, these masks are used to train a deep learning model for automated tri-sectional 3D mask generation.

---

## 📁 Folder Structure

1. **`RB/`** – Rule-based mask generation code  
2. **`DL/`** – Deep learning-based mask generation (simple implementation based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet))

> 🔄 The latest version of the nnUNet code used here is available at:  
> https://github.com/MIC-DKFZ/nnUNet

- Additional utility scripts (e.g., for computing regional ratios) will be uploaded as needed.

---

