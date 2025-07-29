# 📜 Tri-Sectional Lung Mask Generator (Rule-Based)

<p align="center">
  <img src="./MI2RL_logo.png" width="440" height="150">
</p>

---
 
- **Affiliation**: [MI2RL](https://www.mi2rl.co/) @ Asan Medical Center, South Korea  

---

## 📌 Overview

This project presents a **rule-based method** for generating **tri-sectional lung masks** from 3D Chest CT images.

### 🧠 Workflow Summary

<p align="center">
  <img width="100%" alt="workflow" src="https://github.com/user-attachments/assets/a8322f9f-ee23-454f-ad09-de97a69332d7" />
</p>

- The pipeline currently consists of:
  - **(a)** Rule-based mask generation ✔️

- In **Step 1**, 3D tri-sectional masks are generated using heuristic anatomical rules.  
- These masks can serve as **training labels or priors** for more advanced models.

> 🔭 **Future Development**:  
> Step **(b)** involves extending this work toward a deep learning-based mask generation model,  
> trained using the rule-based masks as supervision. This would enable automated and scalable tri-sectional labeling from raw CT volumes.

---

## 📁 Folder Structure

- **`RB/`** – Rule-based mask generation code

> 🛠 Additional utility scripts (e.g., for computing regional ratios) will be uploaded as needed.

---

## 📄 Related Publications

If this code is used in a publication, please consider citing the appropriate work.  
Below is a list of related publications that have used or contributed to this repository:

- *(To be updated)*

<!-- Example format:
- **Kim YJ**, et al. "Title of the paper." *Journal Name*, Year. [DOI or arXiv link]
-->
