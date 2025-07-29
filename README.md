# Tri-Sectional Lung Mask Generator (Rule-Based + Deep learning)

<p align="center"><img src='./MI2RL_logo.png' width="440" height="150"></p>

<br>
* Code Management : Youngjae Kim (provbs36@gmail.com)
<br>
<br>
* [MI2RL] (https://www.mi2rl.co/) @Asan Medical Center, South Korea

<h1>
Overall workflow of the proposed method
</h1>

<img width="1488" height="898" alt="image" src="https://github.com/user-attachments/assets/a8322f9f-ee23-454f-ad09-de97a69332d7" />

<br>
* Schematic representation of the pipeline using the rule based generated tri-sectional masks. 
<br>* The pipeline consists of (a) rule-based mask generation, and (b) deep-learning model training.
<br>
<br> * 3D tri-sectinal masks were generated with the rule-based mask generation method first. 
<br> * Then the generated masks were used for deep learning training for tri-sectional 3D mask generation.

<br>
<h3>
  Folder descriptions:
</h3>
1) <b>RB</b> : rule based mask generation code
<br> 2) <b>DL</b> : deep learing based mask generation code (simple implementation of original 3D nnUNet)
<br> The source code is available at: https://github.com/MIC-DKFZ/nnUNet
<br> * Additional utility codes (e.g., for computing ratios between regions) will be uploaded when necessary
