# Single-Neuron Analysis for Stroop Task

## Overview
This repository contains the code and data for analyzing single-neuron activity from the amygdala (AMY), hippocampus (HIP), and orbitofrontal cortex (OFC) during a Stroop task. The study aims to uncover the neural mechanisms underlying cognitive control and conflict resolution by examining neuronal firing patterns and their correlation with behavioral responses.

## Repository Structure

```bash
single-neuron-analysis/
├── data_analysis/         # Scripts for data preprocessing and analysis
├── jupyter_notebooks/     # Interactive notebooks for exploratory analysis and explanations
├── utils/                 # Utility functions and helper scripts
└── README.md              # Project documentation

```
## Analysis Workflow

1. **Data Preprocessing**
   - Event alignment and trial segmentation

2. **Exploratory Analysis**
   - Firing rate visualizations  
   - Raster plots and peri-stimulus time histograms (PSTHs)
   - Permutation-based cluster analysis to identify periods of significant firing

3. **Statistical Testing**
   - Trial-wise comparisons (e.g., correct vs. error)  
   - Congruent vs. incongruent comparisons

4. **Classification/Decoding**
   - SVM-based decoding of trial types  
   - Cross-validation and accuracy calculation

## Visualizations

#### Correct/Error PSTH and Raster of a Hippocampal Neuron
![image](https://github.com/user-attachments/assets/9186f9ef-5f69-4da1-9d0f-f9143a6dc79e)

#### Correct/Error SVM Classification Over Button Press Window of Hippocampal Neurons
![image](https://github.com/user-attachments/assets/5d9d7ad8-18eb-4c43-89c9-cdc8fb132bb8)

### Correct/Error Heatmap of Likelihood Ratios during Button Press Window of Hippocampal Error Neurons
![image](https://github.com/user-attachments/assets/1f6abf20-6bcd-4702-9a60-6b20cb17263b)

### Correct/Error Differential Latency during Button Press Window of Hippocampal Error Neurons
![image](https://github.com/user-attachments/assets/77de3c07-b317-4dc2-bc09-cc1491777aad)

#### Correct/Error Temporal Generalization Matrix of Hippocampal Neurons
![image](https://github.com/user-attachments/assets/31780dd2-d9e0-48f8-8f38-84eb0f727fdc)




