# A HYBRID DEEP LEARNING MODEL AND DATASET FOR PADDOCK-LEVEL CROP YIELD ESTIMATION

This repository introduces a hybrid regression model for paddock-level crop yield prediction, combining MAMBA blocks, Transformer attention, and Slot Attention to capture spatial and temporal complexities. By leveraging soil, weather, and Sentinel-2 data, the model improves predictive accuracy. Additionally, the Western Australian (WA) Rainfall Paddocks Dataset, encompassing soil, climate, and satellite data for ~450,000 paddocks over three years, is presented as a resource for high-resolution and larg...

---

## 📁 Project Structure

```
A-HYBRID-DEEP-LEARNING-MODEL-AND-DATASET-FOR-CROP-YIELD-ESTIMATION/
├── dataset/
│   ├── training_data22.zip
│   └── testing_data22.csv
├── model/
│   └── Yield_Prediction_Model_Paddock_Level_Final.py
├── processing/
│   ├── soil_*.py
│   ├── Weather-*.ipynb
│   └── Script_For_Downloading_Sentinel-2-Raster.ipynb
├── shap_analysis/
│   └── SHAP-analysis-for-features-selection-yield-prediction.ipynb
├── README.md
├── requirements.txt
└── environment.yml
```

---

## 📊 Dataset Description

- Training and test data are organized in `/dataset`
- ~450,000 paddocks, 2020–2022
- 142 fused features: soil (119), weather (18), Sentinel-2 (5)

![Dataset Overview](https://github.com/IbrahimUWA/A-HYBRID-DEEP-LEARNING-MODEL-AND-DATASET-FOR-CROP-YIELD-ESTIMATION/assets/wa_dataset_figure.png)

---

## 🧠 Architecture Overview

- **Transformer Stream**: Captures long-range dependencies
- **MAMBA Stream**: Memory-efficient modeling
- **Slot Attention**: Dynamic spatial feature grouping

![Model Architecture](https://github.com/IbrahimUWA/A-HYBRID-DEEP-LEARNING-MODEL-AND-DATASET-FOR-CROP-YIELD-ESTIMATION/assets/model_architecture_figure.png)

---

## 🔍 SHAP Feature Importance

The SHAP analysis identifies the top 20 of 41 features for yield prediction:
- SOC, rainfall, red/NIR bands, and NDVI are dominant

![SHAP Summary Plot](https://github.com/IbrahimUWA/A-HYBRID-DEEP-LEARNING-MODEL-AND-DATASET-FOR-CROP-YIELD-ESTIMATION/assets/shap_summary_plot.png)

---

## ⚙️ Environment Setup

### ▶️ Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate crop-yield-env
```

### ▶️ Python + pip

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💡 Install PyTorch with CUDA

Follow instructions: https://pytorch.org/get-started/locally/

Example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Run the Model

```bash
cd model/
python Yield_Prediction_Model_Paddock_Level_Final.py
```

---

## 🧪 SHAP Analysis

```bash
cd shap_analysis/
jupyter notebook SHAP-analysis-for-features-selection-yield-prediction.ipynb
```

---

## 🌍 Run Preprocessing Scripts

- **Soil**: `soil_mean_point_extraction_paddock_level.py`
- **Weather**: `Weather-NC_to_Rasters-Generation.ipynb`
- **Sentinel-2**: `Extracting_Paddocks_Aggregation_Data_Python.py`

---

## 📈 Results

| Model               | R² Accuracy | MSE   | Inference Time (µs) |
|--------------------|-------------|-------|----------------------|
| SVR                | 81.69       | 0.230 | 7665.93              |
| ResNet50           | 84.21       | 0.210 | 65.10                |
| Random Forest      | 84.98       | 0.191 | 33.32                |
| **Hybrid Model**   | **86.43**   | **0.1788** | **58.69**        |

![Prediction Scatter](https://github.com/IbrahimUWA/A-HYBRID-DEEP-LEARNING-MODEL-AND-DATASET-FOR-CROP-YIELD-ESTIMATION/assets/prediction_performance.png)

---

## 📮 Contact

**Muhammad Ibrahim**  
Adjunct Research Fellow – UWA / DPIRD WA  
📧 muhammad.ibrahim@dpird.wa.gov.au
