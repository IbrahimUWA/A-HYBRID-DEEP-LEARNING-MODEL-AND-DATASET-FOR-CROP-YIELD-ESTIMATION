# A SPATIAL HYBRID MODEL FOR CROP YIELD PREDICTION IN WESTERN AUSTRALIA

This repository introduces a hybrid regression model for paddock-level crop yield prediction, combining MAMBA blocks, Transformer attention, and Slot Attention to capture spatial and temporal complexities. By leveraging soil, weather, and Sentinel-2 data, the model improves predictive accuracy. Additionally, the Western Australian (WA) Rainfall Paddocks Dataset, encompassing soil, climate, and satellite data for ~450,000 paddocks over three years, is presented as a resource for high-resolution and large-scale modeling. The proposed model outperforms classical machine learning and ResNet50 in accuracy and inference speed, establishing itself as an efficient solution for precision agriculture.

---

## 🎤 Conference Presentation

Our research paper based on this repository has been **accepted for oral presentation at [IGARSS 2025](https://www.2025.ieeeigarss.org/)**.

📍 **Conference:** The 45th IEEE International Geoscience and Remote Sensing Symposium (IGARSS)  
📅 **Dates:** 3–8 August 2025  
🌍 **Location:** Brisbane, Australia

> IGARSS is the flagship event of the IEEE Geoscience and Remote Sensing Society (GRSS), bringing together scientists, engineers, and industry leaders working in Earth observation and remote sensing technologies.

---

## 📁 Project Structure

```
A-HYBRID-DEEP-LEARNING-MODEL-AND-DATASET-FOR-CROP-YIELD-ESTIMATION/
├── dataset/
│   ├── training_data22.zip
│   └── testing_data22.csv
├── assets/
│ ├── wa_dataset_figure.png
│ ├── model_architecture_figure.png
│ ├── shap_summary_plot.png
│ ├── prediction_performance.PNG
├── model/
│   └── Yield_Prediction_Model_Paddock_Level_Final.py
├── processing/
│   ├── soil_*.py
│   ├── Weather-*.ipynb
│   └── Script_For_Downloading_Sentinel-2-Raster.ipynb
├── shap_analysis/
│   └── SHAP-analysis-for-features-selection-yield-prediction.ipynb
├── Final_paper_accepted_version_IGARRS 2025.pdf
├── README.md
└── environment.yml
├── requirements.txt

```

---

## 📄 Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Architecture Overview](#architecture-overview)
- [SHAP Analysis and Features Importance](#shap-analysis-and-features-importance)
- [Environment Setup](#environment-setup)
- [Run the Model](#run-the-model)
- [Run Preprocessing Scripts](#run-preprocessing-scripts)
- [Results and Discussion](#results-and-discussion)

---

## 📄 Introduction

This research proposes a novel hybrid regression model for paddock-level crop yield prediction, integrating MAMBA blocks, Transformer attention mechanisms, and Slot Attention. This architecture effectively captures spatial and temporal complexities, leveraging diverse data such as soil characteristics, weather patterns, and Sentinel-2 data to enhance predictive accuracy.

We also introduce the Western Australian (WA) Rainfall Paddocks Dataset, a comprehensive resource comprising soil, climate, and satellite data for ~450,000 WA paddocks over three years. Evaluation against classical machine learning models and ResNet50 demonstrates significant improvements in both accuracy and speed.

---

## 📊 Dataset Description

- Training and test data are organized in `/dataset`
-  `training_data22.zip`: Training samples with soil, climate, and remote sensing features.
- `testing_data22.csv`: Test set (temporally or spatially held out).
- Target: `yield` column (tons per hectare).
-  ~450,000 paddocks, 2020–2022
- 142 fused features: soil (119), weather (18), Sentinel-2 (5)

All paddocks are geo-tagged and labeled with rainfall zone, year, and crop type.

![Dataset Overview](assets/wa_dataset_figure.png)


---

## 🧠 Architecture Overview

- **Transformer Stream**: Captures long-range dependencies
- **MAMBA Stream**: Memory-efficient modeling
- **Slot Attention**: Dynamic spatial feature grouping

![Model Architecture](assets/model_architecture_figure.png)
```bash
Implemented in: `Yield_Prediction_Model_Paddock_Level_Final.py`
```
---

## 🧪 SHAP Analysis and Feature Importance
The SHAP analysis identifies the top 20 of 41 features for yield prediction:
- SOC, rainfall, red/NIR bands, and NDVI are dominant
  
```bash
cd shap_analysis/
jupyter notebook SHAP-analysis-for-features-selection-yield-prediction.ipynb
```
Notebook steps:
- Load training dataset
- Train RandomForestRegressor
- Generate SHAP values
- Plot feature importance and dependence plots

![SHAP Summary Plot](assets/shap_summary_plot.png)
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

This will:
- Load CSVs from `dataset/`
- Train the hybrid model
- Save best model to `best_transformer_mamba_slot_attention_model.pth`
- Plot and save test prediction results

To run test predictions only:
```python
from Yield_Prediction_Model_Paddock_Level_Final import test_on_test_data

# Define your paths
model_path = "best_transformer_mamba_slot_attention_model.pth"
test_data_path = "../dataset/testing_data22.csv"
output_path = "model_output_plot.png"
test_on_test_data(test_data_path, model_path, output_path)
```

---


## 🌍 Run Preprocessing Scripts
### Folder: `processing/`

#### ⚡ Soil Data
- `soil_mean_point_extraction_paddock_level.py`: Computes mean per paddock per raster.
- `soil_mean_median_point_extraction_paddock_level.py`: Adds median/mode stats.
- `soil_NC_files_to_rasters_conversion_updated.py`: Converts NetCDF to GeoTIFF rasters.

Run like:
```bash
python soil_mean_point_extraction_paddock_level.py
```
#### ☁️ Weather Data
- `Weather-NC_to_Rasters-Generation.ipynb`: Converts NetCDF to rasters by year.
- `Weather-Point Data-Extraction.ipynb`: Extracts raster values per paddock.

Run in Jupyter:
```bash
jupyter notebook Weather-NC_to_Rasters-Generation.ipynb
```
#### 🌍 Sentinel-2
- `Script_For_Downloading_Sentinel-2-Raster.ipynb`: (Optional) automate data download
- `Extracting_Paddocks_Aggregation_Data_Python.py`: Computes median NDVI, bands

Run like:
```bash
python Extracting_Paddocks_Aggregation_Data_Python.py
```
Each script requires:
- Paddock boundary GeoPackage with year info
- Raster folder (tiled .tif files per feature/year)

---

## 📈 Results and Discussion

| Model               | R² Accuracy | MSE   | Inference Time (µs) |
|--------------------|-------------|-------|----------------------|
| SVR                | 81.69       | 0.230 | 7665.93              |
| ResNet50           | 84.21       | 0.210 | 65.10                |
| Random Forest      | 84.98       | 0.191 | 33.32                |
| **Hybrid Model**   | **86.43**   | **0.1788** | **58.69**        |

![Prediction Scatter](assets/prediction_performance.PNG)

- **+6.5% gain in R²** over ResNet50
- **40% faster inference** compared to deep CNNs
- SHAP shows top features: NDVI, rainfall, Soil Organic Carbon, Albedo, and Silt
- Excellent generalization across rainfall zones and seasons
---


## 📮 Contact

**Muhammad Ibrahim**  
Research Scientist DPIRD WA and Adjunct Research Fellow – UWA 
📧 muhammad.ibrahim@dpird.wa.gov.au

