# A HYBRID DEEP LEARNING MODEL AND DATASET FOR PADDOCK-LEVEL CROP YIELD ESTIMATION
This research proposes a novel hybrid regression model for paddock-level crop yield prediction, integrating MAMBA blocks, Transformer attention mechanisms, and Slot Attention. This architecture effectively captures spatial and temporal complexities, leveraging diverse data such as soil characteristics, weather patterns, and Sentinel-2 data to enhance predictive accuracy. We also introduce the Western Australian (WA) Rainfall Paddocks Dataset, a comprehensive resource comprising soil, climate, and satellite data for ~450,000 WA paddocks over three years. The dataset supports both high-resolution and large-scale modeling, facilitating diverse research applications. Evaluation against classical machine learning models and ResNet50 demonstrates that our hybrid model significantly improves accuracy while achieving faster inference speeds compared to ResNet50 and some classical models. The results establish the proposed method as a robust and efficient solution for precision agriculture

## Table of Contents
- [A HYBRID DEEP LEARNING MODEL AND DATASET FOR PADDOCK-LEVEL CROP YIELD ESTIMATION](A-HYBRID-DEEP-LEARNING-MODEL-AND-DATASET-FOR-PADDOCK-LEVEL-CROP-YIELD-ESTIMATION)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Dataset Description](#Dataset-Description)
    - [1.Study Site](#1-Study-Site)
    - [2.WA Rainfall Paddocks Dataset](#2-WA-Rainfall-Paddocks-Dataset)
    - [3.SHAP Analysis](#3-SHAP-Analysis)
- [Architecture Overview](#architecture-overview)
    - [1. Transformer Stream](#1-Transformer-stream)
    - [2. MAMBA Stream](#2-MAMBA-stream)
    - [3. Slot Attention Block](#3-Slot-Attention-Block)
