# Quantum Neural Network–Assisted Deep Learning for Central Serous Retinopathy (CSCR) Detection

> [!IMPORTANT]
> **Pretrained QNN & Segmentation Model Weights Required!**
> Due to GitHub's file size limits, the trained model files (`.h5` and `.pth`) are hosted in the Releases section.
> place them in the main folder before running `app.py`.

An automated, interpretable, multimodal diagnostic web application for detecting **Central Serous Chorioretinopathy (CSCR / CSR)** from both **Fundus photographs** and **Optical Coherence Tomography (OCT)** scans. The framework integrates classical deep learning backbones, a **4-qubit Variational Quantum Circuit (VQC)** built with **PennyLane**, macular segmentation, and **Grad-CAM** visual explainability.

---

## Key Features

- **Multimodal Retinal Analysis:** Supports dual-modality screening—classifying **Fundus images** (`CSCR` vs. `Healthy`) and **OCT scans** (`Abnormal` vs. `Normal`) within a unified pipeline.
- **Hybrid Quantum-Classical Learning:** Encodes compact classical feature vectors into a **4-qubit parameterized quantum circuit** using **Angle Embedding (`RY` rotation gates)** and **`StronglyEntanglingLayers`**, measuring Pauli-Z expectation values to fuse quantum and classical representations.
- **Modality-Specific Preprocessing:**
  - *Fundus:* Green-channel extraction, Contrast Limited Adaptive Histogram Equalization (**CLAHE**), and noise filtering to highlight macular and vascular structures.
  - *OCT:* Grayscale conversion, intensity normalization, and Gaussian smoothing to suppress speckle noise and sharpen retinal layer boundaries.
- **Explainable AI (Grad-CAM):** Generates gradient-weighted class activation heatmaps overlaid on original scans so clinicians can verify model focus on subretinal fluid and macular abnormalities.
- **Automated Macular Segmentation:** Localizes pathological macular regions on abnormal fundus images using a convolutional segmentation network with thresholding and largest-connected-component post-processing.
- **Real-Time Web Interface:** Interactive web app (`app.py`) for image upload, real-time inference (< 1 second per scan), confidence scoring, heatmap/mask visualization, and result downloads.

---

## System Architecture & Workflow

1. **Input Management (`app.py`):** Validates uploaded retinal images (`JPG`, `JPEG`, `PNG`), resizes to `224×224`, and normalizes pixel intensities to `[0, 1]`.
2. **Feature Extraction (`fundus.py`, `oct.py`):**
   - *Fundus:* Extracts deep hierarchical spatial features via a pretrained **EfficientNet-B0** backbone.
   - *OCT:* Extracts statistical, histogram, gradient, and texture descriptors representing retinal layer continuity, followed by scaling (`scaler_params.npy`) and dimensionality reduction.
3. **Quantum Encoding & Hybrid Classification:** Maps reduced features into a 4-qubit PennyLane quantum node (`default.qubit`), fuses quantum expectation outputs with classical embeddings, and classifies via dense layers with ReLU and Softmax activations.
4. **Visualization & Localization (`seg.py`):** Overlays **Grad-CAM** attention maps and executes macular segmentation (`best_macular_model.h5`) on abnormal fundus cases.

---

## Repository Structure

```text
qnn-retinal-cscr-detection/
├── app.py                                    # Web application & inference pipeline
├── fundus.py                                 # Fundus preprocessing, EfficientNet-QNN & Grad-CAM
├── oct.py                                    # OCT feature extraction & quantum-classical classifier
├── seg.py                                    # Macular segmentation & connected-component post-processing
├── csr_retina.ipynb                          # Fundus QNN model training & evaluation notebook
├── oct_classification.ipynb                  # OCT hybrid QNN training & experiments
├── oct_eda.ipynb                             # Exploratory Data Analysis on OCT retinal scans
├── segmentation_csr.ipynb                    # Macular segmentation network training notebook
├── quantum_efficientnet_cscr_healthy.pth     # Trained PyTorch + PennyLane Fundus QNN weights
├── best_quantum_oct_model.h5                 # Trained OCT hybrid quantum model
├── quantum_neural_network_oct.h5             # Secondary OCT QNN weights
├── quantum_weights.npy                       # Learned variational quantum circuit parameters
├── scaler_params.npy                         # Feature normalization parameters for OCT inputs
├── static/                                   # CSS, JS, and sample visualization outputs
├── templates/                                # HTML web interface (index.html)
└── requirements.txt                          # Python dependencies
