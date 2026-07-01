# Music Genre Classification Pipeline

## Objective
This repository contains a rigorous, large-dataset machine learning pipeline for music genre classification. The project demonstrates an end-to-end workflow from streaming large-scale audio datasets to deploying robust classification models.

## Architectural Rigor
The engineering focus of this pipeline prioritizes reproducibility, efficiency, and data integrity:
- **Mathematical Determinism:** Cryptographic track-level duplicate auditing ensures exact reproducibility across runs.
- **Data Leakage Prevention:** Implements group-aware, stratified data splitting to guarantee watertight boundaries between training, validation, and test sets.
- **Optimized Data Loading:** Utilizes memory-optimized TensorFlow `tf.data.Dataset` materialization to stream and preprocess large-scale audio data efficiently without exceeding RAM limitations.

## Modeling Strategy
The pipeline employs a dual modeling approach to benchmark and maximize performance:
1. **Baseline Model:** A Random Forest classifier trained on robust, track-level tabular features (e.g., MFCCs, spectral centroid, zero-crossing rate) extracted via digital signal processing techniques.
2. **Deep Learning Model:** An EfficientNetV2B0 architecture incorporating a custom registered `ClipToRange` serialization layer, optimized for high-accuracy spectrogram classification.

## Evaluation
Model performance is evaluated using an atomic, clip-level soft-voting strategy. This ensures classification confidence is aggregated accurately across entire tracks, yielding highly reliable metrics for real-world inference.

## Usage and Setup
To execute this pipeline locally, ensure you have a standard Python environment with the required dependencies installed.

1. **Environment Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Execution:**
   Launch Jupyter Notebook or JupyterLab and open `Music_Genre_Classification.ipynb`. Run the cells sequentially to stream the dataset, train both models, and evaluate the final test set.
