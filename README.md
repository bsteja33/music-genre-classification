# Music Genre Classification Pipeline

## Objective
This repository contains a rigorous, production-grade machine learning pipeline for music genre classification. The project demonstrates an end-to-end workflow utilizing the benchmark [GTZAN dataset streamed directly from Hugging Face](https://huggingface.co/datasets/marsyas/gtzan). It showcases engineering mastery over large-scale audio data ingestion, watertight leakage prevention and advanced deep learning architecture deployment.

## Architectural Rigor
The engineering focus of this pipeline prioritizes reproducibility, efficiency, and strict data integrity:
- **Mathematical Determinism:** System-level seeding and strict environment variable configurations ensure exact reproducibility across compute runs before any framework initialization.
- **Data Leakage Prevention:** Implements cryptographic track-level audio hashing (SHA-256) to detect underlying dataset duplicates, enforcing group-aware, stratified splitting to guarantee watertight boundaries between training, validation, and test sets.
- **Optimized Data Pipeline:** Utilizes safe Log-Mel Spectrogram generation with strict amplitude bounds and finite checks, flowing into memory-optimized TensorFlow `tf.data.Dataset` materialization to stream and preprocess audio efficiently without memory bottlenecks.

## Modeling Strategy
The pipeline employs a dual modeling approach to benchmark and maximize performance:
1. **Baseline Model:** A Random Forest classifier (optimized via cross-validated Grid Search) trained on robust, track-level tabular features (e.g., MFCCs, spectral centroid, zero-crossing rate, rhythm features) extracted via digital signal processing techniques.
2. **Deep Learning Model:** A fine-tuned EfficientNetV2B0 architecture incorporating a custom, registered `ClipToRange` serialization layer, optimized for high-accuracy spectrogram classification.

## Evaluation
Model performance is evaluated using an atomic, clip-level soft-voting strategy. This ensures segment-level prediction probabilities are mathematically aggregated across entire audio tracks, preventing state contamination and yielding highly reliable, production-ready accuracy metrics and ROC curves.

## Usage and Setup
To execute this pipeline locally, ensure you have a standard Python environment with the required dependencies installed.

1. **Environment Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. **Execution:**
   Launch Jupyter Notebook or JupyterLab and open `Music_Genre_Classification.ipynb`. Run the cells sequentially to stream the dataset, train both models, and evaluate the final test set.
