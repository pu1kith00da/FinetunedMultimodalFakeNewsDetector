Multimodal Text-Image Classification Project Description


This project implements a multimodal classification system that combines textual and visual features for binary classification. It leverages pre-trained BERT for text embeddings and CLIP for image embeddings, using a late fusion approach. The system is designed to classify data points based on both a clean_title (text) and an associated local_image_path.

Features
Multimodal Fusion: Integrates text features from BERT and image features from CLIP.
Custom Dataset: Uses MultimodalDataset for efficient loading and preprocessing of text and image data.
Flexible Model Architecture: MultimodalBertClipClassifier allows for loading fine-tuned BERT and CLIP models.
Class Weighting: Incorporates class weights during training to handle imbalanced datasets.
Evaluation Metrics: Reports accuracy and F1-score for model performance.
Early Stopping: Prevents overfitting during training.
Setup
1. Environment Setup
Ensure you have a Python environment with PyTorch and the transformers library installed.

2. Data Preparation
The model expects data in TSV (Tab Separated Values) format for training, validation, and testing. These files should be located in your Google Drive at: /content/drive/MyDrive/data/processed_samples/

Specifically, the following files are required:

train_filtered.tsv
val_filtered.tsv
test_filtered.tsv
The DataFrames should contain at least the following columns:

clean_title (or title): For text input.
2_way_label: The target label (0 or 1).
local_image_path: The path to the image relative to /content/drive/MyDrive/ (e.g., path/to/image.jpg).
3. Pre-trained Models
The project is configured to load pre-trained BERT and CLIP models (or their fine-tuned versions) from Google Drive:

BERT Model: /content/drive/MyDrive/data/models/fine_tuned_bert
CLIP Model: /content/drive/MyDrive/data/models/clip_lora_best.pt
Ensure these paths are correct or update them in the MultimodalBertClipClassifier initialization.

Usage
1. Mount Google Drive
Mount your Google Drive to access data and pre-trained models. This is typically done with:

from google.colab import drive
drive.mount('/content/drive')
2. Run Notebook Cells
Execute the cells in the provided Colab notebook sequentially:

Environment Setup: Imports necessary libraries, sets random seeds, and defines data/output directories.
Data Loading: Loads train_df, val_df, and test_df from the specified paths.
Data Validation & Cleaning: Ensures the DataFrames have the required columns and drops rows with missing values.
Dataset Creation: Initializes BertTokenizerFast and CLIPProcessor, then creates MultimodalDataset instances for training, validation, and testing.
Model Definition: Defines the MultimodalBertClipClassifier and loads the specified BERT and CLIP models.
Trainer Initialization: Sets up the MultimodalDataCollator and MultimodalTrainer with training arguments, class weights, and early stopping.
Training and Evaluation: The trainer.train() method starts the fine-tuning process. After training, the model is evaluated on the validation and test sets.
Code Structure
MultimodalDataset: Handles loading and tokenization of text and image data.
MultimodalBertClipClassifier: Defines the neural network architecture for multimodal fusion and classification.
MultimodalDataCollator: Batches processed data for the Trainer.
MultimodalTrainer: Custom Trainer subclass to incorporate class weights in the loss calculation.
compute_metrics: Calculates accuracy and F1-score for evaluation.
Results
After executing the training and evaluation cells, the metrics (accuracy, f1-score) on the validation and test sets will be printed.

Example Output (Validation/Test Metrics):

Validation metrics: {'eval_loss': X.XXX, 'eval_accuracy': X.XXX, 'eval_f1': X.XXX, ...}
Test metrics: {'eval_loss': X.XXX, 'eval_accuracy': X.XXX, 'eval_f1': X.XXX, ...}
