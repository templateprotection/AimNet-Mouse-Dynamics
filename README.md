# Minecraft Mouse Dynamics

This repository contains a machine learning pipeline for analyzing mouse dynamics in Minecraft. The pipeline includes data preprocessing, model training, and evaluation for identifying distinct individuals via their mouse movement patterns.

## Project Overview

The project includes:
- **Preprocessing**: Functions to prepare mouse movement data for model training.
- **Classification**: An LSTM classifier to identify users based on their mouse dynamics.
- **Embedding**: An embedding model trained with triplet loss to generate user embeddings.
- **Evaluation**: Tools to evaluate the performance of the embedding model.

## File Structure

- `AimNet.py`: The main script to execute the pipeline, including preprocessing, training, and evaluation.
- `Preprocessing.py`: Contains functions for creating user sequences and splitting data.
- `ClassificationUtils.py`: Contains utilities for training the classification model and creating data loaders.
- `TripletUtils.py`: Contains utilities for training the embedding model and creating triplet loaders.
- `EvaluationUtils.py`: Contains functions for evaluating the embedding model and generating embeddings.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/templateprotection/Minecraft-Mouse-Dynamics.git
    cd Minecraft-Mouse-Dynamics
    ```

2. **Install Dependencies**:
    Ensure you have Python 3.7 or later. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` should include:
    ```
    numpy
    pandas
    torch
    scikit-learn
    ```

## Configuration

Edit the `AimNet.py` file to configure the following parameters:
- `SEQ_LENGTH`: Sequence length for each input sequence.
- `CLASSIFIER_EPOCHS`: Number of epochs for training the classification model.
- `EMBEDDING_EPOCHS`: Number of epochs for training the embedding model.
- `BATCH_SIZE`: Batch size for training.
- `LEARNING_RATE`: Learning rate for optimization.
- `EMBEDDING_SIZE`: Size of the embedding vector.
- `NUM_CLASSES`: Number of users for classification.
- `VALID_SPLIT`: Ratio of training data to use for validation.
- `FUSE_GALLERY` and `FUSE_TEST`: Number of embeddings to fuse for evaluation.
- `DATA_INPUT_PATH`: Path to the input data file.
- `CLASSIFIER_OUT_PATH`: Path to save the trained classifier model.
- `EMBEDDER_OUT_PATH`: Path to save the trained embedding model.
- `EMBEDDINGS_PATH`: Path to save the generated embeddings.
- `EMBEDDINGS_LABELS_PATH`: Path to save the labels for the generated embeddings.
- `READ_COLUMNS`: Columns to read from the input CSV file.
- `LABEL_COLUMN`: Column to use as labels.
- `FEATURE_COLUMNS`: Columns to use as features.
- `DISTANCE_FUNCTION`: Distance function for evaluation (`'euclidean'`, `'cosine'`, or `'fused'`).

## Usage

1. **Preprocess Data and Train Models**:
    ```bash
    python AimNet.py
    ```

    Adjust the `MODE` variable in `AimNet.py` to:
    - `'train_classifier_model'`: Train the classification model.
    - `'train_embedding_model'`: Train the embedding model.
    - `'evaluate_embedding_model'`: Evaluate the embedding model.

2. **Preprocessing**:
    Set `DO_PREPROCESSING` to `True` to preprocess the data, or false if embedding files are already generated.

3. **Evaluate Model**:
    If `MODE` is set to `'evaluate_embedding_model'`, the script will evaluate the embedding model and save the embeddings and labels.

## Contributing

Contributions are welcome - Please submit a pull request or open an issue if you have suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [templateprotection] at ~~[temporarily_private@example.com]~~, or join the Discord at ~~[temporarily_private.com]~~

