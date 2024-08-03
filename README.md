# AimNet Mouse Dynamics

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
    git clone https://github.com/templateprotection/AimNet-Mouse-Dynamics.git
    cd AimNet-Mouse-Dynamics
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
- `SEQ_LENGTH = 100`: Sequence length for each input sequence.
- `CLASSIFIER_EPOCHS = 100`: Number of epochs for training the classification model.
- `EMBEDDING_EPOCHS = 100`: Number of epochs for training the embedding model.
- `BATCH_SIZE = 256`: Batch size for training.
- `LEARNING_RATE = 0.001`: Learning rate for optimization.
- `EMBEDDING_SIZE = 128`: Size of the embedding vector.
- `NUM_CLASSES = 1000`: Number of users for classification.
- `VALID_SPLIT = 0.2`: Ratio of training data to use for validation.
- `FUSE_GALLERY = 30` and `FUSE_TEST = 30`: Number of embeddings to fuse for evaluation.
- `DATA_INPUT_PATH = 'aim_data.csv'`: Path to the input data file.
- `CLASSIFIER_OUT_PATH = './models/aimnet_classifier.pth'`: Path to save the trained classifier model.
- `EMBEDDER_OUT_PATH = './models/aimnet_embedding_model.pth'`: Path to save the trained embedding model.
- `EMBEDDINGS_PATH = './embeddings/generated_embeddings.npy'`: Path to save the generated embeddings.
- `EMBEDDINGS_LABELS_PATH = './embeddings/generates_labels.npy'`: Path to save the labels for the generated embeddings.
- `READ_COLUMNS = ['uuid', 'delta_yaw', 'delta_pitch']`: Columns to read from the input CSV file.
- `LABEL_COLUMN = 'uuid'`: Column to use as labels.
- `FEATURE_COLUMNS = ['delta_yaw', 'delta_pitch]`: Columns to use as features.
- `DISTANCE_FUNCTION = 'euclidean'`: Distance function for evaluation (`'euclidean'`, `'cosine'`, or `'fused'`).

## Usage

1. **Train the classification model**:

    In AimNet.py, adjust the configuration to `MODE: 'train_classifier_model'` and `DO_PREPROCESSING: True`, then run the script.
    ```bash
    python AimNet.py
    ```
    The model will automatically train with the specified parameters, and (when done) produce a plot of accuracy over time. 

2. **Train the embedding model**:
   
    In AimNet.py, adjust the configuration to `MODE: 'train_embedding_model'` and `DO_PREPROCESSING: True`, then run the script.
    ```bash
    python AimNet.py
    ```
    If desired, you may adjust `FREEZE_BASE` as `True` or `False`, noting that disabling this option will take longer to train, but may produce better results.

4. **Evaluate Embedding Model**:
   
    In AimNet.py, adjust the configuration to `MODE: 'evaluate_embedding_model'` and `DO_PREPROCESSING: True`, then run the script.
    ```bash
    python AimNet.py
    ```
    After running this the first time, it will generate the embeddings and write them to `EMBEDDINGS_PATH` and `EMBEDDINGS_LABEL_PATH`. Once these exist for a given model, you should set `DO_PREPROCESSING = False` to load embeddings directly from those files without processing them again.
    The result will produce a histogram of distances between positive samples (embeddings from the same user) and negative samples (embeddings from different users). The desired result is that these two samples form separable peaks with little to no overlap between them, measured by Equal Error Rate (EER). Below are examples of a single embedding (Figure 1) and 30 embeddings (Figure 2).
    ### LSTM Embedding Performance
    
    
    <table style="border-collapse: collapse; width: 100%;">
      <tr>
        <td style="text-align: center; border: none;">
          <img src="results/Histogram_LSTM_1_1.png" width="100%" style="max-width: 300px;" />
        </td>
        <td style="text-align: center; border: none;">
          <img src="results/Histogram_LSTM_30_30.png" width="100%" style="max-width: 300px;" />
        </td>
      </tr>
      <tr>
        <td style="text-align: center; border: none;">
          <i>Figure 1: Distances between individuals using a single embedding produces an EER of 23.54%.</i>
        </td>
        <td style="text-align: center; border: none;">
          <i>Figure 2: Distances between individuals using 30 combined embeddings produces an EER of 0.00%.</i>
        </td>
      </tr>
    </table>


   These results were achieved on a held-old test set of users that were not part of training.

## Contributing

Contributions are welcome - Please submit a pull request or open an issue if you have suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [templateprotection] at ~~[temporarily_private@example.com]~~, or join the Discord at ~~[temporarily_private.com]~~

