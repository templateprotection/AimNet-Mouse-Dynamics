import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Preprocessing import create_user_sequences, user_based_train_test_split
from ClassificationUtils import train_classification_model, create_data_loader, LSTMClassifier
from TripletUtils import train_embedding_model, create_triplet_loader, EmbeddingModel
from EvaluationUtils import evaluate_model, get_all_embeddings

# Config
SEQ_LENGTH = 100  # Sequence Length: Number of time steps for each sequence
CLASSIFIER_EPOCHS = 100
EMBEDDING_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 128
NUM_CLASSES = 1000  # Number of users used for the classifier
VALID_SPLIT = 0.25  # Ratio of NUM_CLASSES to use for validation
FUSE_GALLERY = 30  # Number of stored embeddings to fuse (Evaluation)
FUSE_TEST = 30  # Number of live embeddings to fuse (Evaluation)
DATA_INPUT_PATH = './dataset/aim_data.csv'
CLASSIFIER_OUT_PATH = './models/aimnet_classifier.pth'
EMBEDDER_OUT_PATH = './models/aimnet_embedding_model.pth'
EMBEDDINGS_PATH = './embeddings/generated_embeddings.npy'
EMBEDDINGS_LABELS_PATH = './embeddings/generated_labels.npy'
READ_COLUMNS = ['uuid', 'delta_yaw', 'delta_pitch']
LABEL_COLUMN = 'uuid'
FEATURE_COLUMNS = ['delta_yaw', 'delta_pitch']
DISTANCE_FUNCTION = 'euclidean'  # 'euclidean', 'cosine', or 'fused'

MODE = 'train_classifier_model'  # Train the classifier to identify the first NUM_CLASSES users.
# MODE = 'train_embedding_model'  # Train the embedding model using triplet loss on the same users.
# MODE = 'evaluate_embedding_model'  # Test the embedding model on unseen data for fair evaluation.

DO_PREPROCESSING = True  # Keep this 'True' unless you want to evaluate pre-saved embeddings in EMBEDDINGS_PATH.
FREEZE_BASE = True  # Set 'True' to freeze the base classifier when training the embedding model.


def main():
    if DO_PREPROCESSING:
        print("Loading data")
        data = pd.read_csv(DATA_INPUT_PATH, usecols=READ_COLUMNS)

        sequences, labels = create_user_sequences(data, seq_length=SEQ_LENGTH)
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

        print("Splitting data into train, validation, and test sets")
        train_sequences, train_labels, test_sequences, test_labels = (
            user_based_train_test_split(sequences, labels, NUM_CLASSES))

        train_sequences, valid_sequences, train_labels, valid_labels = (
            train_test_split(train_sequences, train_labels, test_size=VALID_SPLIT, random_state=42, stratify=train_labels))

        print("Creating loaders")
        train_loader = create_data_loader(train_sequences, train_labels, batch_size=BATCH_SIZE)
        valid_loader = create_data_loader(valid_sequences, valid_labels, batch_size=BATCH_SIZE)
        triplet_train_loader = create_triplet_loader(train_sequences, train_labels, batch_size=BATCH_SIZE)
        triplet_valid_loader = create_triplet_loader(valid_sequences, valid_labels, batch_size=BATCH_SIZE)
    else:
        # TODO: Possibly load preprocessed loaders from numpy files.
        pass

    input_size = len(FEATURE_COLUMNS)
    if MODE == 'train_classifier_model':
        print("Training the classification model")
        classification_model = LSTMClassifier(input_size=input_size, output_size=NUM_CLASSES)
        train_classification_model(model=classification_model,
                                   train_loader=train_loader,
                                   validation_loader=valid_loader,
                                   model_output=CLASSIFIER_OUT_PATH,
                                   num_epochs=CLASSIFIER_EPOCHS,
                                   learning_rate=LEARNING_RATE)

    elif MODE == 'train_embedding_model':
        print("Training the embedding model")
        classification_model = LSTMClassifier(input_size, NUM_CLASSES)
        classification_model.load_state_dict(torch.load(CLASSIFIER_OUT_PATH))
        embedding_model = EmbeddingModel(classification_model, embedding_dimension=EMBEDDING_SIZE, freeze_base=FREEZE_BASE)

        train_embedding_model(model=embedding_model,
                              train_loader=triplet_train_loader,
                              validation_loader=triplet_valid_loader,
                              model_output=EMBEDDER_OUT_PATH,
                              num_epochs=EMBEDDING_EPOCHS,
                              learning_rate=LEARNING_RATE,
                              distance_function=DISTANCE_FUNCTION)

    elif MODE == 'evaluate_embedding_model':
        print("Evaluating embedding model on the hold-out test users")
        if DO_PREPROCESSING:
            classification_model = LSTMClassifier(input_size, NUM_CLASSES)
            classification_model.load_state_dict(torch.load(CLASSIFIER_OUT_PATH))
            embedding_model = EmbeddingModel(classification_model, EMBEDDING_SIZE)
            embedding_model.load_state_dict(torch.load(EMBEDDER_OUT_PATH))

            test_loader = create_data_loader(test_sequences, test_labels)
            embeddings, labels = get_all_embeddings(embedding_model, test_loader)
            np.save(EMBEDDINGS_PATH, embeddings)
            np.save(EMBEDDINGS_LABELS_PATH, labels)
        else:
            embeddings = np.load(EMBEDDINGS_PATH)
            labels = np.load(EMBEDDINGS_LABELS_PATH)

        evaluate_model(embeddings, labels, FUSE_GALLERY, FUSE_TEST, DISTANCE_FUNCTION)


if __name__ == '__main__':
    main()
