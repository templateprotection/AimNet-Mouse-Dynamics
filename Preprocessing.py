import numpy as np


def user_based_train_test_split(sequences, labels, train_size=1000):
    unique_users = np.unique(labels)
    unique_train_labels = unique_users[:train_size]
    train_mask = np.isin(labels, unique_train_labels)

    train_sequences = sequences[train_mask]
    test_sequences = sequences[~train_mask]
    train_labels = labels[train_mask]
    test_labels = labels[~train_mask]

    return train_sequences, train_labels, test_sequences, test_labels


def create_user_sequences(df, seq_length=10):
    y = df[df.columns[0]]
    X = df[df.columns[1:]]

    labels = []
    sequences = []

    grouped = X.groupby(y)

    for user, user_data in grouped:
        user_data = user_data.values
        num_sequences = len(user_data) // seq_length
        user_data = user_data[:num_sequences * seq_length]

        if num_sequences >= 10:  # TODO: Improve this hardcoded check to "hope" samples reach the validation set.
            user_sequences = user_data.reshape(num_sequences, seq_length, -1)
            sequences.extend(user_sequences)
            labels.extend([user] * num_sequences)

    return np.array(sequences), np.array(labels)
