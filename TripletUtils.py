import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EmbeddingModel(nn.Module):
    def __init__(self, base_model, embedding_dimension=128, freeze_base=False):
        super(EmbeddingModel, self).__init__()
        self.base_model = base_model

        last_layer = list(base_model.children())[-1]
        self.embedding_layer = nn.Linear(last_layer.out_features, embedding_dimension)  # Assume layers in order

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x)
        return x


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative):
        if self.distance == 'euclidean':
            positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
            negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)
        elif self.distance == 'cosine':
            positive_distance = 1 - torch.nn.functional.cosine_similarity(anchor, positive)
            negative_distance = 1 - torch.nn.functional.cosine_similarity(anchor, negative)
        losses = torch.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()


class TripletDataset(Dataset):
    def __init__(self, user_sequences):
        self.user_sequences = user_sequences
        self.user_keys = list(user_sequences.keys())
        self.all_sequences = [(user, seq_idx) for user in self.user_keys for seq_idx in
                              range(len(user_sequences[user]))]

        for user in user_sequences:
            if len(user_sequences[user]) < 2:
                raise RuntimeError(f"Invalid dataset: User '{user}' has less than 2 samples.")

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        anchor_user, anchor_idx = self.all_sequences[idx]
        anchor = self.user_sequences[anchor_user][anchor_idx]

        positive_idx = np.random.randint(len(self.user_sequences[anchor_user]))
        while positive_idx == anchor_idx:
            positive_idx = np.random.randint(len(self.user_sequences[anchor_user]))
        positive = self.user_sequences[anchor_user][positive_idx]

        negative_user = anchor_user
        while negative_user == anchor_user:
            negative_user = np.random.choice(self.user_keys)

        negative_idx = np.random.randint(len(self.user_sequences[negative_user]))
        negative = self.user_sequences[negative_user][negative_idx]

        return (torch.tensor(anchor, dtype=torch.float32), \
                torch.tensor(positive, dtype=torch.float32), \
                torch.tensor(negative, dtype=torch.float32)), (torch.tensor(anchor_user), torch.tensor(negative_user))


def create_triplet_loader(sequences, labels, batch_size=32, shuffle=True):
    user_sequences = embeddings_to_dict(sequences, labels)
    dataset = TripletDataset(user_sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def accuracy(anchor, positive, negative):
    positive_dist = torch.nn.functional.pairwise_distance(anchor, positive)
    negative_dist = torch.nn.functional.pairwise_distance(anchor, negative)
    distances = torch.cat([positive_dist, negative_dist], dim=0)
    labels = torch.cat([torch.zeros_like(positive_dist), torch.ones_like(negative_dist)], dim=0)
    sorted_indices = torch.argsort(distances)
    sorted_labels = labels[sorted_indices]

    n_positives = torch.sum(sorted_labels)
    n_negatives = torch.sum(1 - sorted_labels)

    tpr = torch.cumsum(sorted_labels, dim=0) / n_positives
    fpr = torch.cumsum(1 - sorted_labels, dim=0) / n_negatives

    eer_diff = torch.abs(fpr - (1 - tpr))
    eer_idx = torch.argmin(eer_diff)
    eer = 1 - fpr[eer_idx]  # TODO: This shouldn't need to be inverted. Fix issue above
    return 1 - eer


def train_embedding_model(model, train_loader, validation_loader, model_output, num_epochs=10, learning_rate=0.001,
                          distance_function='euclidean'):
    criterion = TripletLoss(margin=1.0, distance=distance_function)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        accs = []
        for (anchor, positive, negative), _ in tqdm(train_loader, total=len(train_loader)):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            accs.append(accuracy(anchor_output, positive_output, negative_output))

            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_accs = []
        with torch.no_grad():
            for (anchor, positive, negative), _ in validation_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)
                val_accs.append(accuracy(anchor_output, positive_output, negative_output))
                val_loss += criterion(anchor_output, positive_output, negative_output).item()

        torch.save(model.state_dict(), model_output)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(validation_loader):.4f}')
        print(
            f'\t\t\t\tTraining Acc: {np.mean(accs):.4f}, Validation Acc: {np.mean(val_accs):.4f}')


def embeddings_to_dict(embeddings, labels):
    user_embeddings = defaultdict(list)
    for emb, label in zip(embeddings, labels):
        user_embeddings[label].append(emb)

    for label in user_embeddings:
        user_embeddings[label] = np.array(user_embeddings[label])
    return user_embeddings
