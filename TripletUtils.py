import random
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses, miners
from tqdm import tqdm


class EmbeddingModel(nn.Module):
    def __init__(self, base_model, embedding_dimension=128, freeze_base=False, normalize=False):
        super(EmbeddingModel, self).__init__()
        self.base_model = base_model

        last_layer = list(base_model.children())[-1]
        self.embedding_layer = nn.Linear(last_layer.out_features, embedding_dimension)  # Assume layers in order

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.normalize = normalize

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x)
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x


class FusionModel(nn.Module):
    def __init__(self, base_model, input_size):
        super(FusionModel, self).__init__()
        self.base_model = base_model
        self.input_size = input_size

    def forward(self, x):
        chunks = torch.split(x, self.input_size, dim=1)
        processed_chunks = [self.base_model(chunk) for chunk in chunks]
        stacked_outputs = torch.stack(processed_chunks, dim=1)  # Shape: [batch_size, fuse_num, out_size]
        averaged_output = torch.mean(stacked_outputs, dim=1)
        return averaged_output


class MultiScaleFusionModel(nn.Module):
    def __init__(self, base_model, input_resolutions):
        super(MultiScaleFusionModel, self).__init__()
        self.base_model = base_model
        self.input_resolutions = input_resolutions

    def forward(self, x):
        processed_chunks = []
        for input_resolution in self.input_resolutions:
            if input_resolution == 1:
                chunks = x.unsqueeze(dim=1)  # Shape: [batch_size, 1, SEQ_LENGTH, 2]
            else:
                reshaped = x.view(x.size(0), x.size(1) // input_resolution, input_resolution, 2)
                chunks = reshaped.sum(dim=2).unsqueeze(dim=1)  # Shape: [batch_size, SEQ_LENGTH // input_resolution, 2]

            scale_chunks = []
            for chunk in chunks:
                scale_chunks.append(self.base_model(chunk))

            stacked_outputs = torch.stack(scale_chunks, dim=1)  # Shape: [batch_size, fuse_num, out_size]
            averaged_output = torch.mean(stacked_outputs, dim=1)
            processed_chunks.append(averaged_output)

        stacked_outputs = torch.stack(processed_chunks, dim=1)  # Shape: [batch_size, fuse_num, out_size]
        final_output = torch.mean(stacked_outputs, dim=1)

        return final_output


class CentroidModel(nn.Module):
    def __init__(self, base_model):
        super(CentroidModel, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        processed_chunks = [self.base_model(sequence) for sequence in x]
        stacked_outputs = torch.stack(processed_chunks, dim=1)  # Shape: [batch_size, fuse_num, out_size]
        averaged_output = torch.mean(stacked_outputs, dim=1)
        return averaged_output


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
    eer = 1 - fpr[eer_idx]
    return 1 - eer


def train_embedding_model(model, train_loader, validation_loader, model_output, num_epochs=10, learning_rate=0.001,
                          distance_function='euclidean', margin=1.0):
    criterion = TripletLoss(margin=margin, distance=distance_function)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Semi hard
    '''
    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard")
    loss_func = losses.TripletMarginLoss(margin=margin)
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tot_accs = []
    tot_val_accs = []
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
            # Semi hard
            '''
            embeddings = torch.cat([anchor_output, positive_output, negative_output], dim=0)
            labels = torch.cat([torch.zeros(anchor_output.size(0)), torch.ones(positive_output.size(0)),
                                torch.ones(negative_output.size(0)) + 1], dim=0)
            hard_triplets = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_triplets)  # Semi hard triplet loss
            '''
            loss = criterion(anchor_output, positive_output, negative_output)  # Standard triplet loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_accs = []
        with torch.no_grad():
            for (anchor, positive, negative), _ in tqdm(validation_loader, total=len(validation_loader)):
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
        tot_accs.append(np.mean(accs))
        tot_val_accs.append(np.mean(val_accs))
    plt.plot(tot_accs)
    plt.plot(tot_val_accs)
    plt.legend(['Training', 'Validation'])
    plt.show()


def embeddings_to_dict(embeddings, labels):
    user_embeddings = defaultdict(list)
    for emb, label in zip(embeddings, labels):
        user_embeddings[label].append(emb)

    for label in user_embeddings:
        user_embeddings[label] = np.array(user_embeddings[label])
    return user_embeddings
