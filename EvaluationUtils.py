import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import roc_curve
from tqdm import tqdm

from TripletUtils import embeddings_to_dict


def evaluate_model(embeddings, labels, fuse_gallery, fuse_test, distance_function):
    user_embeddings = embeddings_to_dict(embeddings, labels)
    p_dists, n_dists, p_labels, n_labels = compute_fused_distances(user_embeddings, fuse_gallery, fuse_test, distance_function)
    fused_embeddings, fused_labels = fuse_embeddings(user_embeddings, fuse_gallery)
    my_dict = embeddings_to_dict(fused_embeddings, fused_labels)
    eer, eer_threshold, _, _ = compute_pairwise_eer(my_dict, num_imposter_users=20)
    p_dists = np.array(p_dists)
    n_dists = np.array(n_dists)

    id_acc, correct_users, total_users = compute_identification_acc(user_embeddings, fuse_gallery, 1000)
    print(f"ID ACCURACY [Fuse={fuse_gallery}]: {id_acc}  ({correct_users}/{total_users})")

    # Distance Histogram
    plot_histogram(p_dists, n_dists, eer, eer_threshold, fuse_gallery, fuse_test)


def get_all_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for sequence, label in tqdm(loader, desc="Processing test data"):
            embedding = model(sequence)
            embeddings.extend(embedding.numpy())
            labels.extend(label.numpy())

    return np.array(embeddings), np.array(labels)


def plot_histogram(p_dists, n_dists, eer, eer_threshold, gal_fusion=1, test_fusion=1):
    plt.figure(figsize=(10, 6))
    plt.hist(p_dists, bins=50, color='blue', alpha=0.7, label='Positive Distances')
    plt.hist(n_dists, bins=50, color='red', alpha=0.7, label='Negative Distances')
    plt.axvline(eer_threshold, color='green', linestyle='--', label=f'EER = {eer:.4f}, Threshold: {eer_threshold:.4f}')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title(f'Histogram of Embedding Distances ({gal_fusion}:{test_fusion}-Fusion)')
    plt.legend()
    plt.grid(True)
    plt.show()


def calc_dist(gallery_embeddings, test_embeddings, distance_function):
    gallery_embeddings = np.mean(gallery_embeddings, axis=0).reshape((1, gallery_embeddings.shape[1]))
    test_embeddings = np.mean(test_embeddings, axis=0).reshape((1, test_embeddings.shape[1]))
    if distance_function == 'euclidean':
        euclidean_dists = np.sqrt(np.sum((gallery_embeddings - test_embeddings) ** 2, axis=1))
        return np.mean(euclidean_dists)

    elif distance_function == 'cosine':
        gallery_norm = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        test_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        cosine_similarity = np.sum(gallery_norm * test_norm, axis=1)
        cosine_distances = 1 - cosine_similarity
        return np.mean(cosine_distances)

    elif distance_function == 'fused':
        euclidean_dists = np.sqrt(np.sum((gallery_embeddings - test_embeddings) ** 2, axis=1))
        gallery_norm = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        test_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        cosine_similarity = np.sum(gallery_norm * test_norm, axis=1)
        cosine_dists = 1 - cosine_similarity
        fused_dists = (cosine_dists + euclidean_dists) / 2
        return np.mean(fused_dists)


def compute_pairwise_distances(embeddings):
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix


def compute_pairwise_eer(user_embeddings, num_imposter_users=None):
    num_imposter_users = num_imposter_users or len(user_embeddings)-1
    genuine_dists = []
    imposter_dists = []

    users = list(user_embeddings.keys())

    for user in tqdm(users):
        current_user_embeddings = user_embeddings[user]
        if len(current_user_embeddings) > 1:
            pairwise_dists = pdist(current_user_embeddings, metric='euclidean')
            genuine_dists.extend(pairwise_dists)

        impostor_users = np.random.choice([u for u in users if u != user], size=num_imposter_users, replace=False)

        for imp_user in impostor_users:
            imposter_embeddings = user_embeddings[imp_user]
            cross_dists = cdist(current_user_embeddings, imposter_embeddings, metric='euclidean')
            imposter_dists.extend(cross_dists.flatten())

    genuine_labels = np.ones(len(genuine_dists))
    imposter_labels = np.zeros(len(imposter_dists))

    dists_flat = np.concatenate([genuine_dists, imposter_dists])
    labels_flat = np.concatenate([genuine_labels, imposter_labels])

    fpr, tpr, thresholds = roc_curve(labels_flat, dists_flat)
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_idx]
    eer = 1 - fpr[eer_idx]

    return eer, eer_threshold, genuine_dists, imposter_dists


def compute_identification_acc(user_embeddings, fuse_num, max_users=1000, samples_per_user=2):
    total_fused_labels = []
    total_fused_embs = []
    unique_users = np.array(list(user_embeddings.keys()))
    np.random.shuffle(unique_users)

    for user in unique_users[:max_users]:
        embs = user_embeddings[user]
        num_groups = len(embs) // fuse_num

        truncate_point = num_groups * fuse_num
        if num_groups < samples_per_user:
            continue

        embs_grouped = np.reshape(embs[:truncate_point], (num_groups, fuse_num, 128))
        fused_embs = np.mean(embs_grouped, axis=1)
        np.random.shuffle(fused_embs)

        total_fused_labels.extend([user, user])
        total_fused_embs.extend(fused_embs[:samples_per_user])

    total_fused_embs = np.array(total_fused_embs)
    total_fused_labels = np.array(total_fused_labels)

    pw_dists = compute_pairwise_distances(total_fused_embs)
    closest_inds = np.argmin(pw_dists, axis=0)
    closest_labels = total_fused_labels[closest_inds]
    total_identified = np.count_nonzero(closest_labels == total_fused_labels)
    id_rate = total_identified / len(total_fused_labels)
    return id_rate, total_identified, len(total_fused_labels)


def fuse_embeddings(user_embeddings, fuse_num):
    fused_labels = []
    fused_embeddings = []
    for user in user_embeddings:
        embeddings = user_embeddings[user]
        num_groups = len(embeddings) // fuse_num
        embeddings = embeddings[:num_groups*fuse_num]  # Truncate excess
        grouped_embeddings = embeddings.reshape((num_groups, fuse_num, embeddings.shape[-1]))
        if num_groups >= 2:
            fused_embeddings.extend(grouped_embeddings.mean(axis=1))
            fused_labels.extend([user] * num_groups)

    return np.array(fused_embeddings), np.array(fused_labels)


def compute_fused_distances(user_embeddings, fuse_gallery, fuse_test, distance_function):
    p_labels = []
    n_labels = []
    p_dists = []
    n_dists = []

    users = list(user_embeddings.keys())

    for user in users:
        embs = user_embeddings[user]
        num_embs = embs.shape[0]

        if num_embs < fuse_gallery + fuse_test:
            continue

        for _ in range(num_embs // (fuse_gallery + fuse_test)):
            indices = random.sample(range(num_embs), fuse_gallery + fuse_test)
            gallery_indices = indices[:fuse_gallery]
            genuine_test_indices = indices[fuse_gallery:]

            gallery_embs = embs[gallery_indices]
            genuine_test_embs = embs[genuine_test_indices]

            genuine_distance = calc_dist(gallery_embs, genuine_test_embs, distance_function)
            p_dists.append(genuine_distance)
            p_labels.append(user)

            imposter_user = random.choice([u for u in users if u != user])
            imposter_embs = user_embeddings[imposter_user]

            if imposter_embs.shape[0] < fuse_test:
                continue

            imposter_test_indices = random.sample(range(imposter_embs.shape[0]), fuse_test)
            imposter_test_embs = imposter_embs[imposter_test_indices]

            imposter_distance = calc_dist(gallery_embs, imposter_test_embs, distance_function)
            n_dists.append(imposter_distance)
            n_labels.append(imposter_user)

    return p_dists, n_dists, p_labels, n_labels
