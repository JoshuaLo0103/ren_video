import os
import math
import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def upsample_features(image_features, new_h, new_w, padded_h, padded_w, upsampling_method='bilinear'):
    if upsampling_method == 'bilinear':
        upsampled_feature = torch.nn.functional.interpolate(image_features, 
                                                            size=[padded_h, padded_w], mode='bilinear')
        upsampled_feature = T.CenterCrop((new_h, new_w))(upsampled_feature)
    else:
        raise ValueError(f'{upsampling_method} is not a valid upsampling method.')
    return upsampled_feature


def visualize_features(features, image, save_path):
    image_height, image_width = image.shape[1], image.shape[2]

    pca = PCA(n_components=3)
    reshaped_features = features.permute(1, 2, 0).reshape(image_height * image_width, -1).float().numpy()
    pca.fit(reshaped_features)
    pca_features = pca.transform(reshaped_features)
    pca_features = (pca_features - pca_features.min(axis = -1)[..., None]) / \
        (pca_features.max(axis = -1)[..., None] - pca_features.min(axis = -1)[..., None])
    vis_features = pca_features.reshape(image_height, image_width, 3)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(vis_features)
    plt.axis('off')
    plt.savefig(save_path)
    plt.clf()


def visualize_regions(regions, image, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, mask in enumerate(regions):
        plt.imshow(mask[:, :, None] * image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{idx}.jpg'))
        plt.clf()
    
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'image.jpg'))
    plt.clf()


def visualize_attn_weights(attn_weights, images, patch_size, grid_points=None, attn_aggregation='max', save_dir='attn_vis'):
    batch_size, num_heads, num_q, _ = attn_weights.shape
    h, w = images.shape[-2:]

    for batch_idx in range(images.shape[0]):
        batch_dir = f'{save_dir}/batch-{batch_idx}'
        os.makedirs(batch_dir, exist_ok=True)
        plt.imshow(images[batch_idx].permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')
        plt.savefig(f'{batch_dir}/image.jpg')
        plt.clf()

        attn_weights = attn_weights.view(batch_size, num_heads, num_q, h // patch_size, w // patch_size)

        for q_idx in range(num_q):
            attn_map = attn_weights[batch_idx, :, q_idx].detach().cpu().numpy()
            if attn_aggregation == 'max':
                combined_attn_map = np.max(attn_map, axis=0)
            elif attn_aggregation == 'mean':
                combined_attn_map = np.mean(attn_map, axis=0)
            plt.imshow(combined_attn_map)
            plt.axis('off')
            if grid_points is not None:
                plt.scatter([grid_points[batch_idx][q_idx][1] / patch_size], [grid_points[batch_idx][q_idx][0] / patch_size],
                            marker='o', s=20, c='red')
            plt.savefig(f'{batch_dir}/query-{q_idx}.jpg')
            plt.close()


def deduplicate_masks(masks, thresh=750):
    deduplicated_masks = []
    for i in range(len(masks)):
        duplicate = False
        for j in range(i + 1, len(masks)):
            diff = np.sum(masks[i] ^ masks[j])
            if (diff < thresh) and (diff < np.sum(masks[i])) and (diff < np.sum(masks[j])):
                duplicate = True
        if not duplicate:
            deduplicated_masks.append(masks[i])
    return deduplicated_masks


def mask_iou(m1, m2):
    inter = (m1 * m2).sum(dim=-1)
    union = m1.sum(dim=-1) + m2.sum(dim=-1) - inter
    return inter / (union + 1e-6)


def group_predictions(preds, similarity_threshold=0.9, min_component_size=3, merge_small_groups=False):
    batch_size = preds.shape[0]
    features = F.normalize(preds.view(batch_size, -1), p=2, dim=1)
    max_chunk = 2048
    rows, cols, values = [], [], []
    for i in range(0, batch_size, max_chunk):
        end_i = min(i + max_chunk, batch_size)
        chunk_i = features[i:end_i]
        
        for j in range(i, batch_size, max_chunk):
            end_j = min(j + max_chunk, batch_size)
            chunk_j = features[j:end_j]
            similarities = torch.mm(chunk_i, chunk_j.t())
            sim_mask = similarities >= similarity_threshold
            chunk_rows, chunk_cols = sim_mask.nonzero(as_tuple=True)
            if i == j:
                valid_edges = chunk_rows != chunk_cols
                chunk_rows_valid = chunk_rows[valid_edges] + i
                chunk_cols_valid = chunk_cols[valid_edges] + j
            else:
                chunk_rows_valid = chunk_rows + i
                chunk_cols_valid = chunk_cols + j
            
            rows.extend(chunk_rows_valid.cpu().numpy())
            cols.extend(chunk_cols_valid.cpu().numpy())
            values.extend([1] * len(chunk_rows_valid))
            if i != j:
                rows.extend(chunk_cols_valid.cpu().numpy())
                cols.extend(chunk_rows_valid.cpu().numpy())
                values.extend([1] * len(chunk_rows_valid))
    
    # Create adjacency matrix and find connected components
    adj_matrix = csr_matrix((values, (rows, cols)), shape=(batch_size, batch_size))
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    groups = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        groups[label].append(idx)
    
    # Identify the large groups and merge smaller groups into larger ones if needed
    large_groups = [group for group in groups if len(group) >= min_component_size]
    if not merge_small_groups or min_component_size <= 1:
        return large_groups
    else:
        small_groups = [group for group in groups if len(group) < min_component_size]
        if not small_groups or not large_groups:
            return large_groups + small_groups
        
        large_centroids = []
        for group in large_groups:
            group_tensor = torch.tensor(group, device=features.device)
            group_features = features[group_tensor]
            centroid = torch.mean(group_features, dim=0)
            large_centroids.append(centroid)
        
        large_centroids_tensor = torch.stack(large_centroids, dim=0)
        large_centroids_normalized = F.normalize(large_centroids_tensor, p=2, dim=1)
        all_small_centroids = []
        for small_group in small_groups:
            group_tensor = torch.tensor(small_group, device=features.device)
            group_features = features[group_tensor]
            small_centroid = torch.mean(group_features, dim=0)
            all_small_centroids.append(small_centroid)
        
        if all_small_centroids:
            all_small_centroids_tensor = torch.stack(all_small_centroids, dim=0)
            all_small_centroids_normalized = F.normalize(all_small_centroids_tensor, p=2, dim=1)
            similarities = torch.mm(all_small_centroids_normalized, large_centroids_normalized.t())
            best_matches = torch.argmax(similarities, dim=1).cpu().numpy()
            for idx, best_idx in enumerate(best_matches):
                large_groups[best_idx].extend(small_groups[idx])
        return large_groups


def pad_or_truncate_tokens(tokens, pad_length, pad_value):
    current_length, dim_size = tokens.shape
    
    if current_length > pad_length:
        return tokens[:pad_length]
    
    if current_length < pad_length:
        padding = torch.full((pad_length - current_length, dim_size), pad_value, 
                              dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, padding], dim=0)


def is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError('invalid input for is_power_of_2: {} (type: {})'.format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def print_log(log_str, save_dir=None):
    print(log_str)
    if save_dir is not None:
        log_file = os.path.join(save_dir, 'log.txt')
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')