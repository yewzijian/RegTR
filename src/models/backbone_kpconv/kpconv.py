"""Predator model and KPConv processing code
"""

from typing import List

import MinkowskiEngine as ME
import numpy as np
import torch.nn
import torch.nn.functional as F
from pytorch3d.ops import packed_to_padded, ball_query

# # Uncomment the following two lines if you want to use the CPU operations for KPConv
# # preprocessing (you'll need to compile the code using the included bash scripts)
# from .cpp_wrappers.cpp_subsampling import grid_subsampling as cpp_subsampling
# from .cpp_wrappers.cpp_neighbors import radius_neighbors as cpp_neighbors
from .kpconv_blocks import *


_logger = logging.getLogger(__name__)


class KPFEncoder(torch.nn.Module):
    def __init__(self, config, d_bottle, increase_channel_when_downsample=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        octave = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim

        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next octave for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     octave,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                octave += 1
                r *= 2
                if increase_channel_when_downsample:
                    out_dim *= 2

        if 'upsample' not in block:
            """If we don't have a decoder, last block will not be an upsampling block,
            and the last block is not appended. This fixes that.
            """
            self.encoder_skips.append(block_i)
            self.encoder_skip_dims.append(in_dim)

    def forward(self, x, batch):
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)  # (N, C)

        return x, skip_x


class KPFDecoder(torch.nn.Module):
    def __init__(self, config, in_dim, encoder_skip_dims, reduce_channel_when_upsample=True):
        """Decoder (upsampling) part of KPConv backbone in Predator. Unused in
        REGTR since we do not perform upsampling.
        """
        super().__init__()

        out_dim = in_dim

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        octave = 0
        start_i = 0
        r = config.first_subsampling_dl * config.conv_radius
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break
            elif 'pool' in block or 'strided' in block:
                octave += 1
                r *= 2

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += encoder_skip_dims[octave]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     octave,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                octave -= 1
                r *= 0.5
                if reduce_channel_when_upsample:
                    out_dim = out_dim // 2

    def forward(self, x, skip_x, batch):

        x_all = []
        pyr = len(batch['stack_lengths']) - 1

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                pyr -= 1

            if isinstance(block_op, UnaryBlock):
                x2 = torch.cat([x, skip_x.pop()], dim=1)
                x = block_op(x2, batch['stack_lengths'][pyr])
            elif isinstance(block_op, UnaryBlock2):
                x2 = torch.cat([x, skip_x.pop()], dim=1)
                x = x + block_op(x2)
            else:
                x = block_op(x, batch)

            if block_i in self.decoder_concats:
                x_all.append(x)

        return x, x_all



######## Functions to compute the KPConv required metadata, i.e. neighbor/pooling indices ######

def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0,
                                  random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(
            s_labels)


def batch_grid_subsampling_kpconv_gpu(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0):
    """
    Same as batch_grid_subsampling, but implemented in GPU. This is a hack by using Minkowski
    engine's sparse quantization functions
    Note: This function is not deterministic and may return subsampled points
      in a different ordering, which will cause the subsequent steps to differ slightly.
    """

    if labels is not None or features is not None:
        raise NotImplementedError('subsampling not implemented for features and labels')
    if max_p != 0:
        raise NotImplementedError('subsampling only implemented by considering all points')

    B = len(batches_len)
    batch_start_end = torch.nn.functional.pad(torch.cumsum(batches_len, 0), (1, 0))
    device = points[0].device

    coord_batched = ME.utils.batched_coordinates(
        [points[batch_start_end[b]:batch_start_end[b + 1]] / sampleDl for b in range(B)], device=device)
    sparse_tensor = ME.SparseTensor(
        features=points,
        coordinates=coord_batched,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    )

    s_points = sparse_tensor.features
    s_len = torch.tensor([f.shape[0] for f in sparse_tensor.decomposed_features], device=device)
    return s_points, s_len


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def batch_neighbors_kpconv_gpu(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    This makes use of the GPU operations provided by PyTorch3D
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    B = len(q_batches)
    N_spts_total = supports.shape[0]
    q_first_idx = F.pad(torch.cumsum(q_batches, dim=0)[:-1], (1, 0))
    queries_padded = packed_to_padded(queries, q_first_idx, q_batches.max().item())  # (B, N_max, 3)
    s_first_idx = F.pad(torch.cumsum(s_batches, dim=0)[:-1], (1, 0))
    supports_padded = packed_to_padded(supports, s_first_idx, s_batches.max().item())  # (B, N_max, 3)

    idx = ball_query(queries_padded, supports_padded,
                     q_batches, s_batches,
                     K=max_neighbors, radius=radius).idx  # (N_clouds, N_pts, K)
    idx[idx < 0] = torch.iinfo(idx.dtype).min

    idx_packed = torch.cat([idx[b][:q_batches[b]] + s_first_idx[b] for b in range(B)], dim=0)
    idx_packed[idx_packed < 0] = N_spts_total

    return idx_packed


class Preprocessor(torch.nn.Module):
    """Computes the metadata used for KPConv"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, pts: List[torch.Tensor]):
        """Compute the neighbor and pooling indices required for KPConv operations.
        Only supports CPU tensors, so we first move all the tensors to CPU before
        moving them back.

        Args:
            pts: List of point clouds XYZ, each of size (Ni, 3), where each Ni can be different

        Returns:

        """
        device = pts[0].device
        pts = [p.cpu() for p in pts]

        config = self.cfg
        neighborhood_limits = self.cfg.neighborhood_limits

        r_normal = config.first_subsampling_dl * config.conv_radius
        layer_blocks = []
        layer = 0

        batched_lengths = torch.tensor([p.shape[0] for p in pts], dtype=torch.int32)
        batched_points = torch.cat(pts, dim=0)

        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_batch_lens = []

        for block_i, block in enumerate(config.architecture):
            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(config.architecture) - 1 and not (
                        'upsample' in config.architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal
                conv_i = batch_neighbors_kpconv(batched_points, batched_points,
                                                batched_lengths, batched_lengths,
                                                r, neighborhood_limits[layer])

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = torch.zeros((0, 1), dtype=torch.int64)

            # Pooling neighbors indices
            # *************************
            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths,
                                                               sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                                neighborhood_limits[layer])

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b,
                                              2 * r, neighborhood_limits[layer])

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = torch.zeros((0, 1), dtype=torch.int64)
                pool_p = torch.zeros((0, 3), dtype=torch.float32)
                pool_b = torch.zeros((0,), dtype=torch.int64)
                up_i = torch.zeros((0, 1), dtype=torch.int64)

            # Updating input lists
            input_points.append(batched_points)
            input_neighbors.append(conv_i.long())
            input_pools.append(pool_i.long())
            input_upsamples.append(up_i.long())
            input_batch_lens.append(batched_lengths)

            # New points for next layer
            batched_points = pool_p
            batched_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer += 1
            layer_blocks = []

        data = {
            'points': [x.to(device) for x in input_points],
            'neighbors': [x.to(device) for x in input_neighbors],
            'pools': [x.to(device) for x in input_pools],
            'upsamples': [x.to(device) for x in input_upsamples],
            'stack_lengths': [x.to(device) for x in input_batch_lens],
        }

        return data


class PreprocessorGPU(torch.nn.Module):
    """Computes the metadata used for KPConv (GPU version, which is much faster)
    However, note that this is not deterministic, even with seeding.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, pts: List[torch.Tensor]):
        """Compute the neighbor and pooling indices required for KPConv operations.

        Args:
            pts: List of point clouds XYZ, each of size (Ni, 3), where each Ni can be different
        """

        config = self.cfg
        neighborhood_limits = self.cfg.neighborhood_limits
        device = pts[0].device

        r_normal = config.first_subsampling_dl * config.conv_radius
        layer_blocks = []
        layer = 0

        batched_lengths = torch.tensor([p.shape[0] for p in pts], dtype=torch.int64, device=device)
        batched_points = torch.cat(pts, dim=0)

        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_batch_lens = []

        for block_i, block in enumerate(config.architecture):
            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(config.architecture) - 1 and not (
                        'upsample' in config.architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal

                conv_i = batch_neighbors_kpconv_gpu(batched_points, batched_points,
                                                    batched_lengths, batched_lengths,
                                                    r, neighborhood_limits[layer])

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = torch.zeros((0, 1), dtype=torch.int64)

            # Pooling neighbors indices
            # *************************
            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling_kpconv_gpu(
                    batched_points, batched_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors_kpconv_gpu(pool_p, batched_points, pool_b, batched_lengths, r,
                                                    neighborhood_limits[layer])

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors_kpconv_gpu(batched_points, pool_p, batched_lengths, pool_b,
                                                  2 * r, neighborhood_limits[layer])

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = torch.zeros((0, 1), dtype=torch.int64)
                pool_p = torch.zeros((0, 3), dtype=torch.float32)
                pool_b = torch.zeros((0,), dtype=torch.int64)
                up_i = torch.zeros((0, 1), dtype=torch.int64)

            # Updating input lists
            input_points.append(batched_points)
            input_neighbors.append(conv_i.long())
            input_pools.append(pool_i.long())
            input_upsamples.append(up_i.long())
            input_batch_lens.append(batched_lengths)

            # New points for next layer
            batched_points = pool_p
            batched_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer += 1
            layer_blocks = []

        data = {
            'points': input_points,
            'neighbors': input_neighbors,
            'pools': input_pools,
            'upsamples': input_upsamples,
            'stack_lengths': input_batch_lens,
        }

        return data


def compute_overlaps(batch):
    """Compute groundtruth overlap for each point+level. Note that this is a
    approximation since
    1) it relies on the pooling indices from the preprocessing which caps the number of
       points considered
    2) we do a unweighted average at each level, without considering the
       number of points used to generate the estimate at the previous level
    """

    overlaps = batch['src_overlap'] + batch['tgt_overlap']
    kpconv_meta = batch['kpconv_meta']
    n_pyr = len(kpconv_meta['points'])

    overlap_pyr = {'pyr_0': torch.cat(overlaps, dim=0).type(torch.float)}
    invalid_indices = [s.sum() for s in kpconv_meta['stack_lengths']]
    for p in range(1, n_pyr):
        pooling_indices = kpconv_meta['pools'][p - 1].clone()
        valid_mask = pooling_indices < invalid_indices[p - 1]
        pooling_indices[~valid_mask] = 0

        # Average pool over indices
        overlap_gathered = overlap_pyr[f'pyr_{p-1}'][pooling_indices] * valid_mask
        overlap_gathered = torch.sum(overlap_gathered, dim=1) / torch.sum(valid_mask, dim=1)
        overlap_gathered = torch.clamp(overlap_gathered, min=0, max=1)
        overlap_pyr[f'pyr_{p}'] = overlap_gathered

    return overlap_pyr



################
# From Predator. We do not use this other than to calibrate neighbors for each dataset
################
def collate_fn_descriptor(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_lengths_list = []
    assert len(list_data) == 1, 'Data loader and model assumes batch size = 1'

    for ind, data in enumerate(list_data):
        batched_points_list.append(data['src_xyz'])
        batched_points_list.append(data['tgt_xyz'])
        batched_lengths_list.append(len(data['src_xyz']))
        batched_lengths_list.append(len(data['tgt_xyz']))

    batched_points = torch.cat(batched_points_list, dim=0)
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not (
                    'upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths,
                                            batched_lengths, r, neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths,
                                                           sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        # Points, neighbor/pooling information required for KPConv
        'points': input_points,  # List (at different octaves) of point clouds (N_src+N_tgt, 3)
        # List (at different octaves) of point cloud (N_src+N_tgt, 3)
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'stack_lengths': input_batches_len,
        'pose': data['pose'],
        # 'correspondences': data['correspondences'],
        'src_xyz_raw': data['src_xyz'],  # (N_src, 3)
        'tgt_xyz_raw': data['tgt_xyz'],  # (N_tgt, 3)
        # 'src_path': data['src_path'],
        # 'tgt_path': data['tgt_path'],
    }

    return dict_inputs


def calibrate_neighbors(dataset, config, collate_fn=collate_fn_descriptor, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in
                  batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            _logger.info(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]),
                         axis=0)  # Just aim to keep keep_ratio(0.8) of neighbors

    neighborhood_limits = percentiles

    return neighborhood_limits