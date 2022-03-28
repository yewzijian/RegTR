import torch


def collate_pair(list_data):
    """Collates data using a list, for tensors which are of different sizes
    (e.g. different number of points). Otherwise, stacks them as per normal.
    """

    batch_sz = len(list_data)

    # Collate as normal, other than fields that cannot be collated due to differing sizes,
    # we retain it as a python list
    to_retain_as_list = ['src_xyz', 'tgt_xyz', 'tgt_raw',
                         'src_overlap', 'tgt_overlap',
                         'correspondences',
                         'src_path', 'tgt_path',
                         'idx']
    data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
    if 'overlap_p' in list_data[0]:
        data['overlap_p'] = torch.tensor([list_data[b]['overlap_p'] for b in range(batch_sz)])
    return data
