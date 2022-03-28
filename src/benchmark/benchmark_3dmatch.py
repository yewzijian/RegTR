"""Evaluation codes for 3DMatch, using the metrics defined in Deep Global Registration
"""
import os

import numpy as np


def read_trajectory(filename, dim=4):
    """
    Function that reads a trajectory saved in the 3DMatch/Redwood format to a numpy array.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    filename (str): path to the '.txt' file containing the trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    final_keys (dict): indices of pairs with more than 30% overlap (only this ones are included in the gt file)
    traj (numpy array): gt pairwise transformation matrices for n pairs[n,dim, dim]
    """

    with open(filename) as f:
        lines = f.readlines()

        # Extract the point cloud pairs
        keys = lines[0::(dim + 1)]
        temp_keys = []
        for i in range(len(keys)):
            temp_keys.append(keys[i].split('\t')[0:3])

        final_keys = []
        for i in range(len(temp_keys)):
            final_keys.append(
                [temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])

        traj = []
        for i in range(len(lines)):
            if i % 5 != 0:
                traj.append(lines[i].split('\t')[0:dim])

        traj = np.asarray(traj, dtype=np.float).reshape(-1, dim, dim)

        final_keys = np.asarray(final_keys)

        return final_keys, traj


def read_trajectory_info(filename, dim=6):
    """
    Function that reads the trajectory information saved in the 3DMatch/Redwood format to a numpy array.
    Information file contains the variance-covariance matrix of the transformation paramaters.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    filename (str): path to the '.txt' file containing the trajectory information data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    n_frame (int): number of fragments in the scene
    cov_matrix (numpy array): covariance matrix of the transformation matrices for n pairs[n,dim, dim]
    """

    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert (len(contents) == 7 * n_pairs)
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep='\t').reshape(1, -1) for item in
             contents[i * 7 + 1:i * 7 + 7]], axis=0)
        info_list.append(info_matrix)

    cov_matrix = np.asarray(info_list, dtype=np.float).reshape(-1, dim, dim)

    return n_frame, cov_matrix


def compute_rte(t, t_est):
    """Computes the translation error.
    Modified from PCAM source code https://github.com/valeoai/PCAM
    """
    return np.linalg.norm(t - t_est)


def compute_rre(R_est, R):
    """Computes the rotation error in degrees
    Modified from PCAM source code https://github.com/valeoai/PCAM
    """

    eps=1e-16

    return np.arccos(
        np.clip(
            (np.trace(R_est.T @ R) - 1) / 2,
            -1 + eps,
            1 - eps
        )
    ) * 180. / np.pi


def benchmark_dgr(est_folder, gt_folder, require_individual_errors=False,
                  re_thres=15, te_thres=0.3):
    """Evaluate 3DMatch using the metrics in Deep Global Registration, i.e.
    success if error is below 15deg, 30cm
    """

    scenes = sorted(os.listdir(gt_folder))
    scene_names = [os.path.join(gt_folder, ele) for ele in scenes]

    n_valids = []

    short_names = ['Kitchen', 'Home 1', 'Home 2', 'Hotel 1', 'Hotel 2', 'Hotel 3', 'Study',
                   'MIT Lab']
    benchmark_str = "Scene\t¦ success.\t¦ rre\t¦ rte\t¦ rre_all\t¦ rte_all\t¦\n"

    success_flag = []
    rte_success, rre_success = [], []  # only success
    rte_all, rre_all = [], []  # Bo

    for idx, scene in enumerate(scene_names):
        success_flag_scene = []
        rte_success_scene, rre_success_scene = [], []  # only success
        rte_all_scene, rre_all_scene = [], []  # Both successful and failure cases

        # ground truth info
        gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
        n_valid = 0
        for ele in gt_pairs:
            diff = abs(int(ele[0]) - int(ele[1]))
            n_valid += diff > 1
        n_valids.append(n_valid)

        # estimated info
        est_pairs, est_traj = read_trajectory(os.path.join(est_folder, scenes[idx], 'est.log'))

        # Evaluate registration for each pair
        # assert est_traj.shape[0] == gt_traj.shape[0], 'Wrong number of estimated trajectories'
        for i in range(len(est_traj)):
            est_traj_inv = np.linalg.inv(est_traj[i])
            gt_traj_inv = np.linalg.inv(gt_traj[i])

            rot_error = compute_rre(est_traj_inv[:3, :3], gt_traj_inv[:3, :3])
            trans_error = compute_rte(est_traj_inv[:3, 3], gt_traj_inv[:3, 3])
            rre_all_scene.append(rot_error)
            rte_all_scene.append(trans_error)
            if rot_error < re_thres and trans_error < te_thres:
                success_flag_scene.append(True)
                rre_success_scene.append(rot_error)
                rte_success_scene.append(trans_error)
            else:
                success_flag_scene.append(False)

        benchmark_str += \
            "{}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}¦\n".format(
                short_names[idx],
                np.mean(success_flag_scene),
                np.mean(rre_success_scene), np.mean(rte_success_scene),
                np.mean(rre_all_scene), np.mean(rte_all_scene))

        success_flag = success_flag + success_flag_scene
        rre_success = rre_success + rre_success_scene
        rte_success = rte_success + rte_success_scene
        rre_all = rre_all + rre_all_scene
        rte_all = rte_all + rte_all_scene

    benchmark_str += \
        "Avg\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}¦\n".format(
            np.mean(success_flag),
            np.mean(rre_success), np.mean(rte_success),
            np.mean(rre_all), np.mean(rte_all))

    return benchmark_str
