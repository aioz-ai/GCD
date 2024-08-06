import multiprocessing


import vedo
import torch
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg

# See https://github.com/google/aistplusplus_api/ for installation
from features.group_kinetic import extract_group_kinetic_features
import glob
import tqdm
from smplx import SMPL
import pickle as pkl
import argparse






def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    Code apapted from https://github.com/mseitzer/pytorch-fid

    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    mu and sigma are calculated through:
    ```
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    ```
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]  # (seq_len, 24, 3)
    return keypoints3d


def extract_group_feature(keypoints3d, mode="kinetic"):
    if mode == "kinetic":
        feature = extract_group_kinetic_features(keypoints3d)
    else:
        raise ValueError("%s is not support!" % mode)
    return feature  # (f_dim,)


def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    frechet_dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    avg_dist = calculate_avg_distance(feature_list2)
    return frechet_dist, avg_dist

def load_motion_keypoints_generated(result_file,smpl):
    data = pkl.load(open(result_file,"rb"))

    transl = data['transl'] # (N,T, 3)
    body_pose = data['body_pose'] # (N, T, 69)
    global_orient = data['global_orient'] # (N, T, 3)

    N, T = body_pose.shape[:2]
    J = smpl.NUM_JOINTS  # 23

    global_orient = global_orient.reshape(-1, 3)
    body_pose = body_pose.reshape(-1, J*3)
    transl = transl.reshape(-1,3)

    with torch.no_grad():
        all_keypoints3d  = smpl.forward(
            global_orient=torch.from_numpy(global_orient).float().to(args.device),
            body_pose=torch.from_numpy(body_pose).float().to(args.device),
            transl=torch.from_numpy(transl).float().to(args.device),
        ).joints.detach().cpu().numpy().reshape(N,T,-1,3)[:, :, 0:24, :] #only take 24 standard joints

    # all_keypoints3d = []
    # for n in range(N): #for each individual person
    #     # keypoints3d = all_joints3d[n]
    #     with torch.no_grad():
    #         keypoints3d = smpl.forward(
    #             global_orient=torch.from_numpy(global_orient[n]).float().to(args.device),
    #             body_pose=torch.from_numpy(body_pose[n]).float().to(args.device),
    #             transl=torch.from_numpy(transl[n]).float().to(args.device),
    #         ).joints.detach().cpu().numpy().reshape(T, -1, 3)[:, 0:24, :]  # only take 24 standard joints
    #     all_keypoints3d.append(keypoints3d)
    # all_keypoints3d = np.stack(all_keypoints3d, axis=0) # (N,T,J,3)


    return all_keypoints3d






if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--result_files', type=str, default='infer_test_out/*.pkl')

    parser.add_argument('--smpl_dir', type=str, default='models_smpl/', help='input local dictionary that stores SMPL data.')
    parser.add_argument('--device',type=str, default='cpu',)
    args = parser.parse_args()

    # set smpl
    smpl = SMPL(model_path=args.smpl_dir, gender='MALE', batch_size=1).to(args.device)

    # get cached motion features for the real data
    real_features = {
        "kinetic": [np.load(f) for f in sorted(glob.glob("./datasets/gdance/eval_group_features/*_group_kinetic.npy"))],
    }

    real_features['kinetic'] = np.stack(real_features['kinetic'], axis=0) # shape (N_samples, 72)

    # get motion features for the results
    result_features = {"kinetic": []}
    result_files = sorted(glob.glob(args.result_files))
    cnt = 0
    for result_file in tqdm.tqdm(result_files):
        keypoints3d = load_motion_keypoints_generated(result_file,smpl) #(N,T,J,3)

        # visualize(result_motion, smpl)
        sample_features = extract_group_feature(keypoints3d, "kinetic") #(72,)

        result_features["kinetic"].append(sample_features)

        # cnt += 1
        # if cnt>10: break




    result_features['kinetic'] = np.stack(result_features['kinetic'], axis=0)  # shape (N_samples, 72)

    # FID metrics
    FID_k, Dist_k = calculate_frechet_feature_distance(
        real_features["kinetic"], result_features["kinetic"])


    # Evaluation: FID_k: ~32, FID_g: ~17
    # Evaluation: Dist_k: ~6, Dist_g: ~6
    # The AIChoreo paper used a bugged version of manual feature extractor from
    # fairmotion (see here: https://github.com/facebookresearch/fairmotion/issues/50)
    # So the FID_g here does not match with the paper. But this value should be correct.
    # In this aistplusplus_api repo the feature extractor bug has been fixed.
    # (see here: https://github.com/google/aistplusplus_api/blob/main/aist_plusplus/features/manual.py#L50)
    print('\nEvaluation: GMR_k: {:.4f}'.format(FID_k))
    print('Evaluation: Dist_k: {:.4f}\n'.format(Dist_k))