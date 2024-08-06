
import os
from librosa import beat
import time
import torch
import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal

import argparse

import glob
import tqdm
from smplx import SMPL

from dtaidistance import dtw_ndim, dtw
from features.kinetic import extract_kinetic_features_per_frame




def cross_corr(a,b):
    #print(np.dot(a,b))
    return np.dot(a,b)


def calculate_gmc_score_gt(seq_name, motion_dir, smpl):
    # get real data motion beats
    data = pkl.load(open(os.path.join(motion_dir, f'{seq_name}.pkl'),"rb"))
    smpl_poses = data['smpl_poses']
    smpl_trans = data['root_trans']

    N, T = smpl_poses.shape[:2]
    J = smpl.NUM_JOINTS  # 23

    global_orient = smpl_poses[:,:,:3].reshape(N, -1, 3)
    body_pose = smpl_poses[:,:,3:].reshape(N, -1, J*3)
    smpl_trans = smpl_trans.reshape(N, -1,3)

    all_keypoints3d = []
    for n in range(N): #for each individual person
        # keypoints3d = all_joints3d[n]
        with torch.no_grad():
            keypoints3d = smpl.forward(
                global_orient=torch.from_numpy(global_orient[n]).float().to(args.device),
                body_pose=torch.from_numpy(body_pose[n]).float().to(args.device),
                transl=torch.from_numpy(smpl_trans[n]).float().to(args.device),
            ).joints.detach().cpu().numpy().reshape(T, -1, 3)[:, 0:24, :]  # only take 24 standard joints
        all_keypoints3d.append(keypoints3d)
    all_keypoints3d = np.stack(all_keypoints3d, axis=0) # (N,T,J,3)

    features = [ extract_kinetic_features_per_frame(all_keypoints3d[n]) for n in range(N)]



    # calculate for each pair:
    scores = 0
    cnt = 0
    for n1 in range(N):
        for n2 in range(n1+1,N):
            # compute time-warping path
            path_matrix = dtw_ndim.warping_paths(all_keypoints3d[n1], all_keypoints3d[n2])[1]
            best_path = dtw.best_path(path_matrix) # list of matched pairs (which frame_n1 corresponds to each frame_n2)

            features_1 = features[n1];
            features_2 = features[n2];

            pp = 0
            val =0
            for (t1,t2) in best_path:
                if t1 >= T-1 or t2>=T-1: continue
                val += cross_corr(features_1[t1], features_2[t2])
                pp +=1
            val = val / pp #normalize by number of frames
            print(val, pp)

            scores += val
            cnt += 1

    print(scores/cnt)
    return scores / cnt

def calculate_gmc_score_generated(result_file, motion_dir, smpl):

    data = pkl.load(open(result_file,"rb"))

    transl = data['transl'] # (N,T, 3)
    body_pose = data['body_pose'] # (N, T, 69)
    global_orient = data['global_orient'] # (N, T, 3)


    N, T = body_pose.shape[:2]
    J = smpl.NUM_JOINTS  # 23

    all_keypoints3d = []
    for n in range(N):  # for each individual person
        # keypoints3d = all_joints3d[n]
        with torch.no_grad():
            keypoints3d = smpl.forward(
                global_orient=torch.from_numpy(global_orient[n]).float().to(args.device),
                body_pose=torch.from_numpy(body_pose[n]).float().to(args.device),
                transl=torch.from_numpy(transl[n]).float().to(args.device),
            ).joints.detach().cpu().numpy().reshape(T, -1, 3)[:, 0:24, :]  # only take 24 standard joints
        all_keypoints3d.append(keypoints3d)
    all_keypoints3d = np.stack(all_keypoints3d, axis=0)  # (N,T,J,3)

    features = [extract_kinetic_features_per_frame(all_keypoints3d[n]) for n in range(N)]

    # calculate for each pair:
    scores = 0
    cnt = 0
    for n1 in range(N):
        for n2 in range(n1 + 1, N):
            # compute time-warping path
            path_matrix = dtw_ndim.warping_paths(all_keypoints3d[n1], all_keypoints3d[n2])[1]
            best_path = dtw.best_path(
                path_matrix)  # list of matched pairs (which frame_n1 corresponds to each frame_n2)

            features_1 = features[n1];
            features_2 = features[n2];

            pp = 0
            val = 0
            for (t1, t2) in best_path:
                if t1 >= T - 1 or t2 >= T - 1: continue
                val += cross_corr(features_1[t1], features_2[t2])
                pp += 1
            val = val / pp  # normalize by number of frames
            print(val, pp)

            scores += val
            cnt += 1

    print(scores / cnt)
    return scores / cnt

def main():

    motion_dir = os.path.join(args.data_dir, "motions_smpl/")

    # set smpl
    smpl = SMPL(model_path=args.smpl_dir, gender='MALE', batch_size=1).to(args.device)

    # create list of sequence names
    seq_names = []
    if "train" in args.split:
        seq_names += np.loadtxt(os.path.join(args.data_dir, "train_split_sequence_names.txt"), dtype=str).tolist()
    if "val" in args.split:
        seq_names += np.loadtxt(os.path.join(args.data_dir, "val_split_sequence_names.txt"), dtype=str).tolist()
    if "test" in args.split:
        seq_names += np.loadtxt(os.path.join(args.data_dir, "test_split_sequence_names.txt"), dtype=str).tolist()
    ignore_list = []
    seq_names = [name for name in seq_names if name not in ignore_list]

    # # calculate score on real data
    # n_samples = len(seq_names)
    # beat_scores = []
    # for i, seq_name in enumerate(seq_names):
    #     print("processing %d / %d" % (i + 1, n_samples))
    #     sample_beat_score = calculate_gmc_score_gt(seq_name, motion_dir, smpl)
    #     beat_scores.append(sample_beat_score)
    # print ("\GMC score on real data: %.3f\n" % (sum(beat_scores) / n_samples))

    # calculate score on generated motion data
    result_files = sorted(glob.glob(args.result_files))
    n_samples = 0 #len(result_files)
    generated_gmc_scores = []
    for result_file in tqdm.tqdm(result_files):
        generated_gmc_score = calculate_gmc_score_generated(result_file,  motion_dir, smpl)
        if generated_gmc_score:
            generated_gmc_scores.append(generated_gmc_score)
            n_samples += 1
            # if n_samples > 20: break
    print ("\nGMC score on generated data: %.3f\n" % (sum(generated_gmc_scores) / n_samples))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='datasets/gdance/')
    parser.add_argument('--split', type=str, default='testval', choices=['train', 'testval'])
    parser.add_argument('--result_files', type=str, default='infer_test_out/*.pkl')

    parser.add_argument('--smpl_dir', type=str, default='models_smpl/', help='input local dictionary that stores SMPL data.')
    parser.add_argument('--device',type=str, default='cuda',)
    args = parser.parse_args()



    main()