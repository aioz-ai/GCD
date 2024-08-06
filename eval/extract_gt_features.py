import os
import numpy as np
import argparse

import sys
from features.kinetic import extract_kinetic_features
from smplx import SMPL

import torch
import multiprocessing
import functools

import pickle as pkl


def load_motion(motion_dir, seq_name, J=23):
    data = pkl.load(open(os.path.join(motion_dir, seq_name),"rb"))



    smpl_poses = data['smpl_poses']
    smpl_trans = data['root_trans']


    global_orient = smpl_poses[:,:,:3].reshape(-1, 3)
    body_pose = smpl_poses[:,:,3:].reshape(-1, J*3)
    smpl_trans = smpl_trans.reshape(-1,3)

    return global_orient, body_pose,smpl_trans


def main(seq_name, motion_dir):
    # Parsing SMPL 24 joints.
    # Note here we calculate `transl` as `smpl_trans/smpl_scaling` for
    # normalizing the motion in generic SMPL model scale.
    smpl = SMPL(model_path=args.smpl_dir, gender='MALE', batch_size=1).to(args.device)
    J = smpl.NUM_JOINTS #23

    print(seq_name)

    # load motion
    data = pkl.load(open(os.path.join(motion_dir, f'{seq_name}.pkl'),"rb"))

    smpl_poses = data['smpl_poses']
    smpl_trans = data['root_trans']

    N, T = smpl_poses.shape[:2]
    J = smpl.NUM_JOINTS  # 23

    global_orient = smpl_poses[:,:,:3].reshape(N, -1, 3)
    body_pose = smpl_poses[:,:,3:].reshape(N, -1, J*3)
    smpl_trans = smpl_trans.reshape(N, -1,3)

    # global_orient = smpl_poses[:,:,:3].reshape(-1, 3)
    # body_pose = smpl_poses[:,:,3:].reshape(-1, J*3)
    # smpl_trans = smpl_trans.reshape(-1,3)


    # all_joints3d  = smpl.forward(
    #     global_orient=torch.from_numpy(global_orient).float().to(args.device),
    #     body_pose=torch.from_numpy(body_pose).float().to(args.device),
    #     transl=torch.from_numpy(smpl_trans).float().to(args.device),
    # ).joints.detach().cpu().numpy().reshape(N,T,-1,3)[:, :, 0:24, :] #only take 24 standard joints

    all_features = []
    for n in range(N):
        # keypoints3d = all_joints3d[n]
        with torch.no_grad():
            keypoints3d = smpl.forward(
                global_orient=torch.from_numpy(global_orient[n]).float().to(args.device),
                body_pose=torch.from_numpy(body_pose[n]).float().to(args.device),
                transl=torch.from_numpy(smpl_trans[n]).float().to(args.device),
            ).joints.detach().cpu().numpy().reshape(T, -1, 3)[:, 0:24, :]  # only take 24 standard joints

        features = extract_kinetic_features(keypoints3d) #extract feats for each individual, (72,)
        all_features.append(features)
    all_features = np.stack(all_features, axis=0) #(N, 72)
    print(all_features.shape);
    np.save(os.path.join(args.save_dir, seq_name + "_kinetic.npy"), all_features)
    print(seq_name, "is done")

    del smpl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='datasets/gdance/')
    parser.add_argument(
        '--smpl_dir',
        type=str,
        default='models_smpl/',
        help='input local dictionary that stores SMPL data.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./datasets/gdance/eval_features/',
        help='output local dictionary that stores features.')
    parser.add_argument('--device',type=str, default='cuda',)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Parsing data info.

    seq_names = []
    # seq_names += np.loadtxt(os.path.join(args.data_dir, "train_split_sequence_names.txt"), dtype=str).tolist()
    seq_names += np.loadtxt(os.path.join(args.data_dir, "val_split_sequence_names.txt"), dtype=str).tolist()
    seq_names += np.loadtxt(os.path.join(args.data_dir, "test_split_sequence_names.txt"), dtype=str).tolist()


    # processing
    process = functools.partial(main, motion_dir=os.path.join(args.data_dir, "motions_smpl"))
    pool = multiprocessing.Pool(2)
    pool.map(process, seq_names)
    # for seq_name in seq_names:
    #     process(seq_name)

