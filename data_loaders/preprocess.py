from .gdance.gdance_dataset import GroupDanceDataset

from .normalizer import Normalizer, ZNormalizer

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl

def preprocess_data(dataset, split='train'):
        data_path = dataset.datapath
        if split == 'train':
            seq_id_list = dataset._train_id_list
        elif split == 'test':
            seq_id_list = dataset._test_id_list


        all_pose_vec_input = []
        for seq_name in tqdm(seq_id_list):
            group_motion_data = dataset._load_motion_sequence(seq_name)
            seq_len = group_motion_data['group_poses'].shape[1]
            group_pose_vec = dataset._process_poses(group_motion_data, frame_ix = np.arange(seq_len)) # (n_persons, n_frames, 151)
            all_pose_vec_input.append(group_pose_vec.reshape(-1, group_pose_vec.shape[-1])) #(-1, 151)
        all_pose_vec_input = torch.cat(all_pose_vec_input) # (N_all, 151)
        print("Loaded all pose data with shape: ", all_pose_vec_input.shape)
        normalizer = ZNormalizer(all_pose_vec_input)
        print("Saving normalizer to normalizer.pkl")
        pkl.dump(normalizer, open(os.path.join(data_path, "normalizer.pkl"), "wb"))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str,  default="gdance", help="folder containing motions and music")
    args = parser.parse_args()

    if args.dataset_name == "gdance":
        dataset = GroupDanceDataset()

    preprocess_data(dataset, split='train')
    # normalizer = pkl.load(open("datasets/gdance_batch_01/normalizer.pkl","rb"))
    # print(normalizer.scaler.scale_.shape, normalizer.scaler.scale_)
    # print(normalizer.scaler.min_.shape, normalizer.scaler.min_)
    #
    # print("")
    # print(normalizer.scaler.data_min_)
    # print(normalizer.scaler.data_max_)







