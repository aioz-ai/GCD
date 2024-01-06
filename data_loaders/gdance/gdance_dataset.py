import random
from abc import abstractmethod
import numpy as np
import torch
from utils.misc import to_torch
import utils.rotation_conversions as geometry
import pandas as pd
import os.path as osp
import pickle as pkl
import cv2
from model.smpl_skeleton import SMPLSkeleton

class GroupDanceDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, datapath="datasets/gdance_batch_04/", split_file = "split_sequence_names.txt", target_seq_len=150, max_persons=2,
                 sampling="conseq", sampling_stride=1, split="train",
                 pose_rep="rot6d", translation=True, glob=True, use_contact = True, num_seq_max=-1, **kwargs
                 ):

        self.target_seq_len = target_seq_len
        self.max_persons = max_persons

        self.sampling = sampling
        self.sampling_stride = sampling_stride
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob #whether extract the root orient as part of the rotations
        self.use_contact = use_contact

        self.num_seq_max = num_seq_max


        self.datapath = datapath
        self.split_file = split_file
        self._train_id_list = pd.read_csv(osp.join(self.datapath, split_file), header=None).iloc[:,0].tolist()  # get the first column

        if use_contact:
            self.smpl = SMPLSkeleton()

        print("Dataset orig len (before filtering target_seq_len): ", len(self._train_id_list))
        # remove the sub-sequence with length < MIN_SAMPLE_LENGTH
        self._train_id_list = list(filter(lambda x: int(x.split("_")[-1]) - int(x.split("_")[-2]) >= target_seq_len , self._train_id_list)) # int(x.split("_")[-1]) = end; end - start >= target_seq_len









    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train_id_list), num_seq_max)
        else:
            return min(len(self._test_id_list), num_seq_max)


    def _load_music_features(self, seq_name):
        music_name = seq_name
        music_features = np.load(osp.join(self.datapath, f"jukebox_features/{music_name}.npy")) #expected shape (orig_nframes, 4800)

        return music_features

    def  _load_motion_sequence(self, seq_name):
        with open(osp.join(self.datapath, f"motions_smpl/{seq_name}.pkl"), "rb") as f:
            seq_data = pkl.load(f)

        smpl_poses = seq_data['smpl_poses'] #shape (n_persons, n_frames, pose_dim) , smpl pose excluding L/R hand joint
        if smpl_poses.shape[-1]<23*3:
            smpl_poses = np.concatenate([smpl_poses, np.zeros(list(smpl_poses.shape[:-1]) + [23*3 -smpl_poses.shape[-1] ])], axis=-1)
        if 'smpl_orients' in seq_data:
            smpl_orients = seq_data['smpl_orients']
            smpl_poses = np.concatenate([smpl_orients, smpl_poses], axis=-1)

        smpl_trans = seq_data['root_trans']

        return {
            'group_poses': smpl_poses,
            'group_trans': smpl_trans,
            'meta': seq_data['meta']
        }


    def _sample_sub_sequence(self, seq_len):
        """
        Sample a list of frame indices from 0 to seq_len
        """

        # if num_frames == -1 mode means that we do not process into the fixed motion sequence length for batching
        # frame_ix is just a list of frame indices of a video sequence to be loaded
        if self.target_seq_len == -1: # just take all frames of the sequence
            frame_ix = np.arange(seq_len)
        else:
            target_seq_len  = self.target_seq_len #desired num_frames

            # if specified or desired num_frames (e.g. 60)
            if seq_len < target_seq_len : #padding if the actual sequence length (nframes) is less than the desired num_frames
                # here we always pad using the last frame until achieving the desired num frames
                n_pad = max(0, target_seq_len - seq_len)
                lastframe = seq_len - 1
                padding = lastframe * np.ones(n_pad, dtype=int)
                frame_ix = np.concatenate([np.arange(0, seq_len), padding])
            elif self.sampling in ["conseq", "random_conseq"]: # randomly sampling consecutive frame sub-sequence
                stride_max = (seq_len - 1) // (target_seq_len - 1) #step_max is the maximum valid stride of the sequence
                if self.sampling == "conseq":
                    if self.sampling_stride == -1 or self.sampling_stride * (target_seq_len - 1) >= seq_len: #if the specified stride (self.sampling_stride) exceed the max valid number.
                        stride = stride_max
                    else:
                        stride = self.sampling_stride
                elif self.sampling == "random_conseq": # sample with random stride value
                    stride = random.randint(1, stride_max)

                lastone = stride * (target_seq_len - 1)
                start_fr_max = seq_len - lastone - 1

                shift = random.randint(0, max(0, start_fr_max - 1)) # random start_frame
                # shift = 0
                frame_ix = shift + np.arange(0, lastone + 1, stride)

            else:
                raise ValueError("Sampling not recognized.")

        return frame_ix


    def _process_poses(self, group_motion_data, frame_ix):
        pose_rep = self.pose_rep
        group_poses = group_motion_data['group_poses'][:,frame_ix] #shape (n_persons, target_seq_len, 72)
        group_poses = to_torch(group_poses)
        n_persons, seq_len = group_poses.shape[:2]
        group_poses = group_poses.view(n_persons, seq_len, -1, 3)  # reshape to (n_persons, seq_len, J, 3)


        if self.translation: #load the root translation vector if using 'rot6d'
            group_trans = group_motion_data['group_trans'][:, frame_ix]
            group_trans = to_torch(group_trans)

        if self.use_contact:

            smpl = self.smpl
            positions = smpl.forward(group_poses, group_trans)  # batch x sequence x J x 3
            feet = positions[:, :, (7, 8, 10, 11)]
            feetv = torch.zeros(feet.shape[:3])
            feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
            contacts = (feetv < 0.01).to(group_poses)  # cast to right dtype, (batch x sequence x 4)


        if pose_rep == "xyz":
            #by now we only consider pose_rep == 'rot6d'
            pass
        elif pose_rep != "xyz":

            if not self.glob: # if not using root orient
                group_poses = group_poses[:, : , 1:, :]
            group_poses = to_torch(group_poses)

            if pose_rep == "rotvec": #3d axis-angle
                pass
            elif pose_rep == "rotmat": # 9d rotation matrix
                group_poses = geometry.axis_angle_to_matrix(group_poses).view(*group_poses.shape[:3], 9)
            elif pose_rep == "rotquat": # 4d quaternion
                group_poses = geometry.axis_angle_to_quaternion(group_poses)
            elif pose_rep == "rot6d":
                group_poses = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(group_poses)) # shape (n_persons, seq_len, J , 6)



        group_poses = group_poses.reshape(n_persons, seq_len, -1) #(n_persons * seq_len, J*6)

        ret = group_poses

        if pose_rep != "xyz" and self.translation:
            ret = torch.cat([group_trans, ret], dim=-1) # pose = [trans; rots], (n_persons,seq_len, J*6+3)

        if self.use_contact:
            ret = torch.cat([ret, contacts], dim=-1)  # pose = [trans; rots; contacts] , (n_persons,seq_len, J*6+3+4)

        ret = ret.float()

        # TODO: normalize the data
        if hasattr(self, "normalizer"):
            ret = self.normalizer.normalize(ret)


        return ret


    def sample_sequence(self, idx):
        '''
        Sample small sub-sequence from a big video sequence from (0 to n_frames)
        each sampled sequence should have shape: (num_persons, seq_len, smpl_dim)
        the sequence is then padded to shape (max_persons, max_len, smpl_dim)
        '''

        if self.split == 'train':
            seq_name = self._train_id_list[idx]
        else:
            seq_name = self._test_id_list[idx]


        group_motion_data = self._load_motion_sequence(seq_name)
        orig_start, orig_end = group_motion_data['meta']['orig_start'], group_motion_data['meta']['orig_end']
        orig_seq_len = group_motion_data['group_poses'].shape[1]  # actual sequence length
        music_features = self._load_music_features(seq_name) # [orig_start:orig_end] #need slicing because the music features were extracted from the whole orig video (both are aligned at the first frame)

        sampled_frame_ix = self._sample_sub_sequence(seq_len = group_motion_data['group_poses'].shape[1])

        group_poses = self._process_poses(group_motion_data, sampled_frame_ix)
        music_features = music_features[sampled_frame_ix]

        # sampling the sub-sequence according to max_seq_len in the sampled sub-sequence
        fr_start = sampled_frame_ix[0]
        eff_seq_len = min(self.target_seq_len, orig_seq_len) if self.target_seq_len != -1 else orig_seq_len
        if orig_seq_len >= self.target_seq_len:
            frame_mask = np.ones((eff_seq_len,)).astype(np.float32)
        else:  # else we need to pad the sequence if the sample_seq_len < desired max_seq_len
            frame_mask = np.zeros((self.target_seq_len,)).astype(np.float32)
            frame_mask[:eff_seq_len] = 1.0


        # pad n_persons to max_persons
        eff_n_persons = group_motion_data['group_poses'].shape[0]
        if eff_n_persons >= self.max_persons:
            sampled_persons = np.random.choice(eff_n_persons, size=self.max_persons, replace=False)
            group_poses = group_poses[sampled_persons]
            eff_n_persons = self.max_persons
            data_mask = np.tile(frame_mask, (self.max_persons, 1))
            person_mask = np.ones((self.max_persons, )).astype(np.float32)
        else:
            #padding
            group_poses = torch.cat([
                group_poses,
                torch.zeros([self.max_persons - eff_n_persons] + list(group_poses.shape[1:]))
            ], dim=0)

            data_mask = np.tile(frame_mask, (self.max_persons, 1))
            data_mask[eff_n_persons:] = 0
            person_mask = np.ones((self.max_persons, )).astype(np.float32)
            person_mask[eff_n_persons:] = 0


        output = {}
        output['meta'] = {}
        output['meta']['seq_name'] = seq_name
        output['meta']['orig_lengths'] = orig_seq_len
        output['meta']['fr_start'] =  fr_start
        output['meta']['eff_seq_len'] = eff_seq_len
        output['meta']['eff_n_persons'] = eff_n_persons


        output['frame_mask'] = frame_mask  # shape (target_seq_len,)
        output['person_mask'] = person_mask  # shape (max_persons, )
        output['data_mask'] = data_mask  # shape (max_persons, target_seq_len)

        output['inp'] = group_poses
        output['lengths'] = eff_seq_len
        output['n_persons'] = eff_n_persons
        output['music'] = to_torch(music_features).float()

        return output

    def __getitem__(self, idx):


        data_dict = self.sample_sequence(idx)

        motion = data_dict.pop('inp')
        cond = {'y': data_dict}

        return  motion, cond


if __name__ == "__main__":
    dataset = GroupDanceDataset(target_seq_len=1000 ,max_persons=10)
    print(len(dataset))
    # for motion, cond in dataset:
    #     print(motion.shape, cond['y'].keys())
    #     print(cond['y']['meta']['seq_name'])
    #     print(cond['y']['data_mask'])
    #     # print(data['meta']['seq_name'], data['meta']['orig_lengths'], data['lengths'],  data['inp'].shape, data['music'].shape)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    for motion, cond in dataloader:
        print(cond['y']['meta']['seq_name'])
        print("Full sequence lengths:", cond['y']['meta']['orig_lengths'])
        print("Actual Lengths: ", cond['y']['lengths'])
        print('Motion shape: ', motion.shape)
        print("Music shape: ", cond['y']['music'].shape)
        print(cond['y']['frame_mask'].shape)

