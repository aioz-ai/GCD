# BSD License

# For fairmotion software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Ruilong Li

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))




import numpy as np
import utils as feat_utils
import multiprocessing


try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default
from functools import partial
from joblib import delayed, Parallel


def extract_kinetic_features(positions):
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32)
    return kinetic_feature_vector


def extract_kinetic_features_per_frame(positions):
    m=1.5
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        # feature_vector = np.stack(
        #     [
        #         m*features.average_kinetic_energy_horizontal(i, return_per_frame=True),
        #         m*features.average_kinetic_energy_vertical(i, return_per_frame=True),
        #         np.sqrt(m)*features.average_energy_expenditure(i,return_per_frame=True),
        #     ],
        #     axis=1
        # )

        feature_vector = np.stack(
            [
                np.log(1 + m*features.average_kinetic_energy_horizontal(i, return_per_frame=True)),
                np.log(1 + m*features.average_kinetic_energy_vertical(i, return_per_frame=True)),
                np.log(1 + m*features.average_energy_expenditure(i,return_per_frame=True)),
            ],
            axis=1
        )
        # print(feature_vector.shape) (T-1, 3)

        # feature_vector = (np.array(features.average_kinetic_energy(i, return_per_frame=True, no_norm=True)))#.reshape(-1,1)
        # feature_vector = (np.array(features.average_kinetic_energy(i, return_per_frame=True))) .reshape(-1,1)

        # feature_vector = np.log(1 + feature_vector)
        kinetic_feature_vector.append(feature_vector)
    kinetic_feature_vector = np.concatenate(kinetic_feature_vector, axis=1)
    return kinetic_feature_vector


class KineticFeatures:
    def __init__(
        self, positions, frame_time=1./30, up_vec="y", sliding_window=2
    ):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint, return_per_frame = False, no_norm=False):
        # average_kinetic_energy = 0
        # for i in range(1, len(self.positions)):
        #     average_velocity = feat_utils.calc_average_velocity(
        #         i, self.positions, joint, self.sliding_window, self.frame_time
        #     )
        #     average_kinetic_energy += average_velocity ** 2
        # average_kinetic_energy = average_kinetic_energy / (len(self.positions) - 1.0)
        #
        #
        # return average_kinetic_energy

        # Parallelized version of the above calculations
        val1 = 0
        average_velocities = [feat_utils.calc_average_velocity(i, self.positions, joint, self.sliding_window, self.frame_time, no_norm) for i in range(1, len(self.positions))]
        val1 = np.array(average_velocities) ** 2
        if return_per_frame:
            return val1 #shape (N_frames-1, )
        val1 = val1.sum() /  (len(self.positions) - 1.0)


        return val1

    def average_kinetic_energy_horizontal(self, joint, return_per_frame=False):
        # val = 0
        # for i in range(1, len(self.positions)):
        #     average_velocity = feat_utils.calc_average_velocity_horizontal(
        #         i,
        #         self.positions,
        #         joint,
        #         self.sliding_window,
        #         self.frame_time,
        #         self.up_vec,
        #     )
        #
        #     val += average_velocity ** 2
        # val = val / (len(self.positions) - 1.0)
        # return val

        # # Parallelized version of the above calculations
        val1 = 0
        # process_func = partial(feat_utils.calc_average_velocity_horizontal,
        #     positions = self.positions, joint_idx = joint, sliding_window= self.sliding_window, frame_time = self.frame_time, up_vec = self.up_vec,
        # )
        # pool = multiprocessing.Pool(processes=cpus)
        # average_velocities = pool.map(process_func, range(1, len(self.positions)))
        # average_velocities = Parallel(n_jobs=cpus)(delayed(process_func)(i) for i in range(1, len(self.positions)))
        # average_velocities = [process_func(i) for i in range(1, len(self.positions))]
        average_velocities = [feat_utils.calc_average_velocity_horizontal(i, self.positions, joint, self.sliding_window, self.frame_time, self.up_vec) for i in range(1, len(self.positions))]
        val1 = np.array(average_velocities) ** 2
        if return_per_frame:
            return val1 #shape (N_frames, )

        val1 = val1.sum() /  (len(self.positions) - 1.0)


        return val1

    def average_kinetic_energy_vertical(self, joint, return_per_frame=False):
        # val = 0
        # for i in range(1, len(self.positions)):
        #     average_velocity = feat_utils.calc_average_velocity_vertical(
        #         i,
        #         self.positions,
        #         joint,
        #         self.sliding_window,
        #         self.frame_time,
        #         self.up_vec,
        #     )
        #     val += average_velocity ** 2
        # val = val / (len(self.positions) - 1.0)
        #
        # return val


        # Parallelized version of the above calculations
        val1 = 0
        # process_func = partial(feat_utils.calc_average_velocity_vertical,
        #                        positions=self.positions, joint_idx=joint, sliding_window=self.sliding_window,
        #                        frame_time=self.frame_time, up_vec=self.up_vec,
        #                        )
        # pool = multiprocessing.Pool(processes=cpus)
        # average_velocities = pool.map(process_func, range(1, len(self.positions)))
        # average_velocities = Parallel(n_jobs=cpus)(delayed(process_func)(i) for i in range(1, len(self.positions)))
        # average_velocities = [process_func(i) for i in range(1, len(self.positions))]
        average_velocities = [feat_utils.calc_average_velocity_vertical(i, self.positions, joint, self.sliding_window, self.frame_time, self.up_vec) for i in range(1, len(self.positions))]
        val1 = np.array(average_velocities) ** 2
        if return_per_frame:
            return val1 #shape (N_frames, )
        val1 = val1.sum() / (len(self.positions) - 1.0)

        return val1

    def average_energy_expenditure(self, joint, return_per_frame=False):
        # val = 0.0
        # for i in range(1, len(self.positions)):
        #     val += feat_utils.calc_average_acceleration(
        #         i, self.positions, joint, self.sliding_window, self.frame_time
        #     )
        # val = val / (len(self.positions) - 1.0)

        # return val

        # Parallelized version of the above calculations
        val1 = 0
        # process_func = partial(feat_utils.calc_average_acceleration,
        #                        positions=self.positions, joint_idx=joint, sliding_window=self.sliding_window,
        #                        frame_time=self.frame_time
        #                        )
        # pool = multiprocessing.Pool(processes=cpus)
        # average_accelerations = pool.map(process_func, range(1, len(self.positions)))
        # average_accelerations = Parallel(n_jobs=cpus)(delayed(process_func)(i) for i in range(1, len(self.positions)))
        # average_accelerations = [process_func(i) for i in range(1, len(self.positions))]
        average_accelerations = [feat_utils.calc_average_acceleration(i, self.positions, joint, self.sliding_window, self.frame_time) for i in range(1, len(self.positions))]
        val1 = np.array(average_accelerations)
        if return_per_frame:
            return val1 #shape (N_frames, )
        val1 = val1.sum() / (len(self.positions) - 1.0)


        return val1


if __name__ =="__main__":
    np.random.seed(10)
    joints3d = np.random.randn(500,24,3)
    features = extract_kinetic_features(joints3d)
    print(features.shape)

    print("Per frame features:")
    features = extract_kinetic_features_per_frame(joints3d)
    print(features.shape)