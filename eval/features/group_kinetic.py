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


def extract_group_kinetic_features(positions):
    m = 1.5
    assert len(positions.shape) == 4  # (N, seq_len, n_joints, 3)
    feature_extractor = GroupKineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[2]): #for each joint
        # feature_vector = np.hstack(
        #     [
        #         m*feature_extractor.average_kinetic_energy_horizontal(i),
        #         m*feature_extractor.average_kinetic_energy_vertical(i),
        #         m*feature_extractor.average_energy_expenditure(i),
        #     ]
        # )
        feature_vector = np.hstack(
            [
                np.log(1+ m*feature_extractor.average_kinetic_energy_horizontal(i)),
                np.log(1+ m*feature_extractor.average_kinetic_energy_vertical(i)),
                np.log(1+ m*feature_extractor.average_energy_expenditure(i)),
            ]
        )

        # feature_vector = (np.array(feature_extractor.average_kinetic_energy(i, no_norm=True)))#.reshape(-1,1)
        # feature_vector = (np.array(feature_extractor.average_kinetic_energy(i))).reshape(1)
        # feature_vector = np.log(1 + feature_vector)

        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32) #(72,)

    return kinetic_feature_vector



class GroupKineticFeatures:
    def __init__(
        self, positions, frame_time=1./30, up_vec="y", sliding_window=1
    ):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint, no_norm=False):
        # average_kinetic_energy = 0
        # for i in range(1, len(self.positions)):
        #     average_velocity = feat_utils.calc_average_velocity(
        #         i, self.positions,  joint, self.sliding_window, self.frame_time, no_norm
        #     )
        #     average_kinetic_energy += average_velocity ** 2
        # average_kinetic_energy = average_kinetic_energy / (len(self.positions) - 1.0)
        # return average_kinetic_energy

        val = 0
        N, T = self.positions.shape[:2]
        for n in range(N):  # n-th person
            for i in range(1, T):  # i-th frame
                average_velocity = feat_utils.calc_average_velocity(
                    i,
                    self.positions[n],
                    joint,
                    self.sliding_window,
                    self.frame_time,
                    no_norm
                )
                val += average_velocity ** 2
        val = val / (T - 1.0) / N
        return val

    def average_kinetic_energy_horizontal(self, joint):
        val = 0
        N, T = self.positions.shape[:2]
        for n in range(N): # n-th person
            for i in range(1, T): #i-th frame
                average_velocity = feat_utils.calc_average_velocity_horizontal(
                    i,
                    self.positions[n],
                    joint,
                    self.sliding_window,
                    self.frame_time,
                    self.up_vec,
                )
                val += average_velocity ** 2
        val = val / (T - 1.0) / N
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        N, T = self.positions.shape[:2]
        for n in range(N): # n-th person
            for i in range(1, T): #i-th frame
                average_velocity = feat_utils.calc_average_velocity_vertical(
                    i,
                    self.positions[n],
                    joint,
                    self.sliding_window,
                    self.frame_time,
                    self.up_vec,
                )
                val += average_velocity ** 2
        val = val / (T - 1.0) / N
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        N, T = self.positions.shape[:2]
        for n in range(N): # n-th person
            for i in range(1, T): #i-th frame
                val += feat_utils.calc_average_acceleration(
                    i, self.positions[n], joint, self.sliding_window, self.frame_time
                )
        val = val / (T - 1.0) / N
        return val


if __name__ =="__main__":
    joints3d = np.random.randn(3, 300,24,3)
    features = extract_group_kinetic_features(joints3d)
    print(features.shape)

