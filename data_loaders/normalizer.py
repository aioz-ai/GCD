import glob
import os
import re
from pathlib import Path

import torch



def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


class Normalizer:
    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])
        self.scaler = MinMaxScaler((-1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        orig_shape = x.shape
        data_dim = x.shape[-1]
        x = x.reshape(-1, data_dim)
        return self.scaler.transform(x).reshape(orig_shape)

    def unnormalize(self, x):
        orig_shape = x.shape
        data_dim = x.shape[-1]
        x = x.reshape(-1, data_dim)
        x = torch.clip(x, -1, 1)  # clip to force compatibility
        return self.scaler.inverse_transform(x).reshape(orig_shape)

class ZNormalizer:
    def __init__(self, data, eps=1e-10):
        flat = data.reshape(-1, data.shape[-1])
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0)
        self.eps = eps

    def normalize(self, x, start_dim=0, end_dim=None):

        mean = self.mean.to(x.device)[..., start_dim:end_dim]
        std = self.std.to(x.device)[..., start_dim:end_dim]
        orig_shape = x.shape
        data_dim = x.shape[-1]
        x = x.reshape(-1, data_dim)
        return ((x - mean)/(std + self.eps)).reshape(orig_shape)

    def unnormalize(self, x, start_dim=0, end_dim=None):

        mean = self.mean.to(x.device)[..., start_dim:end_dim]
        std = self.std.to(x.device)[..., start_dim:end_dim]
        orig_shape = x.shape
        data_dim = x.shape[-1]
        x = x.reshape(-1, data_dim)
        return (x * std + mean).reshape(orig_shape)


def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt




def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    # if we are fitting on 1D arrays, scale might be a scalar
    if constant_mask is None:
        # Detect near constant values to avoid dividing by a very small
        # value that could lead to surprising results and numerical
        # stability issues.
        constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

    if copy:
        # New array to avoid side-effects
        scale = scale.clone()
    scale[constant_mask] = 1.0
    return scale


class MinMaxScaler:
    _parameter_constraints: dict = {
        "feature_range": [tuple],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X):
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        data_min = torch.min(X, axis=0)[0]
        data_max = torch.max(X, axis=0)[0]

        self.n_samples_seen_ = X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        X *= self.scale_.to(X.device)
        X += self.min_.to(X.device)
        if self.clip:
            torch.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        return X

    def inverse_transform(self, X):
        X -= self.min_[-X.shape[1] :].to(X.device)
        X /= self.scale_[-X.shape[1] :].to(X.device)
        return X




