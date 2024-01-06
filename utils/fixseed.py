import numpy as np
import torch
import random


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True



# SEED = 10
# EVALSEED = 0
# # Provoc warning: not fully functionnal yet
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# fixseed(SEED)
