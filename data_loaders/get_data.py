from torch.utils.data import DataLoader

def get_dataset_class(name):
    if name == 'aist':
        from .audio2dance.aist_dataset import AistDataset
        return AistDataset
    elif name =='gdance':
        from .gdance.gdance_dataset import GroupDanceDataset
        return GroupDanceDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name):
    if name == 'gdance':
        return None # not using collate since we did this inside the dataset class
    elif name =='aist':
        return None
        # return all_collate


def get_dataset(name, datapath, split_file, target_seq_len, max_persons, split='train'):
    DATA = get_dataset_class(name) #get the  dataset class corresponding to name
    dataset = DATA(datapath=datapath, split_file=split_file,
                   split=split, target_seq_len=target_seq_len, max_persons=max_persons)
    return dataset


def get_dataset_loader(name, datapath, split_file, batch_size, target_seq_len, max_persons, split='train', num_workers = 8, shuffle=True):
    dataset = get_dataset(name, datapath, split_file, target_seq_len, max_persons, split)
    collate = get_collate_fn(name)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers = num_workers,
        drop_last = False,
        collate_fn=collate,
        pin_memory = True,
        persistent_workers=True if num_workers > 0 else False, #https://github.com/Lightning-AI/lightning/issues/10389
    )
    return loader

if __name__ == "__main__":
    dataloader = get_dataset_loader(name="aist", batch_size=32, target_seq_len=1200)
    print(dataloader.dataset, "\n")
    for motion, cond in dataloader:

        print(cond['y']['lengths'])
        print(cond['y']['frame_mask'].shape, cond['y']['frame_mask'])
        print(cond['y']['music'].shape)
        print(motion.shape)
        print(motion.permute(0, 3, 1, 2)[0, :10, -1])  # root trans
        print("="*50)

        break



