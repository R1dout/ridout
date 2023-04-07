from torch.utils.data import DataLoader


def get_dataset_by_type(args, is_train=False):
    type2data = {

    }
    dataset = type2data[args.data_type]
    return dataset


def select_train_loader(args):
    train_dataset = get_dataset_by_type(args, True)
    print('{} samples found in train'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader


def select_eval_loader(args):
    eval_dataset = get_dataset_by_type(args)
    print('{} samples found in val'.format(len(eval_dataset)))
    val_loader = DataLoader(eval_dataset, 1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    return val_loader


