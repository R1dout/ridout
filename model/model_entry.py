import torch.nn as nn


def select_model(args):
    type2model = {

    }
    model = type2model[args.model_type]
    return model
