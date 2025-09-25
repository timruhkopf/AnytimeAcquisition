import torch
def reward_fn(sequences):

    # TODO reward will be auc or reverse of remaining regret (if we calculate the volume)
    return torch.rand(sequences.shape[0])