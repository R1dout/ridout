import numpy as np

def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_lr(optim):
    return optim.param_groups[0]["lr"]

def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr
    
def set_triangular_lr(optimizer, it, epoch_it, args, worse_epochs):
    cycle_length = epoch_it // args.cycles
    cycle = np.floor(1 + it / (2 * cycle_length))
    x = np.abs(it / cycle_length - 2 * cycle + 1)
    if args.lr_mode == 'triangular1':
        lr = args.min_lr + (args.max_lr - args.min_lr) * np.maximum(0, (1 - x))
    if args.lr_mode == 'triangular2':
        if (worse_epochs == 5 or worse_epochs == 10 or worse_epochs == 15) and it == 0:
            args.max_lr = args.max_lr / 2
        if args.max_lr < args.min_lr:
            args.max_lr = args.min_lr
        lr = args.min_lr + (args.max_lr - args.min_lr) * np.maximum(0, (1 - x))
    if args.lr_mode == 'exp_range':
        if (worse_epochs == 5 or worse_epochs == 10 or worse_epochs == 15) and it == 0:
            args.max_lr = args.max_lr / 2
        if args.max_lr < args.min_lr:
            args.max_lr = args.min_lr
        lr = args.min_lr + (args.max_lr - args.min_lr) * np.maximum(0, (1 - x))*args.lr_gamma**(it)
    set_lr(optimizer, lr)
