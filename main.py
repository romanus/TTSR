from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR
from loss.loss import get_loss_dict
from trainer import Trainer

import random
import os
import torch
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# this function guarantees reproductivity
# other packages also support seed options, you can add to this function
def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():

    seed_everything(42)

    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args).to(device)
    # count_parameters(_model)
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    else:
        current_epoch = 0
        # current_epoch = 30
        # t.load(model_path='./train/CUFED/TTSR/model/model_00030.pt')
        # t.evaluate(current_epoch=current_epoch)

        # t.loadLTE(model_path='./train/CUFED/TTSR3-5/model/model_00030.pt')

        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(current_epoch + 1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)

if __name__ == '__main__':
    main()
