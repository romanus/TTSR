from torch.utils.data import DataLoader
from importlib import import_module
from torchvision import transforms
from dataset.cufed import ToTensor

def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())

    if (args.dataset == 'CUFED'):
        data_train = getattr(m, 'TrainSet')(args)
        data_train_no_shuffle = getattr(m, 'TrainSet')(args, transform=transforms.Compose([ToTensor()]))
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_train_no_shuffle = DataLoader(data_train_no_shuffle, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'train_no_shuffle': dataloader_train_no_shuffle, 'test': dataloader_test}
    elif (args.dataset == 'ffhq'):
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_train_no_shuffle = DataLoader(data_train, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader_test = {}
        data_test = getattr(m, 'TestSet')(args=args)
        dataloader_test['1'] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'train_no_shuffle': dataloader_train_no_shuffle, 'test': dataloader_test}
    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader