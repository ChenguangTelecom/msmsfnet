from torch.utils.data import DataLoader
from datasets_train import TrainDataset
from datasets_test import TestDataset

def prepare_datasets(args):
    trainset=TrainDataset(dataset_dir=args.train_dataset_dir, trainlist=args.trainlist, mean_pixel_value= args.mean_pixel_value)
    testset=TestDataset(dataset_dir=args.test_dataset_dir, mean_pixel_value=args.mean_pixel_value)
    trainloader=DataLoader(trainset, batch_size=args.train_batch_size, num_workers=4, drop_last=True, shuffle=True)
    testloader=DataLoader(testset, batch_size=args.test_batch_size, num_workers=4, drop_last=False, shuffle=False)
    return trainloader, testloader