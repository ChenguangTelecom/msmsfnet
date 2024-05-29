from train_one_epoch import train
from test_single_multi import single_scale_test, multi_scale_test
from utils import save_checkpoint
import os

def train_loop(train_loader, test_loader, net, opt, lr_schd, output_dir, args, device, log):
    train_epoch_losses=[]
    for epoch in range(args.start_epoch, args.max_epoch):
        if epoch==0:
            print('initial test')
            single_scale_test(test_loader, net, save_dir=os.path.join(output_dir, 'initial-test'), device=device)
            multi_scale_test(test_loader, net, save_dir=os.path.join(output_dir, 'initial-test'), scales=[0.5, 1.5], device=device)

        train_epoch_loss=train(train_loader, net, opt, lr_schd, epoch, save_dir=os.path.join(output_dir, 'epoch-{}-train'.format(epoch)), args=args, device=device)
        single_scale_test(test_loader, net, save_dir=os.path.join(output_dir, 'epoch-{}-test'.format(epoch)), device=device)
        multi_scale_test(test_loader, net, save_dir=os.path.join(output_dir, 'epoch-{}-test'.format(epoch)), scales=[0.5, 1.5], device=device)
        log.flush()
        save_checkpoint(state={'net': net.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, path=os.path.join(output_dir, 'epoch-{}-checkpoint.pth'.format(epoch)))
        train_epoch_losses.append(train_epoch_loss)

    return train_epoch_losses