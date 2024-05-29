python main_msmsfnet.py --max_epoch 30 --start_epoch 0 --train_dataset_dir /media/user/data/datasets/bsds_voc/data --test_dataset_dir ./data --trainlist bsds_pascal_train_pair.lst --train_batch_size 1 --itersize 5 --test_batch_size 1 --lr 1e-4 --lr_stepsize 20 --lr_gamma 0.1 --weight_decay 1e-12 --print_freq 1000 --output output-msmsfnet-test --test --checkpoint ./output-msmsfnet-wd12/epoch-20-checkpoint.pth


