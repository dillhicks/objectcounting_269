from FCRNA import FCRN_A
from data_loader import H5Dataset
from matplotlib import pyplot as plt
from looper import Looper
import numpy as np
import os
import torch



def train(device, dataloader, 
          learning_rate = 1e-2,
          epochs = 150,
          filters = 64,
          convolutions = 2,
          plot = False):
          
    """Train chosen model on cell data dataset."""

    # initialize a model based on chosen network_architecture
    network = FCRN_A(input_filters=3,
                            filters=filters,
                            N=convolutions).to(device)
    network = torch.nn.DataParallel(network)

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
                                
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        plt.ion()
        fig, plots = plt.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss, optimizer,
                          dataloader['train'], len(dataloader['train']), plots[0])
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader['valid'], len(dataloader['valid']), plots[1],
                          validation=True)

    # current best results (lowest mean absolute error on validation set)
    current_best = np.infty

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            torch.save(network.state_dict(),
                       f'cell_fcrna.pth')

            print(f"\nNew best result: {result}")

        print("\n", "-"*80, "\n", sep='')

    print(f"[Training done] Best result: {current_best}")



def get_loader(path,
                horizontal_flip = 0,
                vertical_flip = 0,
                batch_size = 8):

    dataset = {}     # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ['train', 'valid']:
        data_path = os.path.join(path, f"{mode}.h5")
        # expected HDF5 files in dataset_name/(train | valid).h5
        # turn on flips only for training dataset
        dataset[mode] = H5Dataset(data_path,
                                  horizontal_flip if mode == 'train' else 0,
                                  vertical_flip if mode == 'train' else 0)
        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size= batch_size)

    return dataloader