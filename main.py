from models import CNNDIQAnet
from Performance import DIQAPerformance
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from DIQADataset import DIQADataset
import numpy as np
import yaml
from pathlib import Path
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanSquaredError
from DataInfoLoader import DataInfoLoader
import math
from ignite.contrib.handlers import ProgressBar


def ensure_dir(path):
    p = Path(path)
    if not p.exists():
        p.mkdir()


def custom_loss_fn(y_pred, y):
    loss = 0
    loss += F.mse_loss(y_pred[:, 0], y[:, 0])
    loss += F.mse_loss(y_pred[:, 1], y[:, 1])
    return loss


def loss_fn(y_pred, y):
    y = y.view_as(y_pred)  # Reshape target to match prediction
    return custom_loss_fn(y_pred, y)  # Use MSE loss


def get_data_loaders(dataset_name, config, train_batch_size):
    datainfo = DataInfoLoader(dataset_name, config)
    img_num = datainfo.img_num
    index = np.arange(img_num)
    np.random.shuffle(index)

    train_index = index[0:math.floor(img_num * 0.6)]
    val_index = index[math.floor(img_num * 0.6):math.floor(img_num * 0.8)]
    test_index = index[math.floor(img_num * 0.8):]

    train_dataset = DIQADataset(dataset_name, config, train_index, status='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=4)

    val_dataset = DIQADataset(dataset_name, config, val_index, status='val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = DIQADataset(dataset_name, config, test_index, status='test')
        test_loader = torch.utils.data.DataLoader(test_dataset)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class Solver:
    def __init__(self):
        self.model = CNNDIQAnet()

    def run(self, dataset_name, train_batch_size, epochs, lr, weight_decay, model_name, config, trained_model_file,
            save_result_file, disable_gpu=False):
        if config['test_ratio']:
            train_loader, val_loader, test_loader = get_data_loaders(dataset_name, config, train_batch_size)
        else:
            train_loader, val_loader = get_data_loaders(dataset_name, config, train_batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() and not disable_gpu else "cpu")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")

        self.model = self.model.to(device)

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        global best_criterion
        best_criterion = float('inf')  # Initialize with infinity for MSE
        trainer = create_supervised_trainer(self.model, optimizer, loss_fn, device=device)

        evaluator = create_supervised_evaluator(self.model, metrics={'MSE': DIQAPerformance()}, device=device)

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            pass

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            mse = metrics['MSE']
            print(f"Validation Results - Epoch: {engine.state.epoch} MSE: {mse:.4f}")
            global best_criterion
            global best_epoch
            if mse < best_criterion:
                best_criterion = mse
                best_epoch = engine.state.epoch
                torch.save(self.model.state_dict(), trained_model_file)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_testing_results(engine):
            if config["test_ratio"] > 0 and config['test_during_training']:
                evaluator.run(test_loader)
                metrics = evaluator.state.metrics
                mse = metrics['MSE']
                print(f"Testing Results - Epoch: {engine.state.epoch} MSE: {mse:.4f}")

        @trainer.on(Events.COMPLETED)
        def final_testing_results(engine):
            if config["test_ratio"] > 0:
                self.model.load_state_dict(torch.load(trained_model_file))
                evaluator.run(test_loader)
                metrics = evaluator.state.metrics
                mse = metrics['MSE']
                global best_epoch
                print(f"Final Test Results - Epoch: {best_epoch} MSE: {mse:.4f}")
                np.save(save_result_file, mse)

        trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNDIQA')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='optical_problems_contrast_change', type=str,
                        help='exp id (default: bt)')
    parser.add_argument('--dataset_name', default='SOC', type=str,
                        help='dataset name (default: SOC)')
    parser.add_argument('--model', default='CNNDIQA', type=str,
                        help='model name (default: CNNDIQA)')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    print('exp id: ' + args.exp_id)
    print('model: ' + args.model)
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-EXP{}-lr={}.pth'.format(args.model, args.dataset_name, args.exp_id, args.lr)
    ensure_dir('results')
    save_result_file = 'results/{}-{}-EXP{}-lr={}.npy'.format(args.model, args.dataset_name, args.exp_id, args.lr)

    dataset_name = 'SOC'

    solver = Solver()
    solver.run(dataset_name, args.batch_size, args.epochs, args.lr, args.weight_decay, args.model, config,
               trained_model_file, save_result_file, args.disable_gpu)
