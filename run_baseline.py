import argparse
import os

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import get_dataset
from model import get_model_pretrained
from training import train, evaluate
from utils import pickle_dump, unique_string


def schedule(epoch):
    if epoch < 2:
        ub = 1
    elif epoch < 4:
        ub = 0.1
    else:
        ub = 0.01
    return ub


def train_validate(args):
    trainset, validset, validset_subjects, class_weights = get_dataset(args.dataroot)
    class_weights = class_weights.to(args.device)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=6, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=6, shuffle=False)

    model = get_model_pretrained().to(args.device)

    params01 = model.get_params('layer0') + model.get_params('layer1')
    params234 = model.get_params('layer2') + model.get_params('layer3') + model.get_params('layer4')
    paramscls = model.get_params('cls')

    opt = torch.optim.Adam([
        {'params': params01, 'lr': 1e-6},
        {'params': params234, 'lr': 1e-4},
        {'params': paramscls, 'lr': 1e-2},
    ])
    scheduler = LambdaLR(opt, lr_lambda=[lambda e: schedule(e),
                                         lambda e: schedule(e),
                                         lambda e: schedule(e)])

    steps = len(train_loader)
    best_f1 = 0.
    history = []
    for e in trange(args.epochs, desc='Epoch'):
        scheduler.step(e)
        train(model, opt, steps, train_loader, class_weights, args.device)
        valid_loss, cm, auc, prec, rec, f1 = evaluate(model, valid_loader, class_weights, args.device)
        history.append((valid_loss, cm, auc, prec, rec, f1))
        pickle_dump(history, f'{args.out}/history.pickle')

        if f1 > best_f1:
            print(f"\nNew best model with F1={f1:.6f}")
            torch.save(model.state_dict(), f'{args.out}/model.pt')
            best_f1 = f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=f'results/baseline/{unique_string()}')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_validate(args)
