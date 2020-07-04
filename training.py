import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from model import get_model_pretrained
from utils import IncrementalAverage, loop_iter


def evaluate(model, valid_loader, class_weights, device):
    model.eval()

    all_labels = []
    all_preds = []
    loss_avg = IncrementalAverage()
    for img, label in tqdm(valid_loader):
        img, label = img.to(device), label.to(device)
        bs, nrot, c, h, w = img.size()
        with torch.no_grad():
            pred = model(img.view(-1, c, h, w))
            pred = pred.view(bs, nrot).mean(1)
            loss = lossfn(pred, label.to(pred.dtype), class_weights)
            all_labels.append(label.cpu())
            all_preds.append(pred.cpu())
            loss_avg.update(loss.item())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_preds_binary = all_preds > 0

    cm = confusion_matrix(all_labels, all_preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds_binary, average='weighted')
    return loss_avg.value, cm, auc, prec, rec, f1


def train(model, opt, steps, train_loader, class_weights, ptp, lmbdas, device):
    model.train()
    for i, (img, label) in tqdm(zip(range(steps), loop_iter(train_loader)), total=steps):
        img, label = img.to(device), label.to(device)
        pred = model(img)
        pred = pred.view(-1)
        div_loss = divergence_loss(model, ptp, lmbdas, device)
        loss = lossfn(pred, label.to(pred.dtype), class_weights) + div_loss

        opt.zero_grad()
        loss.backward()
        opt.step()


def divergence_loss(model, ptp, lmbdas, device):
    vecs = layer_vectors(model, device)
    loss = []
    for p1, p2, lmbda in zip(ptp, vecs, lmbdas):
        loss.append(lmbda * (p1 - p2).abs().mean())
    loss = torch.stack(loss).sum()
    return loss


def layer_vectors(model, device, copy=False):
    return [nn.utils.parameters_to_vector(model.get_params(k)).to(device, copy=copy)
            for k in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'cls']]


def lossfn(prediction, target, class_weights):
    pos_weight = (class_weights[0] / class_weights[1]).expand(len(target))
    return F.binary_cross_entropy_with_logits(prediction, target, pos_weight=pos_weight)


def train_validate(model, hp, args):
    pm = get_model_pretrained()
    for p in pm.parameters():
        p.requires_grad = False
    ptp = layer_vectors(pm, args.device, True)
    del pm

    trainset, validset, validset_subjects, class_weights = get_dataset(args.dataroot)
    class_weights = class_weights.to(args.device)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=6, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=6, shuffle=False)

    lmbdas = sorted([10 ** hp[key].value for key in ['lmbda0', 'lmbda1', 'lmbda2', 'lmbda3', 'lmbda4', 'lmbda5']], reverse=True)

    opt = torch.optim.Adam([
        {'params': model.get_params('layer0'), 'lr': lmbdas[0]},
        {'params': model.get_params('layer1'), 'lr': lmbdas[1]},
        {'params': model.get_params('layer2'), 'lr': lmbdas[2]},
        {'params': model.get_params('layer3'), 'lr': lmbdas[3]},
        {'params': model.get_params('layer4'), 'lr': lmbdas[4]},
        {'params': model.get_params('cls'), 'lr': lmbdas[5]},
    ])

    train(model, opt, args.steps, train_loader, class_weights, ptp, lmbdas, args.device)
    valid_loss, cm, auc, prec, rec, f1 = evaluate(model, valid_loader, class_weights, args.device)

    return f1
