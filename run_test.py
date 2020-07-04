import argparse
import pickle

import torch
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import get_model
from training import evaluate


def test_model(dataroot, model_path, batch_size, device):
    trainset, validset, validset_subjects, class_weights = get_dataset(dataroot, folds_train=(0, 1, 2), folds_valid=(3,))
    class_weights = class_weights.to(device)
    valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=6, shuffle=False)

    model = get_model()
    sd = torch.load(model_path)
    # PBT saves the model as part of a dict that also contains other information about the individual
    if 'model' in sd and 'sd' in sd['model']:
        sd = sd['model']['sd']
    model.load_state_dict(sd)
    model.to(device)

    valid_loss, cm, auc, prec, rec, f1 = evaluate(model, valid_loader, class_weights, device)

    print(f"Results for model {model_path}")
    print(f"valid_loss={valid_loss:.4e}")
    print(f"auc={auc:.4f}")
    print(f"prec={prec:.4f}")
    print(f"rec={rec:.4f}")
    print(f"f1={f1:.4f}")
    print(f"cm=\n{cm}")

    return {
        'valid_loss': valid_loss,
        'cm': cm,
        'auc': auc,
        'prec': prec,
        'rec': rec,
        'f1': f1
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--model', required=True)
    parser.add_argument('--outfile')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    out = test_model(args.dataroot, args.model, args.batch_size, args.device)

    if args.outfile:
        with open(args.outfile, 'wb') as f:
            pickle.dump(out, f)
