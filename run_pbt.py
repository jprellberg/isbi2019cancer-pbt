import argparse
import subprocess
import sys

import zmq
from copy import deepcopy
from functools import partial

import numpy as np
import torch

from hp import FloatHP
from pbt import SimplePBTClient, SimplePBTServer
from utils import unique_string
from model import get_model_pretrained, get_model
from training import train_validate


def initfn():
    lmbda_min = 7
    lmbda_max = 3
    return {
        'model': {
            'sd': get_model_pretrained().state_dict(),
        },
        'hp': {
            'lr': FloatHP(-5, -1),
            'lmbda0': FloatHP(-lmbda_min, -lmbda_max),
            'lmbda1': FloatHP(-lmbda_min, -lmbda_max),
            'lmbda2': FloatHP(-lmbda_min, -lmbda_max),
            'lmbda3': FloatHP(-lmbda_min, -lmbda_max),
            'lmbda4': FloatHP(-lmbda_min, -lmbda_max),
            'lmbda5': FloatHP(-lmbda_min, -lmbda_max),
        },
    }


def mutatefn(individual):
    result = deepcopy(individual)
    for val in result['hp'].values():
        val.mutate_()
    return result


def recombinefn(population):
    off = np.random.choice(population)
    return off, (off,)


def stepfn(individual, args):
    network = get_model()
    network.load_state_dict(individual['model']['sd'])
    network = network.to(args.device)
    fitness = train_validate(network, individual['hp'], args)

    result = deepcopy(individual)
    result['model']['sd'] = network.state_dict()
    result['fitness'] = fitness
    return result


def start_server_std(args):
    print(args)
    server = SimplePBTServer(initfn, recombinefn, mutatefn, args.popsize, args.savedir, args.timelimit)
    server.listen_loop(args.addr)


def start_client(args):
    print(args)
    stepfn_partial = partial(stepfn, args=args)
    client = SimplePBTClient(stepfn_partial, args.max_client_steps)
    try:
        client.step_loop(args.addr)
        cmdargs = sys.argv
        print(cmdargs)
        subprocess.call(['sbatch', 'gpu.sh'] + cmdargs)
    except zmq.Again:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_client = subparsers.add_parser('client')
    parser_client.add_argument('--addr', default='tcp://localhost:5555')
    parser_client.add_argument('--steps', type=int, default=333, help="number of training steps per evolution step")
    parser_client.add_argument('--max-client-steps', type=int, default=10, help="number of evaluations before a client restarts itself")
    parser_client.add_argument('--batch-size', type=int, default=32)
    parser_client.add_argument('--dataroot', default='data')
    parser_client.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser_client.set_defaults(func=start_client)

    parser_server_std = subparsers.add_parser('server')
    parser_server_std.add_argument('--addr', default='tcp://*:5555')
    parser_server_std.add_argument('--timelimit', type=float, default=48, help="total worker time in hours until termination")
    parser_server_std.add_argument('--popsize', type=int, default=20)
    parser_server_std.add_argument('--savedir', default=f'results/pbt/{unique_string()}')
    parser_server_std.set_defaults(func=start_server_std)

    args = parser.parse_args()
    args.func(args)
