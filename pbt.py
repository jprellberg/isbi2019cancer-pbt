import logging
import os
import pprint
import re
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from io import BytesIO
from operator import itemgetter
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import zmq

from utils import pickle_dump


def serialize_to_file(obj, file):
    torch.save(obj, file)


def serialize(obj):
    f = BytesIO()
    torch.save(obj, f)
    byteval = f.getvalue()
    f.close()
    return byteval


def deserialize_from_file(file, map_location=None):
    return torch.load(file, map_location)


def deserialize(byteval, map_location=None):
    f = BytesIO(byteval)
    obj = torch.load(f, map_location)
    f.close()
    return obj


def individual2str(individual):
    return pprint.pformat(lightweight_individual(individual, copy=False))


def lightweight_individual(individual, copy=True):
    copyfn = deepcopy if copy else lambda x: x
    return {key: copyfn(individual[key]) for key in individual if key != 'model'}


def initfn_wrapper(initfn):
    def _wrapped():
        x = initfn()
        if not (isinstance(x, dict) and 'model' in x and 'hp' in x):
            raise ValueError("initfn must return dictionary with keys 'model' and 'hp'")
        x['id'] = uuid4().hex
        x['fitness'] = -np.inf
        x['parent'] = None
        x['age'] = 0
        x['timestamp_start'] = None
        x['timestamp_stop'] = None
        return x
    return _wrapped


def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')

        # Create console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


class PBTClient:
    def __init__(self, max_steps):
        self.logger = get_logger()
        self.max_steps = max_steps

    def step_loop(self, server_addr):
        self.logger.info("Connecting to server")
        self._unsafe_step_loop(server_addr)
        self.logger.info("Shutdown")

    def _unsafe_step_loop(self, server_addr, timeout=30000):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        # 10s timeouts
        socket.setsockopt(zmq.RCVTIMEO, timeout)
        socket.setsockopt(zmq.SNDTIMEO, timeout)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(server_addr)

        for step_count in range(self.max_steps):
            self.logger.info(f"Step loop iteration {step_count}")
            socket.send_multipart([b'get', b''])
            individual = deserialize(socket.recv())
            individual = self.step(individual)
            if individual is not None:
                socket.send_multipart([b'put', serialize(individual)])
                # Call recv because we need to for a REQ socket even though we don't return any data after put
                socket.recv()

    def step(self, individual):
        raise NotImplementedError


class SimplePBTClient(PBTClient):
    def __init__(self, stepfn, max_steps):
        """
        :param stepfn: Function that trains and individual for some steps and evaluates it. The fitness should
                       be saved as 'fitness' in the Individual. Signature: Individual -> Individual.
        """
        super().__init__(max_steps)
        self.stepfn = stepfn

    def step(self, individual):
        try:
            return self.stepfn(individual)
        except Exception:
            ex_msg = traceback.format_exc()
            self.logger.warning(f"Exception during step:\n{ex_msg}")
            return None


class PBTServer:
    def __init__(self):
        self.logger = get_logger()

    def listen_loop(self, bind_addr):
        self.restore_state()

        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.bind(bind_addr)

        self.logger.info("Server is listening")
        while not self.should_terminate():
            try:
                addr, empty, request, payload = socket.recv_multipart()
            except ValueError:
                ex_msg = traceback.format_exc()
                self.logger.warning(f"Malformed request:\n{ex_msg}")
                continue

            if request == b'get':
                individual = self.get()
                socket.send_multipart([addr, empty, serialize(individual)])
            elif request == b'put':
                payload = deserialize(payload, map_location='cpu')
                self.put(payload)
                socket.send_multipart([addr, empty, b''])
            else:
                self.logger.warning("Invalid request:\n" + str(request))

            self.save_state()

        self.logger.info("Server shut down")

    def save_state(self):
        raise NotImplementedError

    def restore_state(self):
        raise NotImplementedError

    def should_terminate(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def put(self, individual):
        raise NotImplementedError


class SimplePBTServer(PBTServer):
    def __init__(self, initfn, recombinefn, mutatefn, popsize, savedir, max_hours=0):
        """
        :param initfn: Function that creates a random individual. Signature: None -> Individual.
        :param recombinefn: Function that recombines members from the population to create an offspring Individual. It
                            should also return the parent(s) used for recombination.
                            Signature: List[Individual] -> (Individual, List[Individual]).
        :param mutatefn: Function that mutates an individual. Signature: Individual -> Individual.
        :param popsize: population size
        :param savedir: save directory for the server
        :param max_hours: termination condition in terms of total worker hours (default=0 means unlimited)
        """
        super().__init__()
        self.initfn = initfn_wrapper(initfn)
        self.recombinefn = recombinefn
        self.mutatefn = mutatefn
        self.total_worker_hours = 0
        self.max_worker_hours = max_hours
        self.state_file = os.path.join(savedir, 'serverstate.pickle')

        # Create savedirs if they do not exist
        popdir = os.path.join(savedir, 'population')
        leaderdir = os.path.join(savedir, 'leaderboard')
        os.makedirs(popdir, exist_ok=True)
        os.makedirs(leaderdir, exist_ok=True)

        # Fill population to requested size with random individuals
        for _ in range(len(os.listdir(popdir)), popsize):
            individual = self.initfn()
            individual['parents'] = ('initial-seed',)
            serialize_to_file(individual, os.path.join(popdir, individual['id']))

        # Create leaderboard which saves top performing individuals and their models
        self.leaderboard = Leaderboard(leaderdir, num_spots=10)
        # Create population (members are loaded from popdir)
        self.population = Population(popdir, popsize)

    def _state(self):
        return {'total_worker_hours': self.total_worker_hours}

    def save_state(self):
        serialize_to_file(self._state(), self.state_file)

    def restore_state(self):
        if os.path.exists(self.state_file):
            state = deserialize_from_file(self.state_file)
            for k, v in state.items():
                setattr(self, k, v)

    def should_terminate(self):
        if self.max_worker_hours == 0:
            return False
        else:
            return self.total_worker_hours > self.max_worker_hours

    def get(self):
        self.logger.info(f"Creating offspring")

        offspring, parents = self.recombinefn(self.population.members)
        offspring = self.mutatefn(offspring)
        offspring['id'] = uuid4().hex
        offspring['parents'] = [x['id'] for x in parents]
        offspring['age'] = max(x['age'] for x in parents) + 1
        offspring['timestamp_start'] = datetime.now()
        return offspring

    def put(self, individual):
        self._put_indexation(individual)
        self._put_selection(individual)

    def _put_indexation(self, individual):
        self.logger.info(f"Received result:\n{individual2str(individual)}")

        # Timestamp individual and calculate time taken for termination condition
        individual['timestamp_stop'] = datetime.now()
        hours = (individual['timestamp_stop'] - individual['timestamp_start']) / timedelta(hours=1)
        self.total_worker_hours += hours
        self.logger.info(f"Timelimit: {self.total_worker_hours:.2f}h/{self.max_worker_hours:.2f}h")

        # Record in population lineage
        self.population.record_lineage(individual)

        # Submit to leaderboard
        self.leaderboard.submit(individual)
        self.leaderboard.log_fitness(self.logger)

    def _put_selection(self, individual):
        # Binary-tournament selection but replace all -inf fitness members first
        neginf_individuals = [x for x in self.population.members if np.isneginf(x['fitness'])]
        if neginf_individuals:
            rival = np.random.choice(neginf_individuals)
        else:
            rival = np.random.choice(self.population.members)
        if rival['fitness'] < individual['fitness']:
            self.population.replace(rival, individual)
            self.logger.info(f"Binary tournament won (kept result)")
        else:
            self.logger.info(f"Binary tournament lost (discarded result)")
        self.population.log_fitness(self.logger)


class Leaderboard:
    def __init__(self, savedir, num_spots):
        self.savedir = savedir
        self.num_spots = num_spots
        self.members = []

        # Load members from savedir
        self.members_file = os.path.join(self.savedir, 'members.pickle')
        if os.path.exists(self.members_file):
            self.members = deserialize_from_file(self.members_file)

    def submit(self, individual):
        # Add newest member to leaderboard if either there is still empty place on the leaderboard or the individual
        # is better than the worst one
        if len(self.members) < self.num_spots or individual['fitness'] > self.members[-1]['fitness']:
            serialize_to_file(individual, os.path.join(self.savedir, individual['id']))
            # Only save light-weight version without model in leaderboard because the model is already
            # serialized in case we want to retrieve it later
            self.members.append(lightweight_individual(individual))
            self.members = sorted(self.members, key=itemgetter('fitness'), reverse=True)

        # Remove last member of leaderboard if it is now overfilled
        if len(self.members) > self.num_spots:
            removed = self.members.pop()
            os.remove(os.path.join(self.savedir, removed['id']))

        serialize_to_file(self.members, self.members_file)

    def log_fitness(self, logger):
        fvals = [x['fitness'] for x in self.members]
        leader_stats = "Leaderboard statistics:"
        for x in self.members:
            leader_stats += f"\nfit={x['fitness']:.4f} age={x['age']:2d} id={x['id'][:8]}..."
        leader_stats += f"\nmean_fit={np.mean(fvals):.4f}"
        logger.info(leader_stats)


class Population:
    def __init__(self, savedir, popsize):
        self.savedir = savedir
        self.popsize = popsize
        self.members = []
        self.lineage = []

        # Load population from savedir
        for f in Path(self.savedir).iterdir():
            if re.match(r"^[a-f0-9]{32}$", f.name):
                obj = deserialize_from_file(f.absolute())
                self.members.append(obj)

        # Load lineage from savedir
        self.lineage_file = os.path.join(self.savedir, 'lineage.pickle')
        if os.path.exists(self.lineage_file):
            self.lineage = deserialize_from_file(self.lineage_file)

    def remove(self, individual):
        # Remove individual from pop and delete its model checkpoint
        self.members = [x for x in self.members if x['id'] != individual['id']]
        os.remove(os.path.join(self.savedir, individual['id']))

    def add(self, individual):
        # Add new individual to population
        self.members.append(individual)
        self.members = sorted(self.members, key=itemgetter('fitness'), reverse=True)
        serialize_to_file(individual, os.path.join(self.savedir, individual['id']))

    def replace(self, individual, replacement):
        self.remove(individual)
        self.add(replacement)

    def record_lineage(self, individual):
        # Backup a light-weight copy for lineage
        self.lineage.append(lightweight_individual(individual))
        serialize_to_file(self.lineage, self.lineage_file)

    def log_fitness(self, logger):
        fvals = [x['fitness'] for x in self.members]
        pop_stats = "Population statistics:"
        for x in self.members:
            pop_stats += f"\nfit={x['fitness']:.4f} age={x['age']:2d}"
        pop_stats += f"\nmean_fit={np.mean(fvals):.4f}"
        logger.info(pop_stats)
