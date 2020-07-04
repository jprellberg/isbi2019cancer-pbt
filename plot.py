import random
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from pbt import deserialize_from_file


def get_ts_fitness(lineage):
    ts = [(x['timestamp_stop'] - x['timestamp_start']) / timedelta(hours=1) for x in lineage]
    ts = np.cumsum(ts)
    fitness = [x['fitness'] for x in lineage]
    return ts, fitness


def get_ancestors(lineage, start):
    lineage_dict = {x['id']: x for x in lineage}
    ancestors = [start]
    while 'parent' in ancestors[-1]:
        if ancestors[-1]['parent'] not in lineage_dict:
            break
        next = lineage_dict[ancestors[-1]['parent']]
        ancestors.append(next)
    return list(reversed(ancestors))


def get_age_fit(lineage):
    lineage_dict = {x['id']: x for x in lineage}
    for start in lineage:
        count = 0
        ind = start
        while 'parent' in ind:
            if ind['parent'] not in lineage_dict:
                yield count, start['fitness']
                break
            ind = lineage_dict[ind['parent']]
            count += 1


def plot_lineage(ax, lineage_file, **kwargs):
    lineage = deserialize_from_file(lineage_file)
    ts, fitness = get_ts_fitness(lineage)
    # Find oldest ancestor of the best individual
    best = lineage[np.argmax(fitness)]
    ancestors = get_ancestors(lineage, best)
    idx = lineage.index(ancestors[0])

    print(ancestors[0])

    # Mark ancestor with a cross
    res = ax.scatter([ts[idx]], [fitness[idx]], marker='x', s=100)
    color = res.get_facecolor()[0]

    # Plot fitness values of population
    ax.plot(ts, np.maximum.accumulate(fitness), c=color, **kwargs)
    ax.scatter(ts, fitness, c=color, s=5, alpha=0.5)


def plot_avg_fitness(ax, lineage_file, **kwargs):
    lineage = deserialize_from_file(lineage_file)
    ts, fitness = get_ts_fitness(lineage)
    # Plot moving average fitness of population
    window = 50
    avg_mask = np.ones(window) / window
    avg = np.convolve(fitness, avg_mask, 'same')
    ax.plot(ts, avg, **kwargs)


def plot_best_fitness_evolution(ax, lineage_file, **kwargs):
    lineage = deserialize_from_file(lineage_file)
    ts, fitness = get_ts_fitness(lineage)
    best = lineage[np.argmax(fitness) - 100]
    ancestors = get_ancestors(lineage, best)
    ts, fitness = get_ts_fitness(ancestors)
    ax.plot(ts, fitness, **kwargs)


fig, ax = plt.subplots(figsize=(9, 6))
plot_lineage(ax, 'results/lineage.pickle', label="Baseline")
ax.set_xlabel("Accumulated worker time in hours")
ax.set_ylabel("Fitness")
ax.legend(loc='lower right')

# fig, ax = plt.subplots(figsize=(9, 6))
# plot_avg_fitness(ax, 'results/lineage_simplepbt.pickle', label="SimplePBT")
# #plot_avg_fitness(ax, 'results/lineage_rankpartitionpbt.pickle', label="RankPartitionPBT")
# plot_avg_fitness(ax, 'results/lineage_inccomp_bugged.pickle', label="IncreasingCompetitivenessPBT (bugged)")
# plot_avg_fitness(ax, 'results/lineage.pickle', label="IncreasingCompetitivenessPBT")
# ax.set_xlabel("Accumulated worker time in hours")
# ax.set_ylabel("Average population fitness")
# ax.legend(loc='lower right')

# fig, ax = plt.subplots(figsize=(9, 6))
# plot_best_fitness_evolution(ax, 'results/lineage_simplepbt.pickle', label="SimplePBT")
# plot_best_fitness_evolution(ax, 'results/lineage.pickle', label="IncreasingCompetitivenessPBT")
# ax.legend(loc='lower right')

# age_fit = get_age_fit(deserialize_from_file('results/lineage.pickle'))
# age, fit = zip(*age_fit)
# fig, ax = plt.subplots(figsize=(9, 6))
# plt.scatter(age, fit, s=5, alpha=0.5)

plt.show()
