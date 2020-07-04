import numpy as np


class Hyperparameter:
    def reset_(self):
        raise NotImplementedError

    def mutate_(self):
        raise NotImplementedError


class FloatHP(Hyperparameter):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.reset_()

    def reset_(self):
        self.value = np.random.uniform(self.lower, self.upper)

    def mutate_(self, p_noop=1/3):
        p_op = (1 - p_noop) / 2
        coeff = np.random.choice([0.8, 1.0, 1.2], p=[p_op, p_noop, p_op])
        self.value = np.clip(coeff * self.value, self.lower, self.upper)

    def __repr__(self):
        return f"{self.value:.4f}"


class IntHP(Hyperparameter):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.reset_()

    def reset_(self):
        self.value = np.random.randint(self.lower, self.upper + 1)

    def mutate_(self, p_noop=1/3):
        p_op = (1 - p_noop) / 2
        coeff = np.random.choice([0.8, 1.0, 1.2], p=[p_op, p_noop, p_op])
        self.value = np.clip(coeff * self.value, self.lower, self.upper)
        self.value = int(round(self.value))

    def __repr__(self):
        return str(self.value)
