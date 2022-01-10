import numpy as np
import jax.numpy as jnp

from pymoo.core.problem import Problem


class Knapsack(Problem):
    def __init__(self,
                 n_items,  # number of items that can be picked up
                 W,  # weights for each item
                 P,  # profit of each item
                 C,  # maximum capacity
                 ):
        super().__init__(n_var=n_items, n_obj=1, n_constr=1, xl=0, xu=1, type_var=bool)

        self.W = W
        self.P = P
        self.C = C

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -jnp.sum(self.P * x, axis=1)
        out["G"] = (jnp.sum(self.W * x, axis=1) - self.C)


class MultiObjectiveKnapsack(Knapsack):
    def __init__(self, *args):
        super().__init__(*args)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = - jnp.sum(self.P * x, axis=1)
        f2 = jnp.sum(x, axis=1)

        out["F"] = jnp.column_stack([f1, f2])
        out["G"] = (jnp.sum(self.W * x, axis=1) - self.C)


def create_random_knapsack_problem(n_items, seed=1, variant="single"):
    jnp.random.seed(seed)
    P = jnp.random.randint(1, 100, size=n_items)
    W = jnp.random.randint(1, 100, size=n_items)
    C = int(jnp.sum(W) / 10)

    if variant == "single":
        problem = Knapsack(n_items, W, P, C)
    else:
        problem = MultiObjectiveKnapsack(n_items, W, P, C)

    return problem
