import jax.numpy as jnp

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote


class Kursawe(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=-5, xu=5, type_var=jnp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        l = []
        for i in range(2):
            l.append(-10 * jnp.exp(-0.2 * jnp.sqrt(jnp.square(x[:, i]) + jnp.square(x[:, i + 1]))))
        f1 = jnp.sum(jnp.column_stack(l), axis=1)

        f2 = jnp.sum(jnp.power(jnp.abs(x), 0.8) + 5 * jnp.sin(jnp.power(x, 3)), axis=1)

        out["F"] = jnp.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, **kwargs):
        return Remote.get_instance().load("pf", "kursawe.pf")



