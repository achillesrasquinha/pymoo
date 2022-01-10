import jax.numpy as jnp

from pymoo.core.problem import Problem


class Ackley(Problem):

    def __init__(self, n_var=2, a=20, b=1/5, c=2 * jnp.pi):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-32.768, xu=+32.768, type_var=jnp.double)
        self.a = a
        self.b = b
        self.c = c

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = -1. * self.a * jnp.exp(-1. * self.b * jnp.sqrt((1. / self.n_var) * jnp.sum(x * x, axis=1)))
        part2 = -1. * jnp.exp((1. / self.n_var) * jnp.sum(jnp.cos(self.c * x), axis=1))
        out["F"] = part1 + part2 + self.a + jnp.exp(1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return jnp.full(self.n_var, 0)
