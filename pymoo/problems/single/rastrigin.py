import jax.numpy as jnp

from pymoo.core.problem import Problem


class Rastrigin(Problem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-5, xu=5, type_var=jnp.double)
        self.A = A

    def _evaluate(self, x, out, *args, **kwargs):
        z = jnp.power(x, 2) - self.A * jnp.cos(2 * jnp.pi * x)
        out["F"] = self.A * self.n_var + jnp.sum(z, axis=1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return jnp.full(self.n_var, 0)
