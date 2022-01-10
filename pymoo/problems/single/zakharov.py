import jax.numpy as jnp

from pymoo.core.problem import Problem


class Zakharov(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-10, xu=10, type_var=jnp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        a = jnp.sum(0.5 * jnp.arange(1, self.n_var + 1) * x, axis=1)
        out["F"] = jnp.sum(jnp.square(x), axis=1) + jnp.square(a) + jnp.power(a, 4)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return jnp.full(self.n_var, 0)
