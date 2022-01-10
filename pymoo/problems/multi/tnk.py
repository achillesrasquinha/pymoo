import jax.numpy as jnp

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote


class TNK(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, type_var=jnp.double)
        self.xl = jnp.array([0, 1e-30])
        self.xu = jnp.array([jnp.pi, jnp.pi])

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        f2 = x[:, 1]
        g1 = -(jnp.square(x[:, 0]) + jnp.square(x[:, 1]) - 1.0 - 0.1 * jnp.cos(16.0 * jnp.arctan(x[:, 0] / x[:, 1])))
        g2 = 2 * (jnp.square(x[:, 0] - 0.5) + jnp.square(x[:, 1] - 0.5)) - 1

        out["F"] = jnp.column_stack([f1, f2])
        out["G"] = jnp.column_stack([g1, g2])

    def _calc_pareto_front(self, *args, **kwargs):
        return Remote.get_instance().load("pf", "tnk.pf")
