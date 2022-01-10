import jax.numpy as jnp

from pymoo.core.decomposition import Decomposition


class Tchebicheff(Decomposition):

    def _do(self, F, weights, **kwargs):
        v = jnp.abs(F - self.utopian_point) * weights
        tchebi = v.max(axis=1)
        return tchebi
