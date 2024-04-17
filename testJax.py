import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# https://stackoverflow.com/questions/68327863/importing-jax-fails-on-mac-with-m1-chip

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

size = 300
x = random.normal(key, (size, size), dtype=jnp.float32)
jnp.dot(x, x.T).block_until_ready()
# %timeit jnp.dot(x, x.T).block_until_ready()  # runs on the GPU

import numpy as np
x = np.random.normal(size=(size, size)).astype(np.float32)
# %timeit jnp.dot(x, x.T).block_until_ready()

from jax import device_put

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
# %timeit jnp.dot(x, x.T).block_until_ready()

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = random.normal(key, (1000000,))
# %timeit selu(x).block_until_ready()

selu_jit = jit(selu)
print(selu_jit(x))
# %timeit selu_jit(x).block_until_ready()

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))

def first_finite_differences(f, x):
  eps = 1e-3
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])


print(first_finite_differences(sum_logistic, x_small))

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

from jax import jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)


def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
naively_batched_apply_matrix(batched_x)
# %timeit naively_batched_apply_matrix(batched_x).block_until_ready()

@jit
def batched_apply_matrix(v_batched):
  return jnp.dot(v_batched, mat.T)

print('Manually batched')
batched_apply_matrix(batched_x)
# %timeit batched_apply_matrix(batched_x).block_until_ready()

@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

print('Auto-vectorized with vmap')
vmap_batched_apply_matrix(batched_x)
# %timeit vmap_batched_apply_matrix(batched_x).block_until_ready()