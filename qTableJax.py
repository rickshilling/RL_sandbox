import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import SkeletonEnv

observation_space_n = 4
action_space_n = 2

Q = jnp.zeros((observation_space_n, action_space_n))

key = random.PRNGKey(0)
env = SkeletonEnv.SkeletonEnv()
env_state, inital_obsv = env.reset(key)

# @jit
# def Q_table(lr=.8, y=.95, num_episodes=2000):
#     jax.lax.fori_loop(0, num_episodes,

@jit
def inner_body(i, val):
    return i + val

r = jax.lax.fori_loop(0, 10, inner_body, 0)
print(r)

#-----------------------------------------------
@jit
def inner_body2(i, c):
    return (i + c[0], i -c[1])

r = jax.lax.fori_loop(0, 10, inner_body2, (0,0))
print(r)

#-----------------------------------------------

# @jit
# def inner_body2(j, Q, s, env):
#     j += 1
#     a = jnp.argmax(Q[s,:])
#     # env_state, obsv = env.reset(key)
#     return (Q,s,env)

# s = 0
# (Qf,sf,envf) = jax.lax.fori_loop(0, 10, inner_body2, (Q, s, env))