import jax.numpy as jnp
from jax import random, lax, jit, tree_util

class MyEnv:
   
    def __init__(self):
        self.random_limit = 0.05
        self.dynamic_place_holder = 0
   
    # @jit
    # def _get_obsv(self, state):
    #     return state
   
    # @jit
    # def _maybe_reset(self, env_state, done):
    #     key = env_state[1]
    #     return lax.cond(done, self._reset, lambda key: env_state, key)
   
    # @jit
    # def _reset(self, key):
    @jit
    def reset(self, key):
        new_state = 0
        new_key = key
        # new_state = random.uniform(key, minval=-self.random_limit, 
        #                                 maxval=self.random_limit, shape=(4,))
        # new_key = random.split(key)[0]
        return new_state, new_key

    # @jit
    # def step(self, env_state, action):
    #     state, key = env_state
    #     new_state = state + action
      
    #     reward, done, info = 1., False, None
      
    #     env_state = new_state, key
    #     env_state = self._maybe_reset(env_state, done)
    #     new_state = env_state[0]
    #     return env_state, self._get_obsv(new_state), reward, done, info
      
    # @jit
    # def reset(self, key):
    #     env_state = self._reset(key)
    #     new_state = env_state[0]
    #     return env_state, self._get_obsv(new_state)
    
    def _tree_flatten(self):
        children = (self.dynamic_place_holder,) #dynamic
        aux_data = {'random_limit': self.random_limit} #static
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
tree_util.register_pytree_node(MyEnv,
                               MyEnv._tree_flatten,
                               MyEnv._tree_unflatten)

key = random.PRNGKey(0)
env = MyEnv()
env_state, inital_obsv = env.reset(key)
# action = 1
# env_state, obsv, reward, done, info = env.step(env_state, 1)
