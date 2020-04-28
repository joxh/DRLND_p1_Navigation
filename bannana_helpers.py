from contextlib import contextmanager
from unityagents import UnityEnvironment


def reset_and_get_first_state(env, train_mode=True):
    env_info = env[0].reset(train_mode=train_mode)[env[1]]
    state = env_info.vector_observations[0]
    return state


def get_next_state_reward_done(env, action):
    env_info = env[0].step(action)[env[1]]
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]

    return next_state, reward, done


@contextmanager
def get_environment(*args, **kwds):
    env = UnityEnvironment(*args, **kwds)
    brain_name = env.brain_names[0]
    try:
        yield env, brain_name
    finally:
        env.close()
