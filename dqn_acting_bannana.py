import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from contextlib import contextmanager


from bannana_helpers import (
    reset_and_get_first_state,
    get_next_state_reward_done,
    get_environment,
)

from visualization_banana import plot_score_cumulative_distribution


def main(
    file_name="/Users/joshuaschoenfield/Downloads/Banana.app",
    weights_file="checkpoint_bannana_2_LONG_SAFE.pth",
):
    with get_environment(file_name=file_name) as env:
        from dqn_agent import Agent

        agent = Agent(state_size=37, action_size=4, seed=0)

        agent.qnetwork_local.load_state_dict(torch.load(weights_file))
        scores = []
        num_iterations = 100
        for i in range(num_iterations):
            state = reset_and_get_first_state(env, train_mode=True)
            score = 0
            for j in range(2000):
                action = agent.act(state, eps=0)
                # env.render()
                state, reward, done = get_next_state_reward_done(env, action)
                score += reward
                if done:
                    break
            scores.append(score)
            # print(f"Score: {score}")
        # print(f"Average Score: {np.mean(scores)}")
        ax = plot_score_cumulative_distribution(scores)
        ax.figure.savefig("Media/validation_scores_cumulative.png")
        # plt.show()
        np.savetxt("validation_scores.txt", scores)
        return scores


if __name__ == "__main__":
    main()
