import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from contextlib import contextmanager
from banana_helpers import (
    reset_and_get_first_state,
    get_next_state_reward_done,
    get_environment,
)


def dqn(
    *,
    env,
    agent,
    n_episodes=6000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    with_plotting=True,
):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    running_average = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    fig = None
    for i_episode in range(1, n_episodes + 1):
        state = reset_and_get_first_state(env)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = get_next_state_reward_done(env, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        running_average.append(np.mean(scores_window))
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_window)
            ),
            end="",
        )
        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
        if with_plotting:
            if i_episode % 10 == 0:
                if fig is None:
                    plt.ion()
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    line = ax.scatter(np.arange(len(scores)), scores, alpha=0.5)
                    if i_episode > 20:
                        ax.plot(
                            np.arange(20, len(running_average)),
                            running_average[20:],
                            color="red",
                            label="Trailing Average",
                        )
                    ax.axhline(0, color="black")
                    ax.axhline(13, color="orange", label="target")
                    plt.ylabel("Score")
                    plt.xlabel("Episode #")
                    ax.legend()
                    plt.draw()
                else:
                    ax.clear()
                    ax.scatter(np.arange(len(scores)), scores, alpha=0.5)
                    if i_episode > 20:
                        ax.plot(
                            np.arange(len(running_average))[20:],
                            running_average[20:],
                            color="red",
                            label="Trailing Average",
                        )
                    ax.axhline(0, color="black")
                    ax.axhline(13, color="orange", label="target")
                    ax.legend()
                    plt.draw()
                # fig.canvas.draw()
                # fig.canvas.flush_events()
                plt.show()
                plt.pause(0.001)

        if (np.mean(scores_window) >= 14.0) | (
            (np.mean(scores_window) >= 13.0) & (i_episode == n_episodes)
        ):
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_window)
                )
            )
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint_banana_2.pth")
            np.savetxt("training_scores_checkpoint.txt", scores)
            if with_plotting:
                ax.clear()
                ax.scatter(np.arange(len(scores)), scores, alpha=0.5)
                if i_episode > 20:
                    ax.plot(
                        np.arange(20, len(running_average)),
                        running_average[20:],
                        color="red",
                        label="Trailing Average",
                    )
                ax.axhline(0, color="black")
                ax.axhline(13, color="orange", label="target (13)")
                ax.legend()
                ax.figure.savefig("Media/training_scores.png")
                plt.draw()
            break
    return scores, running_average


def main(file_name="/Users/joshuaschoenfield/Downloads/Banana.app", with_plotting=True):
    with get_environment(file_name=file_name) as env:
        from dqn_agent import Agent

        agent = Agent(state_size=37, action_size=4, seed=0)

        scores, running_average = dqn(env=env, agent=agent, with_plotting=with_plotting)

        return scores, running_average


if __name__ == "__main__":
    main()
