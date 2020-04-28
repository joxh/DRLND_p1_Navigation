
# Project 1: Navigation

### Project Details

This project contains code for training an agent to navigate in a large, square world while collecting bananas. Collecting a yellow banana corresponds to a reward of +1 and collecting a blue banana corresponds to a reward of -1. The goal is to collect the highest possible cumulative reward by collecting yellow bananas while avoiding the blue ones.

The **state space has 37 dimensions**. According to the description of the Unity-ML code:

> The state space...contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

There are **4 discrete actions** available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, meaning that it will periodically encounter a terminal state. In order **to be considered "solved" the trained agent must get an average score of +13** over 100 consecutive episodes.

### Getting Started

#### Step 1: Clone the DRLND Repository

The dependencies are located in Udacity's [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). Clone that repository locally, since it contains the necessary python dependencies.

The instructions for creating a correctly configured virtual environment can be found in the `README.md` file in that repository.

#### Step 2: Download the Unity Environment

(The following section is copied from the `README.md` supplied by Udacity and contains the most comprehensive description of how to install the environment)

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. (Different from Udacity's) Place the file in the same folder as you will execute your training code from.

### Instructions

From a jupyter notebook, with the , one can do:

```python
import dqn_training_banana
scores, running_average = dqn_training_banana.main(
    file_name="<path_to_environment>",
    with_plotting=False,
)
```
where `<path_to_environment>` is replaced with the path to the Banana file. This will produce a weights file when completed.

Then, to evaluate the environment: 

```python
import dqn_acting_banana
validation_scores = dqn_acting_banana.main(
    weights_file="checkpoint_banana_2.pth",
    file_name="<path_to_environment>",
)
```

A three session exploration and training can be found in  `Navigation.ipynb`! 