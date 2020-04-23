from agent.master import Agent
from controller.controllers import Brain, OptiContMCTS
import gym
from PIL import Image
import numpy as np

full_mapping = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "11": 11,
    "12": 12,
    "13": 13,
    "14": 14,
    "15": 15,
    "16": 16,
    "17": 17,
}

action_mapping = {"0": 2, "1": 3, "2": 4, "3": 5}


def pacman_frame(frame):
    frame = frame[3:170, 7:-7, :]
    frame = Image.fromarray(frame)
    # frame = frame.convert("L").resize((200, 200))
    frame = frame.resize((200, 200))
    return np.array(frame)[:, :, :] / 255.0


def montezuma_frame(frame):
    frame = Image.fromarray(frame)
    frame = frame.convert("L").resize((200, 200))
    return np.array(frame)[:, :, None] / 255.0


if __name__ == "__main__":
    env = gym.make("MsPacman-v0").unwrapped  # 'MsPacman-v0' 'MontezumaRevenge-v0'
    env.frameskip = 1
    brain = Brain(
        action_space_size=4,  # env._n_actions
        embedding=100,
        hidden_size=100,
        uniform_init=(-0.1, 0.1),
        device=0,
    )
    cont_opti = OptiContMCTS(brain, lr=0.001, weight_decay=0.0)
    agent = Agent(
        env=env,
        controller=brain,
        controller_optimizer=cont_opti,
        forward_limit=100,
        frameskip=6,
        mcts_iter_per_node=100,
        reward_decay=0.9,
        value_coeff=3.0,
        embedding_coeff=1.0,
        c_puct=1.0,
        temperature=1.0,
        action_mapping=action_mapping,
        image_ttt=pacman_frame,
    )
    agent.train(num_games=100000, exploit=False)
