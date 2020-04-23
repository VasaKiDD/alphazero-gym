from agent.master import Ram_Agent
from controller.controllers import Ram_Brain, OptiContMCTS
import gym
import numpy as np

"""
full_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    '11': 11,
    '12': 12,
    '13': 13,
    '14': 14,
    '15': 15,
    '16': 16,
    '17': 17
}
"""

if __name__ == "__main__":
    env = gym.make("MsPacman-ram-v0").unwrapped  # MontezumaRevenge-v0
    env.frameskip = 6
    action_mapping = {"0": 2, "1": 3, "2": 4, "3": 5}
    brain = Ram_Brain(
        action_space_size=4,  # env._n_actions
        embedding=128,
        hidden_size=128,
        uniform_init=(-0.1, 0.1),
        device=0,
    )
    cont_opti = OptiContMCTS(brain, lr=0.001, weight_decay=0.0)
    agent = Ram_Agent(
        env=env,
        controller=brain,
        controller_optimizer=cont_opti,
        forward_limit=np.inf,
        frameskip=env.frameskip,
        mcts_iter_per_node=100,
        value_coeff=1.0,
        embedding_coeff=1.0,
        c_puct=4.0,
        temperature=1.0,
        action_mapping=action_mapping,
    )
    agent.train(num_games=100000, exploit=False, checkpoint_name="ram_params.pck")
