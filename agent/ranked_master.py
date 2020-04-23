import torch
from torch.nn.functional import log_softmax, softmax
import numpy as np
from tqdm import tqdm
import time
import os
import pickle


class Agent:
    def __init__(
        self,
        env,
        controller,
        controller_optimizer,
        mcts_iter_per_node,
        backward_thinking,
        value_coeff,
        c_puct,
        temperature,
        reward_decay,
        frame_stack,
        action_mapping,
        image_ttt,
    ):
        self.env = env
        self.controller = controller
        self.controller_optimizer = controller_optimizer
        self.c_puct = c_puct
        self.temperature = temperature
        self.backward_thinking = backward_thinking
        self.mcts_iter_per_node = mcts_iter_per_node
        self.value_coeff = value_coeff
        self.frames = 0
        self.reward_decay = reward_decay
        self.frame_stack = frame_stack
        self.action_mapping = action_mapping
        self.image_ttt = image_ttt
        self.average_reward = 0.0

    def expand_node(self, node, path_len, oh_action, h, c):
        priors, reward, h, c, oh_action, op = self.controller.forward_once(oh_action, h, c)
        n_children = priors.shape[1]
        # Initialize its children with its probability of being chosen
        # and then stop expanding
        for num_child in range(n_children):
            new_child = {
                "parent": node,
                "childs": [],
                "visit_count": 0.0,
                "total_action_value": 0.0,
                "prior": float(priors[0, num_child]),  # probability of being chosen
            }
            node["childs"].append(new_child)
        # This reward will be propagated backwards through the tree
        reward = float(torch.tanh(reward))  # torch.tanh(reward)
        return node, reward, h, c

    def estimate_q_val(self, node: dict) -> tuple:
        best_child = None
        best_action = 0
        best_val = -np.inf
        # Iterate all the children to fill up the node dict and estimate Q val.
        # Then store the best child found according to the Q value estimation
        for num in range(len(node["childs"])):
            child = node["childs"][num]
            if child["prior"] > 0.0:
                Q = (
                    child["total_action_value"] / child["visit_count"]
                    if child["visit_count"] > 0.0
                    else 0.0
                )
                U = (
                    self.c_puct
                    * child["prior"]
                    * np.sqrt(node["visit_count"])
                    * (1.0 / (1.0 + child["visit_count"]))
                )
                Q += U
                if Q > best_val:
                    best_val = Q
                    best_child = child
                    best_action = num
        return best_child, best_action

    def choose_action(self, node: dict, exploit: bool = False) -> tuple:
        """
		Select the action with the highest Q value from the root node in the MCTS tree.
		:param node: Node to choose the best action from. It should be the root node of the tree.
		:return: probabilities over possible actions, action with highest probability.
		"""
        best_val = -np.inf
        mcts_policy = torch.zeros(1, len(node["childs"])).cuda(self.controller.device)
        n_children = len(node["childs"])

        # Get back to the root node and evaluate its children. Choose the best estimated action
        # and move to the next node. This next node will become the new root
        for num in range(n_children):
            child = node["childs"][num]
            if child["prior"] > 0.0:
                Q = (
                    child["total_action_value"] / child["visit_count"]
                    if child["visit_count"] > 0.0
                    else 0.0
                )
                mcts_policy[0, num] = child["visit_count"]
                if Q > best_val:
                    best_val = Q
                    action_choice = num

        policy = torch.pow(mcts_policy, self.temperature)
        policy = policy / policy.sum()
        act = int(torch.multinomial(policy, 1)[0, 0])
        mcts_policy = mcts_policy / mcts_policy.sum()
        if not exploit:
            return mcts_policy, act
        else:
            return mcts_policy, action_choice

    def choose_new_root(
        self, root_node, action_choice, state_h, state_c, sampled_oh_action, state_ops
    ):
        """
	    Select new root node when MCTS sampling if finished
	    :param root_node:
	    :param action_choice:
	    :param state_h:
	    :param state_c:
	    :param sampled_oh_action:
	    :param state_ops:
	    :return:
	    """
        new_root_node = root_node["childs"][action_choice]
        new_root_node["parent"] = None
        data = self.controller.forward_once(sampled_oh_action, state_h, state_c)
        _, _, state_h, state_c, sampled_oh_action, _ = data
        return new_root_node, sampled_oh_action, state_h, state_c

    def ___mcts(self, obs, exploit=False):
        """Perform an MCTS search over the parameter space and update the controller with the
	    estimated Q value function.
	    """
        root_node = {
            "parent": None,
            "childs": [],
            "visit_count": 1,
            "total_action_value": 0.0,
            "prior": None,
        }
        sampled_oh_action, state_h, state_c = self.controller.init_tensors(obs)
        actions_chosen = []
        mcts_policies = []

        self.mcts_epochs = 0
        for _ in range(self.forward_thinking):
            # Spend some time expanding the tree from your current root node
            for it in range(self.mcts_iter_per_node):
                node = root_node  # go back to the root node
                h = state_h
                c = state_c
                oh_action = sampled_oh_action
                curr_path = actions_chosen[:]
                while True:
                    if len(node["childs"]) == 0:
                        node, reward, h, c = self.expand_node(
                            node, len(curr_path), oh_action, h, c
                        )
                        break
                    else:
                        best_child, best_action = self.estimate_q_val(node)
                        node = best_child
                        curr_path.append(best_action)

                # Propagate information backwards
                while node["parent"] is not None:
                    node["total_action_value"] += reward
                    node["visit_count"] += 1
                    node = node["parent"]
                node["total_action_value"] += reward
                node["visit_count"] += 1

                self.mcts_epochs += 1

            mcts_policy, action_choice = self.choose_action(root_node, exploit)
            root_node, sampled_oh_action, state_h, state_c = self.choose_new_root(
                root_node, action_choice, state_h, state_c, sampled_oh_action, actions_chosen
            )
            mcts_policies.append(mcts_policy)
            actions_chosen.append(action_choice)
        return actions_chosen, mcts_policies

    def mcts(self, obs, exploit=False):
        """Perform an MCTS search over the parameter space and update the controller with the
        estimated Q value function.
        """
        root_node = {
            "parent": None,
            "childs": [],
            "visit_count": 1,
            "total_action_value": 0.0,
            "prior": None,
        }
        sampled_oh_action, state_h, state_c = self.controller.init_tensors(obs)
        actions_chosen = []
        mcts_policies = []

        self.mcts_epochs = 0
        # for _ in range(max_path_len):
        d = time.time()
        for it in range(self.mcts_iter_per_node):
            node = root_node  # go back to the root node
            h = state_h
            c = state_c
            oh_action = sampled_oh_action
            curr_path = actions_chosen[:]
            while True:
                if len(node["childs"]) == 0:
                    node, reward, h, c = self.expand_node(node, len(curr_path), oh_action, h, c)
                    break
                else:
                    best_child, best_action = self.estimate_q_val(node)
                    node = best_child
                    curr_path.append(best_action)

            # Propagate information backwards
            while node["parent"] is not None:
                node["total_action_value"] += reward
                node["visit_count"] += 1
                node = node["parent"]
            node["total_action_value"] += reward
            node["visit_count"] += 1

            self.mcts_epochs += 1

        mcts_policy, action_choice = self.choose_action(root_node, exploit)
        root_node, sampled_oh_action, state_h, state_c = self.choose_new_root(
            root_node, action_choice, state_h, state_c, sampled_oh_action, actions_chosen
        )
        mcts_policies.append(mcts_policy)
        actions_chosen.append(action_choice)
        # print("time :", time.time() - d)
        return actions_chosen, mcts_policies

    def play_game(self, exploit=False):
        obs = self.env.reset()
        obs = self.image_ttt(obs)
        self.env.render()
        self.controller.reset()
        for i in range(270):
            obs, r, e, l = self.env.step(0)
            self.env.render()
        obs = self.image_ttt(obs)
        data = []
        end = False
        curr_live = self.env.ale.lives()
        reward = 0.0
        while not end:
            actions, policies = self.mcts(obs, exploit)
            # print(len(actions))
            episode = {}
            for i in range(len(actions)):
                episode["state"] = obs
                obs = np.zeros_like(obs)
                for f in range(self.frame_stack):
                    ob, rew, end, live = self.env.step(self.action_mapping[str(actions[i])])
                    obs += self.image_ttt(ob)
                    reward += rew
                obs = obs / self.frame_stack
                self.env.render()
                episode["action"] = actions[i]
                episode["mcts_policy"] = policies[i]
                episode["reward"] = reward
                if end or live["ale.lives"] != curr_live:
                    end = True
                    print("End !")
                    break
            data.append(episode)
        return data

    def optimize_controller(
        self, actions_chosen, mcts_policies, rewards, observations, backward=True
    ) -> tuple:
        """

        :param actions_chosen: List of actions that have been selectd by the MCTS algorithm
        :param mcts_policies: List of prob. dist. over actions for each action in the sequence
        :return:
        """
        mcts_policies = torch.stack(tuple(mcts_policies), dim=0)
        mcts_policies = torch.squeeze(mcts_policies)  # (40, 12 vector)
        rewards.insert(0, 0.0)
        # rewards = torch.stack(tuple(rewards), dim=0)
        # rewards = torch.squeeze(rewards, dim=0)

        controller_policies, values, h, c = self.controller.forward_input(
            actions_chosen, observations
        )
        values = torch.tanh(values)  # torch.tanh(values)
        rewards = torch.tensor(rewards).type_as(values)
        cross_entropy = (
            torch.nn.functional.log_softmax(controller_policies, dim=-1) * mcts_policies
        )
        cross_entropy = -1.0 * cross_entropy.sum(dim=-1).mean()
        value_loss = torch.pow(values - rewards, 2).mean()
        total_loss = self.value_coeff * value_loss + cross_entropy
        total_loss.backward()
        self.controller_optimizer.step()
        self.controller_optimizer.zero_grad()

        cross_ent = float(cross_entropy.cpu())
        val_loss = float(value_loss.cpu())

        return cross_ent, val_loss

    def train(
        self,
        num_games,
        exploit=False,
        load: bool = True,
        checkpoint_name="params.pck",
        load_path: str = "",
    ):
        self.frames = 0

        if load:
            try:
                print(
                    "Laoding model data from path {}".format(
                        os.path.join(load_path, checkpoint_name)
                    )
                )
                self.load_model_data(path=load_path, name=checkpoint_name)
            except:
                print("loading model failed")
            print("Resuming from frames {}".format(self.frames))

        for i in range(num_games):
            print("Game ", i + 1)
            game_data = self.play_game(exploit)
            print("Score :", game_data[-1]["reward"])
            if not exploit:
                obss = []
                for j in tqdm(range(len(game_data))):
                    obs = game_data[j]["state"]
                    obss.append(obs)
                    rewards = []
                    actions = []
                    mcts_policies = []
                    for k in range(j, len(game_data)):
                        r = 1.0 if game_data[-1]["reward"] > self.average_reward else -1.0
                        rewards.append(r)
                        actions.append(game_data[k]["action"])
                        mcts_policies.append(game_data[k]["mcts_policy"])
                    ix = max(j - self.backward_thinking, 0)
                    cross_ent, val_loss = self.optimize_controller(
                        actions, mcts_policies, rewards, obss[ix:], True
                    )
                    print(cross_ent, val_loss)
                self.frames += len(game_data)
                self.save_model_data(name=checkpoint_name)
            self.average_reward = 0.9 * self.average_reward + 0.1 * game_data[-1]["reward"]

    def save_model_data(self, path: str = "", name="params.pck"):
        state_dict = {
            "frames": self.frames,
            "model_state": self.controller.state_dict(),
            "controller_optimizer": self.controller_optimizer.state_dict(),
        }
        target_path = os.path.join(path, name)
        pickle.dump(state_dict, open(target_path, "wb"))

    def load_model_data(self, path: str = "", name="params.pck"):
        target_path = os.path.join(path, name)
        state_dict = pickle.load(open(target_path, "rb"))
        self.frames = int(state_dict["frames"])
        print("Just loaded epoch {}".format(self.frames))
        self.controller.load_state_dict(state_dict["model_state"])
        self.controller_optimizer.load_state_dict(state_dict["controller_optimizer"])
