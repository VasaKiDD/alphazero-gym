import torch
from torch.nn.functional import log_softmax, softmax
import numpy as np
from tqdm import tqdm
import time
import os
import pickle

"""
for k in range(len(rewards) - 2, -1, -1):
    if rewards[k + 1] < 0.0:
        decay = rewards[k + 1] * self.reward_decay
    else:
        decay = np.tanh(
            0.001 * rewards[k + 1]) * self.reward_decay
rewards[k] += decay
rewards[k] = rewards[k] if rewards[k] < 1.0 else 1.0


"""


class Agent:
    def __init__(
        self,
        env,
        controller,
        controller_optimizer,
        forward_limit,
        frameskip,
        reward_decay,
        mcts_iter_per_node,
        value_coeff,
        embedding_coeff,
        c_puct,
        temperature,
        action_mapping,
        image_ttt,
    ):
        self.env = env
        self.controller = controller
        self.controller_optimizer = controller_optimizer
        self.c_puct = c_puct
        self.temperature = temperature
        self.mcts_iter_per_node = mcts_iter_per_node
        self.value_coeff = value_coeff
        self.frames = 0
        self.action_mapping = action_mapping
        self.image_ttt = image_ttt
        self.embedding_coeff = embedding_coeff
        self.forward_thinking = 5
        self.frameskip = frameskip
        self.forward_limit = forward_limit
        self.reward_decay = reward_decay
        self.average_reward = 0.0

    def expand_node(self, node, dirichlet=False):
        priors, reward, h, c, oh_action, op = self.controller.forward_once(
            node["oh_action"], node["h"], node["c"]
        )
        n_children = priors.shape[1]
        # Initialize its children with its probability of being chosen
        # and then stop expanding
        priors = torch.squeeze(priors)
        priors = priors.cpu().numpy()
        if dirichlet:
            priors = 0.75 * priors + 0.25 * np.random.dirichlet([0.03] * priors.size)
        for num_child in range(n_children):
            new_child = {
                "parent": node,
                "childs": [],
                "visit_count": 0.0,
                "total_action_value": 0.0,
                "prior": float(priors[num_child]),  # probability of being chosen
                "h": h,
                "c": c,
                "oh_action": self.controller.convert_to_hot(num_child),
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

        if not exploit:
            policy = torch.pow(mcts_policy, self.temperature)
            policy = policy / policy.sum()
            act = int(torch.multinomial(policy, 1)[0, 0])
            mcts_policy = mcts_policy / mcts_policy.sum()
            # print(self.normal_entropy(mcts_policy))
            return mcts_policy, act
        else:
            mcts_policy = mcts_policy / mcts_policy.sum()
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

    def mcts(self, obs, exploit=False):
        """Perform an MCTS search over the parameter space and update the controller with the
        estimated Q value function.
        """
        sampled_oh_action, state_h, state_c = self.controller.init_tensors(obs)
        root_node = {
            "parent": None,
            "childs": [],
            "visit_count": 1,
            "total_action_value": 0.0,
            "prior": None,
            "h": state_h,
            "c": state_c,
            "oh_action": sampled_oh_action,
        }
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
                    if node == root_node:
                        node, reward, h, c = self.expand_node(node, True)
                    else:
                        node, reward, h, c = self.expand_node(node, False)
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
        for i in range(280):
            obs, r, e, l = self.env.step(0)
            self.env.render()
        obs = self.image_ttt(obs)
        obs = [obs, obs, obs, obs]
        observation = np.concatenate(obs, axis=2)
        data = []
        end = False
        curr_reward = 0.0
        curr_live = self.env.ale.lives()
        while not end:
            actions, policies = self.mcts(observation, exploit)
            # print(len(actions))
            episode = {}
            for i in range(len(actions)):
                episode["state"] = observation
                rew = 0.0
                for j in range(self.frameskip):
                    ob, r, end, live = self.env.step(self.action_mapping[str(actions[i])])
                    ob = self.image_ttt(ob)
                    obs.pop(0)
                    obs.append(ob)
                    rew += r
                curr_reward += rew
                observation = np.concatenate(obs, axis=2)
                self.env.render()
                episode["action"] = actions[i]
                episode["mcts_policy"] = policies[i]
                episode["reward"] = curr_reward
                if end or live["ale.lives"] != curr_live:
                    end = True
                    print("End !")
                    break
            data.append(episode)
        return data

    def __optimize_controller(
        self, actions_chosen, mcts_policies, rewards, observations, backward=True
    ) -> tuple:
        """

        :param actions_chosen: List of actions that have been selectd by the MCTS algorithm
        :param mcts_policies: List of prob. dist. over actions for each action in the sequence
        :return:
        """
        mcts_policies = torch.stack(tuple(mcts_policies), dim=0)
        mcts_policies = torch.squeeze(mcts_policies)  # (40, 12 vector)
        # rewards = torch.stack(tuple(rewards), dim=0)
        # rewards = torch.squeeze(rewards, dim=0)

        controller_policies, values, obs_tensor, embedding_tensor = self.controller.forward_input(
            actions_chosen, observations
        )
        values = torch.tanh(values)  # torch.tanh(values)
        rewards = torch.tensor(rewards).type_as(values)

        cross_entropy = (
            torch.nn.functional.log_softmax(controller_policies, dim=-1) * mcts_policies
        )
        cross_entropy = -1.0 * cross_entropy.sum(dim=-1).mean()

        value_loss = torch.pow(values - rewards, 2).mean()

        auto_encoder_loss = torch.pow(embedding_tensor[: len(observations)] - obs_tensor, 2).mean()

        total_loss = (
            self.value_coeff * value_loss
            + cross_entropy
            + self.embedding_coeff * auto_encoder_loss
        )
        total_loss.backward()
        self.controller_optimizer.step()
        self.controller_optimizer.zero_grad()

        cross_ent = float(cross_entropy.cpu())
        val_loss = float(value_loss.cpu())

        return cross_ent, val_loss, auto_encoder_loss

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

        controller_policies, values = self.controller.forward_input(actions_chosen, observations)
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

        return cross_ent, val_loss, 0.0

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
            print("Game ", self.frames)
            game_data = self.play_game(exploit)
            print("Score :", game_data[-1]["reward"])
            until = 80 // self.frameskip
            game_data = game_data[:-until]
            rew = 1.0 if game_data[-1]["reward"] > self.average_reward else -1.0
            self.average_reward = 0.05 * self.average_reward + 0.95 * game_data[-1]["reward"]
            if not exploit:
                perm = np.arange(len(game_data))
                perm = np.random.permutation(perm)
                for j in tqdm(range(len(game_data))):
                    rewards = [rew]
                    actions = []
                    mcts_policies = []
                    forward_obs = []
                    for k in range(perm[j], len(game_data)):
                        rewards.append(rew)
                        actions.append(game_data[k]["action"])
                        mcts_policies.append(game_data[k]["mcts_policy"])
                        forward_obs.append(game_data[k]["state"])
                    ix = min(len(game_data) - perm[j], self.forward_limit)
                    cross_ent, val_loss, encod_loss = self.optimize_controller(
                        actions, mcts_policies, rewards, forward_obs[:ix], True
                    )
                    print(cross_ent, val_loss, float(encod_loss), len(forward_obs))
                self.frames += len(game_data)
                self.save_model_data(name=checkpoint_name)

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

    def normal_entropy(self, vector):
        log_probs1 = -1.0 * torch.log(vector) * vector
        entropy1 = log_probs1.sum(dim=-1).mean()
        log_probs2 = (
            -1.0
            * log_softmax(torch.ones_like(vector) / vector.shape[-1], dim=-1)
            * softmax(torch.ones_like(vector) / vector.shape[-1], dim=-1)
        )
        entropy2 = log_probs2.sum(dim=-1).mean()
        return float(entropy1) / float(entropy2)
