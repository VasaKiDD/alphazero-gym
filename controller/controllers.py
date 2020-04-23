import torch
from agent.resnet import resnet34
from torch.nn import Linear, LSTMCell, Module, ZeroPad2d
from torch.nn.functional import softmax
from torch.nn.init import uniform_
from torch.optim import Adam


class OptiContMCTS(Module):
    def __init__(self, model, lr=0.001, weight_decay=0.0):
        super(OptiContMCTS, self).__init__()
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class Brain(Module):
    def __init__(
        self,
        action_space_size=18,
        embedding=100,
        hidden_size=100,
        uniform_init=(-0.1, 0.1),
        device=0,
    ):
        """

        :param nodes_per_cells:
        :param hidden_size:
        :param number_of_ops:
        :param uniform_init:
        :param device:
        """
        super(Brain, self).__init__()
        # Internal parameters
        self.device = device
        self.uniform_init = uniform_init

        self.hidden_size = hidden_size
        self.embedding = embedding
        self.action_space = action_space_size

        # Initialize network
        self.rnn = None
        self.actor = None
        self.middle_critic = None
        self.encoder = None
        self.critic = None
        self.padding = None
        self.init_network()
        self.cuda(self.device)

    def init_network(self):
        """Initialize network parameters. This is an actor-critic build on top of a RNN cell. The
        actor is a fully connected layer, and the critic consists of two fully connected layers"""
        self.rnn = LSTMCell(self.action_space, self.hidden_size)
        for p in self.rnn.parameters():
            uniform_(p, self.uniform_init[0], self.uniform_init[1])

        self.actor = Linear(self.hidden_size, self.action_space)
        for p in self.actor.parameters():
            uniform_(p, self.uniform_init[0], self.uniform_init[1])

        self.middle_critic = Linear(self.hidden_size, self.hidden_size // 2)
        for p in self.middle_critic.parameters():
            uniform_(p, self.uniform_init[0], self.uniform_init[1])

        self.critic = Linear(self.hidden_size // 2, 1)
        for p in self.critic.parameters():
            uniform_(p, self.uniform_init[0], self.uniform_init[1])

        self.encoder = resnet34(**{"num_classes": self.embedding})

        self.padding = ZeroPad2d((30, 20, 0, 0))

    def predict(self, oh_action, h, c):
        """
        Run the model for the given internal state and action.
        :param oh_action:
        :param h:
        :param c:
        :return:
        """
        h, c = self.rnn(oh_action, (h, c))
        actor_out = self.actor(h)
        critic_out = torch.nn.functional.relu(self.middle_critic(h))
        critic_out = self.critic(critic_out)
        return actor_out, critic_out, h, c

    def __forward_input(self, sampled, observations):
        """
        From a full state vector (40 dims) predict all the values and actions
         for that vector by building a full path of actions
        :param sampled: 40 dim vector representing the full network connections.
        :return: actor_outs, critic_outs, h, c
        """
        obs_tensor = []
        for i in range(len(observations)):
            obs = torch.from_numpy(observations[i]).cuda(self.device).type(torch.cuda.FloatTensor)
            obs = torch.unsqueeze(obs, 0)
            obs = torch.transpose(obs, 1, 3)
            obs = torch.transpose(obs, 3, 2)
            h = self.encoder(obs)
            obs_tensor.append(h)
        c = torch.zeros(1, self.hidden_size).cuda(self.device)
        oh_action = torch.zeros(1, self.action_space).cuda(self.device)
        actor_outs = []
        critic_outs = []
        h = obs_tensor[0]
        # TODO: stack mask to avoid rep
        embedding_tensor = []
        for i in range(len(sampled)):
            actor_out, critic_out, h, c = self.predict(oh_action, h, c)
            embedding_tensor.append(h)
            actor_outs.append(actor_out)
            critic_outs.append(critic_out)
            # One hot encode the next action to be taken
            action = sampled[i]
            oh_action = torch.zeros_like(oh_action)
            oh_action[0, action] = 1.0

        # Get last value pred from the critic
        h, c = self.rnn(oh_action, (h, c))
        critic_out = torch.nn.functional.relu(self.middle_critic(h))
        critic_out = self.critic(critic_out)
        critic_outs.append(critic_out)

        # Squeeze stuff
        actor_outs = torch.stack(tuple(actor_outs), dim=1)
        actor_outs = torch.squeeze(actor_outs)
        critic_outs = torch.stack(tuple(critic_outs), dim=1)
        critic_outs = torch.squeeze(critic_outs)
        obs_tensor = torch.cat(tuple(obs_tensor), dim=0)
        embedding_tensor = torch.cat(tuple(embedding_tensor), dim=0)
        return actor_outs, critic_outs, obs_tensor, embedding_tensor

    def forward_input(self, sampled, observations):
        """
        From a full state vector (40 dims) predict all the values and actions
         for that vector by building a full path of actions
        :param sampled: 40 dim vector representing the full network connections.
        :return: actor_outs, critic_outs, h, c
        """
        obs = torch.from_numpy(observations[0]).cuda(self.device).type(torch.cuda.FloatTensor)
        obs = torch.unsqueeze(obs, 0)
        obs = torch.transpose(obs, 1, 3)
        obs = torch.transpose(obs, 3, 2)
        # obs = self.padding(obs)
        h = self.encoder(obs)
        c = torch.zeros(1, self.hidden_size).cuda(self.device)
        oh_action = torch.zeros(1, self.action_space).cuda(self.device)
        actor_outs = []
        critic_outs = []
        # TODO: stack mask to avoid rep
        for i in range(len(sampled)):
            actor_out, critic_out, h, c = self.predict(oh_action, h, c)

            actor_outs.append(actor_out)
            critic_outs.append(critic_out)
            # One hot encode the next action to be taken
            action = sampled[i]
            oh_action = torch.zeros_like(oh_action)
            oh_action[0, action] = 1.0

        # Get last value pred from the critic
        h, c = self.rnn(oh_action, (h, c))
        critic_out = torch.nn.functional.relu(self.middle_critic(h))
        critic_out = self.critic(critic_out)
        critic_outs.append(critic_out)

        # Squeeze stuff
        actor_outs = torch.stack(tuple(actor_outs), dim=1)
        actor_outs = torch.squeeze(actor_outs)
        critic_outs = torch.stack(tuple(critic_outs), dim=1)
        critic_outs = torch.squeeze(critic_outs)
        return actor_outs, critic_outs

    def forward_once(self, oh_action, h, c):
        """
        Given an state, a one hot encoded actions and its corresponding index, predict the
         next value of the actor-critic.
        :param oh_action:
        :param h:
        :param c:
        :return:
        """
        with torch.no_grad():
            actor_out, critic_out, h, c = self.predict(oh_action.cuda(self.device), h, c)
            # Mask to build the true action distribution
            action_probs = softmax(actor_out, dim=-1)
            action_probs = action_probs / torch.sum(action_probs)
            # Sample and one hot encoding
            action = torch.multinomial(action_probs, 1)
            oh_action = torch.zeros_like(oh_action)
            oh_action[0, action[0, 0]] = 1.0
        action = torch.squeeze(action)
        return action_probs, critic_out, h, c, oh_action, int(action)

    def init_tensors(self, observation):
        init_action = torch.zeros(1, self.action_space).cuda(self.device)
        c = torch.zeros(1, self.hidden_size).cuda(self.device)
        with torch.no_grad():
            obs = torch.from_numpy(observation).cuda(self.device).type(torch.cuda.FloatTensor)
            obs = torch.unsqueeze(obs, 0)
            obs = torch.transpose(obs, 1, 3)
            obs = torch.transpose(obs, 3, 2)
            # obs = self.padding(obs)
            encoded = self.encoder(obs)
        return init_action, encoded, c

    def convert_to_hot(self, action):
        oh_action = torch.zeros(1, self.action_space).cuda(self.device)
        oh_action[0, action] = 1.0
        return oh_action

    def reset(self):
        return None
