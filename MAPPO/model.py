"""
This file defines the core neural network architectures for the MAPPO agent.

It includes:
1.  PopArt: A specialized normalization layer that adaptively rescales the targets
    for the value function, which is crucial for stabilizing learning in multi-agent
    settings with varying reward scales.
2.  Policy: The actor network, which takes local observations and outputs action
    probabilities for a single agent. It can be configured as a recurrent network (GRU).
3.  Value: The centralized critic network, which takes a global state representation
    (constructed from all agents' information) and outputs a value estimate for each agent.
    It can also be configured as a recurrent network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopArt(nn.Module):
    """
    Preserving Outputs Precisely while Adaptively Rescaling Targets (PopArt).

    This is a normalization technique used for the value function targets (the returns).
    It normalizes the targets to have zero mean and unit variance, which stabilizes
    the learning process of the value network. Crucially, it then de-normalizes the
    output of the value network, ensuring that the value estimates remain on their
    original, unnormalized scale.

    Reference: Hessel et al., 2018, "Multi-task Deep Reinforcement Learning with PopArt"
    """
    def __init__(self, input_shape, num_agents, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()
        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Running statistics for mean and variance
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

    def running_mean_var(self):
        """Calculates the debiased running mean and variance."""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector, mask, train=True):
        """
        Normalizes an input vector and updates the running statistics if in training mode.

        Args:
            input_vector (torch.Tensor): The vector to be normalized (e.g., returns).
            mask (torch.Tensor): A mask to exclude padded/inactive steps from statistics.
            train (bool): Whether to update the running statistics.

        Returns:
            torch.Tensor: The normalized input vector.
        """
        if train:
            # Update running statistics with the current batch
            detached_input = input_vector.detach()
            batch_mean = (detached_input * mask).sum() / mask.sum()
            batch_sq_mean = ((detached_input ** 2) * mask).sum() / mask.sum()

            self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
            self.running_mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
            self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var).clamp(min=self.epsilon)
        return out

    def denormalize(self, input_vector):
        """Transforms a normalized vector back to its original scale."""
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var).clamp(min=self.epsilon) + mean
        return out

def init(module, weight_init, bias_init, gain=1):
    """Applies a specified weight and bias initialization to a module."""
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    """A helper function for orthogonal initialization of linear layers."""
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Policy(nn.Module):
    """
    The Actor network for a single agent.

    It takes an agent's local observation and last action as input and outputs a
    probability distribution over the discrete action space. It can be configured
    to use a GRU for recurrent processing of sequential data.
    """
    def __init__(self, use_recurrent_policy, obs_input_dim, num_actions, num_agents, rnn_num_layers, rnn_hidden_actor, device):
        super(Policy, self).__init__()
        self.use_recurrent_policy = use_recurrent_policy
        self.rnn_hidden_actor = rnn_hidden_actor
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.device = device
        self.mask_value = torch.tensor(torch.finfo(torch.float).min, dtype=torch.float).to(self.device)

        # Embeddings for agent ID and the previous action
        self.agent_embedding = nn.Embedding(self.num_agents, self.rnn_hidden_actor)
        self.action_embedding = nn.Embedding(self.num_actions + 1, self.rnn_hidden_actor)

        # Main network body
        if self.use_recurrent_policy:
            self.obs_embedding = nn.Sequential(
                init_(nn.Linear(obs_input_dim, rnn_hidden_actor), activate=True),
                nn.GELU(),
            )
            self.obs_embed_layer_norm = nn.LayerNorm(self.rnn_hidden_actor)
            self.RNN = nn.GRU(input_size=rnn_hidden_actor, hidden_size=rnn_hidden_actor, num_layers=rnn_num_layers, batch_first=True)
            for name, param in self.RNN.named_parameters():
                if 'bias' in name: nn.init.constant_(param, 0)
                elif 'weight' in name: nn.init.orthogonal_(param)
            self.Layer_2 = nn.Sequential(
                nn.LayerNorm(rnn_hidden_actor),
                init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
            )
        else: # Non-recurrent policy
            self.net = nn.Sequential(
                init_(nn.Linear(obs_input_dim, rnn_hidden_actor), activate=True),
                nn.GELU(),
                nn.LayerNorm(rnn_hidden_actor),
                init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
            )

    def forward(self, local_observations, last_actions, hidden_state, mask_actions):
        batch, timesteps, _, _ = local_observations.shape
        
        # Create input embeddings
        agent_embedding = self.agent_embedding(torch.arange(self.num_agents, device=self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.rnn_hidden_actor)
        last_action_embedding = self.action_embedding(last_actions.long())
        obs_embedding = self.obs_embedding(local_observations)
        
        # Combine embeddings
        final_obs_embedding = self.obs_embed_layer_norm(obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch * self.num_agents, timesteps, -1)

        if self.use_recurrent_policy:
            hidden_state = hidden_state.reshape(self.rnn_num_layers, batch * self.num_agents, -1)
            output, h = self.RNN(final_obs_embedding, hidden_state)
            output = output.reshape(batch, self.num_agents, timesteps, -1).permute(0, 2, 1, 3)
            logits = self.Layer_2(output)
        else:
            logits = self.net(local_observations)
            h = hidden_state # Pass hidden state through for consistent API

        # Apply action mask to logits
        logits = torch.where(mask_actions, logits, self.mask_value)
        
        return F.softmax(logits, dim=-1), h

class Value(nn.Module):
    """
    The centralized Critic network.

    It takes a global state representation, constructed from the observations and actions
    of all agents, and outputs a single value estimate for each agent. This is the "V"
    in the actor-critic setup. It can be configured to use a GRU.
    """
    def __init__(self, environment, use_recurrent_critic, global_observation_input_dim, ally_obs_input_dim, enemy_obs_input_dim,
                 num_agents, num_enemies, num_actions, rnn_num_layers, comp_emb_shape, device):
        super(Value, self).__init__()
        self.environment = environment
        self.use_recurrent_critic = use_recurrent_critic
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.num_actions = num_actions
        self.device = device

        # Define the input dimension based on the environment
        if "StarCraft" in self.environment:
            # For SC2, the input is a complex concatenation of agent-centric features
            input_dim = (self.num_agents + ally_obs_input_dim + self.num_actions) * self.num_agents + enemy_obs_input_dim * self.num_enemies
        elif "GFootball" in self.environment:
            input_dim = global_observation_input_dim + self.num_agents + self.num_actions * self.num_agents

        self.embedding = nn.Sequential(
            init_(nn.Linear(input_dim, comp_emb_shape * 2), activate=True),
            nn.GELU(),
            nn.LayerNorm(comp_emb_shape * 2),
            init_(nn.Linear(comp_emb_shape * 2, comp_emb_shape), activate=True),
            nn.GELU(),
            nn.LayerNorm(comp_emb_shape),
        )

        if self.use_recurrent_critic:
            self.RNN = nn.GRU(input_size=comp_emb_shape, hidden_size=comp_emb_shape, num_layers=rnn_num_layers, batch_first=True)
            for name, param in self.RNN.named_parameters():
                if 'bias' in name: nn.init.constant_(param, 0)
                elif 'weight' in name: nn.init.orthogonal_(param)

        self.value_layer = nn.Sequential(
            nn.LayerNorm(comp_emb_shape),
            init_(nn.Linear(comp_emb_shape, 1), activate=False)
        )
        
        self.one_hot_actions = torch.eye(self.num_actions, device=self.device)
        self.agent_ids = torch.eye(self.num_agents, device=self.device)

    def forward(self, global_observations, ally_states, enemy_states, actions, rnn_hidden_state):
        batch, timesteps, _, _ = ally_states.shape if "StarCraft" in self.environment else global_observations.shape

        # --- Construct Centralized State Representation ---
        if "StarCraft" in self.environment:
            # For each agent, create a view of the state where its own information is first,
            # followed by the circularly shifted information of its allies.
            one_hot_actions = self.one_hot_actions[actions.long()]
            agent_ids = self.agent_ids.reshape(1, 1, self.num_agents, self.num_agents).repeat(batch, timesteps, 1, 1)
            ally_states_with_info = torch.cat([agent_ids, ally_states, one_hot_actions], dim=-1)
            
            # Create agent-centric views by rolling the agent dimension
            rolled_ally_states = torch.stack([torch.roll(ally_states_with_info, shifts=-i, dims=2) for i in range(self.num_agents)], dim=2)
            # Mask out the focal agent's own action from its input
            rolled_ally_states[:, :, :, 0, -self.num_actions:] = 0
            
            # Concatenate all information
            flat_ally_states = rolled_ally_states.reshape(batch, timesteps, self.num_agents, -1)
            flat_enemy_states = enemy_states.reshape(batch, timesteps, 1, -1).repeat(1, 1, self.num_agents, 1)
            central_state = torch.cat([flat_ally_states, flat_enemy_states], dim=-1)
            
        elif "GFootball" in self.environment:
            # For GFootball, concatenate global obs with agent IDs and actions of other agents
            one_hot_actions = self.one_hot_actions[actions.long()].unsqueeze(2).repeat(1, 1, self.num_agents, 1, 1)
            # Mask out each agent's own action
            for i in range(self.num_agents):
                one_hot_actions[:, :, i, i, :] = 0
            flat_other_actions = one_hot_actions.reshape(batch, timesteps, self.num_agents, -1)
            agent_ids = self.agent_ids.reshape(1, 1, self.num_agents, self.num_agents).repeat(batch, timesteps, 1, 1)
            central_state = torch.cat([global_observations, agent_ids, flat_other_actions], dim=-1)
        
        # --- Process through Network ---
        final_state_embedding = self.embedding(central_state).permute(0, 2, 1, 3).reshape(batch * self.num_agents, timesteps, -1)
        
        if self.use_recurrent_critic:
            final_state_embedding, h = self.RNN(final_state_embedding, rnn_hidden_state)
        else:
            h = rnn_hidden_state # Pass through for consistent API

        Value = self.value_layer(final_state_embedding)
        
        # Reshape to (batch, timesteps, num_agents)
        Value = Value.reshape(batch, self.num_agents, timesteps, -1).permute(0, 2, 1, 3).squeeze(-1)

        return Value, h