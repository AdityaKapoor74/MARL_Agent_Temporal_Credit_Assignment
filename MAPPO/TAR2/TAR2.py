"""
This file contains the core implementation of the Temporal-Agent Reward Redistribution (TAR²) model.

The main components are:
1.  ShapelyAttention: A module that uses a multi-head attention mechanism inspired by
	Shapley values to compute agent-specific contributions within a timestep.
2.  TAR2: The main sequence-to-sequence model that processes trajectory data. It uses a
	dual temporal-agent transformer architecture to produce unnormalized credit scores.
	It is regularized by an auxiliary inverse dynamics model and uses final-state
	conditioning to produce a stable learning signal.
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .modules import EncoderLayer, init_model, MultiAgentAttention

class ShapelyAttention(nn.Module):
    """
    Approximates Shapley values for agent importance using multi-head attention.

    This module uses a structured coalition sampling strategy that is fully vectorized
    to ensure high performance and handle variable numbers of active agents without errors.
    """
    def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0, coalition_method='structured'):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.n_agents = n_agents
        self.sample_num = sample_num # Note: In structured sampling, this is a factor, not a fixed number.
        self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)
        self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)
        self.coalition_method = coalition_method

    def _generate_structured_coalitions(self, n_agents, agent_mask_batch):
        """
        Generates structured coalition vectors for each item in the batch in a vectorized manner.

        For each active agent in an item, it creates two coalition vectors: one with the
        agent and one without. The results are padded to a uniform size for batch processing.

        Args:
            n_agents (int): The total number of agents.
            agent_mask_batch (torch.Tensor): Mask of active agents. Shape (batch_size, n_agents).

        Returns:
            tuple: A tuple containing:
                - padded_coalitions (torch.Tensor): Padded coalition vectors.
                - sample_mask (torch.Tensor): A mask to ignore padded samples.
        """
        b, _ = agent_mask_batch.shape
        
        # Determine the number of active agents for each item in the batch
        num_active = agent_mask_batch.sum(dim=1).long()
        max_samples_needed = 2 * num_active.max()
        
        if max_samples_needed == 0:
            # Handle the edge case where no agents are active in the entire batch
            return torch.zeros(b, 1, n_agents, device=self.device), torch.zeros(b, 1, device=self.device)

        # Generate random base coalitions for "other" agents
        random_others = torch.bernoulli(torch.full((b, max_samples_needed, n_agents), 0.5, device=self.device))

        # --- Vectorized creation of (with/without) coalitions ---
        agent_indices = torch.arange(n_agents, device=self.device)
        
        # Create masks to isolate each agent
        i_mask = agent_indices.expand(b, n_agents, n_agents) == agent_indices.unsqueeze(1)
        
        # Create base masks for including or excluding agent i
        with_i_mask = torch.ones(b, n_agents, n_agents, device=self.device)
        without_i_mask = with_i_mask.clone()
        without_i_mask[i_mask] = 0
        
        # Combine to form all possible structured samples for all batch items
        base_coalitions = random_others.unsqueeze(1).repeat(1, n_agents, 1, 1)
        coalitions_with = base_coalitions * ~i_mask.unsqueeze(2) + with_i_mask.unsqueeze(2)
        coalitions_without = base_coalitions * ~i_mask.unsqueeze(2) + without_i_mask.unsqueeze(2)
        
        # Interleave the with/without coalitions: [with_a1, without_a1, with_a2, without_a2, ...]
        all_samples = torch.stack([coalitions_with, coalitions_without], dim=3).reshape(b, n_agents * 2, n_agents)

        # --- Create a mask to select only the valid samples for each batch item ---
        # A sample is valid if it corresponds to an *active* agent.
        sample_indices = torch.arange(n_agents * 2, device=self.device).expand(b, n_agents * 2)
        agent_of_sample = sample_indices // 2 # Determine which agent each sample pair belongs to
        
        # The mask should be true only if the agent for that sample is active
        validity_mask = agent_mask_batch[torch.arange(b).unsqueeze(1), agent_of_sample]
        
        # Pad the coalition tensor and the mask to the max number of samples
        padded_coalitions = torch.zeros(b, max_samples_needed, n_agents, device=self.device)
        padded_mask = torch.zeros(b, max_samples_needed, device=self.device)

        # Use the mask to fill the padded tensors in a vectorized way
        for i in range(b):
            valid_samples = all_samples[i][validity_mask[i]]
            num_valid = valid_samples.shape[0]
            if num_valid > 0:
                padded_coalitions[i, :num_valid] = valid_samples
                padded_mask[i, :num_valid] = 1.0

        return padded_coalitions, padded_mask

    def forward(self, input_tensor, agent_temporal_mask):
        b, n_a, t, e = input_tensor.size()
        input_reshaped = input_tensor.permute(0, 2, 1, 3).contiguous().reshape(b * t, n_a, -1)
        agent_mask_reshaped = agent_temporal_mask.permute(0, 2, 1).contiguous().reshape(b * t, n_a)

        coalition = np.arange(self.n_agents)
        np.random.shuffle(coalition)
        agent_embedding = self.agent_embedding(torch.tensor(coalition, device=self.device))[None, :, :].expand(b * t, n_a, self.emb_dim)
        input_with_embedding = input_reshaped + agent_embedding

        # --- Vectorized Coalition Generation ---
        coalition_vectors, sample_mask = self._generate_structured_coalitions(n_a, agent_mask_reshaped)
        num_samples = coalition_vectors.shape[1]

        # Convert coalition vectors to attention masks
        attn_masks = coalition_vectors.unsqueeze(-1) * coalition_vectors.unsqueeze(-2)
        attn_masks = (attn_masks + torch.eye(n_a, device=self.device).unsqueeze(0).unsqueeze(0) > 0).float()
        attn_masks = attn_masks.reshape(-1, n_a, n_a)
        
        # Expand input to match the number of samples for a single parallel forward pass
        input_expanded = input_with_embedding.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(-1, n_a, e)
        
        # --- Single, Fast, Parallelized Attention Computation ---
        marginal_rewards, _ = self.phi(input_expanded, input_expanded, input_expanded, attn_masks)
        
        # --- Masked Averaging ---
        # Reshape results to separate the sample dimension
        marginal_rewards = marginal_rewards.reshape(b * t, num_samples, n_a, e)
        
        # Use the sample_mask to compute a masked average, ignoring padded samples
        sample_mask_expanded = sample_mask.unsqueeze(-1).unsqueeze(-1)
        avg_shapley_reward = (marginal_rewards * sample_mask_expanded).sum(dim=1) / (sample_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1).clamp(min=1))

        # Average attention weights and scores for logging, using the same mask
        self.phi.agent_weights = (self.phi.agent_weights.reshape(b*t, num_samples, n_a, n_a) * sample_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / (sample_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1).clamp(min=1))
        self.phi.agent_scores = (self.phi.agent_scores.reshape(b*t, num_samples, -1, n_a, n_a) * sample_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / (sample_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).clamp(min=1))

        avg_shapley_reward = avg_shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)
        return avg_shapley_reward


class TAR2(nn.Module):
	"""
	The core Temporal-Agent Reward Redistribution (TAR²) model.

	This model processes entire episode trajectories to produce unnormalized contribution
	scores (c_i,t) for each agent at each timestep. It features a dual-transformer
	architecture to capture both temporal and inter-agent dependencies.

	Key architectural components:
	1.  Final-State Conditioning: The reward prediction is conditioned on an embedding
		of the final trajectory outcome to provide a lower-variance learning target.
	2.  Inverse Dynamics Model: An auxiliary task that regularizes the learned
		representations by forcing them to be predictive of agent actions, thus
		grounding them in causal behavior.
	3.  Deterministic Normalization (not in this class): The scores produced by this
		model are passed to a separate, non-learned function that guarantees
		strict return equivalence.
	"""
	def __init__(self, environment, ally_obs_shape, enemy_obs_shape, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, sample_num,
			device, emb_dropout=0.5):
		super().__init__()

		self.environment = environment
		self.emb_dim = emb_dim
		self.n_heads = n_heads
		self.n_layer = n_layer
		self.seq_length = seq_length
		self.device = device
		self.n_agents = n_agents

		# --- Input Embeddings ---
		# Environment-specific observation encoders
		if "StarCraft" in self.environment:
			self.ally_obs_compress_input = nn.Linear(ally_obs_shape, self.emb_dim)
			self.enemy_obs_compress_input = nn.Linear(enemy_obs_shape, self.emb_dim)
		elif "GFootball" in self.environment:
			self.ally_obs_compress_input = nn.Linear(ally_obs_shape, self.emb_dim)

		self.action_emb = nn.Embedding(n_actions + 1, emb_dim)
		self.pos_embedding = nn.Embedding(seq_length, emb_dim)

		# --- Core Transformer Architecture ---
		# A stack of layers, each containing a temporal attention block and an agent attention block
		self.layers = nn.ModuleList([
			nn.ModuleList([
				EncoderLayer(self.emb_dim, self.n_heads, self.emb_dim, emb_dropout),
				ShapelyAttention(emb_dim, n_heads, self.n_agents, sample_num, device, emb_dropout)
			]) for _ in range(self.n_layer)
		])

		# --- Auxiliary Inverse Dynamics Model ---
		# Predicts action from a combination of the current global state and the
		# agent's own historical context.
		# Input: [current_global_state, past_state_action_embedding]
		# Dims:  [emb_dim,            emb_dim*n_layer]
		self.dynamics_model = nn.Sequential(
			nn.Linear(self.emb_dim * (self.n_layer + 2), self.emb_dim),
			nn.GELU(),
			nn.Linear(self.emb_dim, self.emb_dim),
			nn.GELU(),
			nn.Linear(self.emb_dim, n_actions),
		)

		# --- Main Reward Prediction Head ---
		# Predicts the unnormalized scores c_i,t from the post-attention embeddings
		# concatenated with a representation of the final trajectory outcome.
		# Input: [current_context, final_outcome_context]
		# Dims:  [emb_dim*n_layer, emb_dim*n_layer]
		self.reward_prediction = nn.Sequential(
			nn.Linear(2 * emb_dim * self.n_layer, emb_dim),
			nn.GELU(),
			nn.Linear(self.emb_dim, self.emb_dim),
			nn.GELU(),
			nn.Linear(emb_dim, 1),
		)

		init_model(self)

	def get_time_mask(self, episode_length):
		"""Creates a causal attention mask for the temporal transformer."""
		mask = (torch.arange(self.seq_length, device=self.device)[None, :] < episode_length[:, None]).float()
		# Create a triangular mask for causal attention (a timestep can only attend to past timesteps)
		mask = torch.triu(torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))).transpose(-1, -2)
		return mask

	def forward(self, ally_states, enemy_states, actions, episode_length, agent_temporal_mask):
		"""
		Processes a batch of trajectories to produce unnormalized credit scores.

		Args:
			ally_states (torch.Tensor): Ally observations.
			enemy_states (torch.Tensor): Enemy observations.
			actions (torch.Tensor): Actions taken by agents.
			episode_length (torch.Tensor): Length of each episode in the batch.
			agent_temporal_mask (torch.Tensor): Mask for inactive agents/timesteps.

		Returns:
			tuple: A tuple containing:
				- rewards (torch.Tensor): The unnormalized scores c_i,t.
				- temporal_weights (torch.Tensor): Attention weights from the temporal layers.
				- agent_weights (torch.Tensor): Attention weights from the agent layers.
				- temporal_scores (torch.Tensor): Raw attention scores from temporal layers.
				- agent_scores (torch.Tensor): Raw attention scores from agent layers.
				- action_prediction (torch.Tensor): Output of the inverse dynamics model.
		"""
		b, n_a, t, _ = ally_states.size()

		# --- 1. Input Embedding ---
		# Create initial token embeddings from observations, actions, and positions.
		if "StarCraft" in self.environment:
			enemy_obs_embedding = self.enemy_obs_compress_input(enemy_states).mean(dim=1, keepdim=True).repeat(1, n_a, 1, 1)
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
			x = ally_obs_embedding + enemy_obs_embedding
		elif "GFootball" in self.environment:
			ally_obs_embedding = self.ally_obs_compress_input(ally_states)
			x = ally_obs_embedding

		positions = self.pos_embedding(torch.arange(self.seq_length, device=self.device))[None, None, :, :].expand(b, n_a, t, self.emb_dim)
		actions_embed = self.action_emb(actions.long()).squeeze()

		x = x + actions_embed + positions
		state_action_embedding = x # Keep a copy for later use

		# --- 2. Dual Transformer Processing ---
		# Pass embeddings through the stack of temporal and agent attention layers.
		time_mask = self.get_time_mask(episode_length).repeat(n_a, 1, 1)
		x = x.reshape(b * n_a, t, -1).squeeze()

		x_intermediate = []
		temporal_weights, agent_weights, temporal_scores, agent_scores = [], [], [], []

		for layer in self.layers:
			# Temporal attention
			x, _ = layer[0](x, time_mask)
			temporal_scores.append(layer[0].self_attn.temporal_scores)
			temporal_weights.append(layer[0].self_attn.temporal_weights)
			
			x = x.reshape(b, n_a, t, -1)
			
			# Agent attention
			x = layer[1](x, agent_temporal_mask)
			agent_scores.append(layer[1].phi.agent_scores)
			agent_weights.append(layer[1].phi.agent_weights)
			
			x = x.reshape(b * n_a, t, -1).squeeze()
			x_intermediate.append(x)

		# Apply masks to the collected attention weights and scores
		agent_weights = torch.stack(agent_weights, dim=0).reshape(self.n_layer, b, t, n_a, n_a) * agent_temporal_mask.unsqueeze(0).unsqueeze(-1) * agent_temporal_mask.unsqueeze(0).unsqueeze(-2)
		temporal_weights = torch.stack(temporal_weights, dim=0).reshape(self.n_layer, b, n_a, t, t) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(-1) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(-2)
		agent_scores = torch.stack(agent_scores, dim=0).reshape(self.n_layer, b, self.n_heads, t, n_a, n_a) * agent_temporal_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_temporal_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-2)
		temporal_scores = torch.stack(temporal_scores, dim=0).reshape(self.n_layer, b, self.n_heads, n_a, t, t) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-1) * agent_temporal_mask.permute(0, 2, 1).unsqueeze(0).unsqueeze(2).unsqueeze(-2)

		# Concatenate the outputs of all layers to get the final rich representation
		x_intermediate = torch.cat(x_intermediate, dim=-1).reshape(b, n_a, t, -1)

		# --- 3. Inverse Dynamics Model ---
		# This implementation predicts the action at time t based on the global state at t
		# and the agent's contextualized history up to t-1.

		# inverse dynamics model
		# 1. Calculate the global state embedding (pre-attention) for each timestep.
		global_state_embeddings = (state_action_embedding.view(b, n_a, t, self.emb_dim) - actions_embed).reshape(b, n_a, t, self.emb_dim).sum(dim=1, keepdim=True).repeat(1, n_a, 1, 1).reshape(b, n_a, t, -1) / (agent_temporal_mask.transpose(1, 2).sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-5)
		# 2. Get the next global state embedding by shifting the tensor.
		# For the last timestep, there is no "next" state, so we pad with zeros.
		next_global_state_embeddings = torch.cat([global_state_embeddings[:, :, 1:, :], torch.zeros(b, n_a, 1, self.emb_dim).to(self.device)], dim=-2)
		# 3. Get the agent-specific state-action context from the previous timestep (post-attention).
		first_past_state_action_embedding = torch.zeros(b, n_a, 1, self.n_layer*self.emb_dim).to(self.device)
		past_state_action_embeddings = torch.cat([first_past_state_action_embedding, x_intermediate[:, :, :-1, :]], dim=-2)
		# 4. Concatenate all three embeddings to form the input.
		dynamics_model_input = torch.cat([global_state_embeddings, next_global_state_embeddings, past_state_action_embeddings], dim=-1)
		# 5. Predict the action.
		action_prediction = self.dynamics_model(dynamics_model_input)

		# --- 4. Final Reward Prediction ---
		# Get the embedding of the final state for each agent in the batch
		indiv_agent_episode_len = (agent_temporal_mask.sum(dim=-2) - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.emb_dim * self.n_layer).long()
		final_x = torch.gather(x_intermediate, 2, indiv_agent_episode_len).squeeze(2)

		# Create the final outcome embedding (Z) by averaging final states across agents.
		# Detach to treat it as a fixed conditioning variable.
		final_outcome_embedding = final_x.mean(dim=1, keepdim=True).detach()

		# Condition the reward prediction on both the current context and the final outcome
		reward_prediction_embeddings = torch.cat([x_intermediate, final_outcome_embedding.unsqueeze(1).repeat(1, n_a, t, 1)], dim=-1)

		rewards = self.reward_prediction(reward_prediction_embeddings).view(b, n_a, t).contiguous().transpose(1, 2) * agent_temporal_mask.to(self.device)

		return rewards, temporal_weights, agent_weights, temporal_scores, agent_scores, action_prediction
