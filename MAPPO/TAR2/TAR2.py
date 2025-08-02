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

    This module supports three coalition sampling methods:
    - 'random': Pure Monte Carlo sampling of random coalitions.
    - 'structured': For each agent, samples coalitions with and without it.
    - 'stratified': Samples coalitions stratified by size to reduce variance.
    """
    def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0, coalition_method='stratified'):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.n_agents = n_agents
        self.sample_num = sample_num
        self.coalition_method = coalition_method
        
        self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)
        self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)

    def _generate_random_masks(self, batch_shape, n_agents):
        """Generates purely random binary masks for coalitions."""
        # For random sampling, the number of samples is fixed.
        # Returns a list of tensors, where each tensor is for one batch item.
        mask = torch.bernoulli(torch.full((batch_shape, self.sample_num, n_agents), 0.5, device=self.device))
        return list(torch.unbind(mask, dim=0))

    def _generate_structured_masks(self, n_agents, agent_mask_batch):
        """
        Generates structured coalitions (with/without each agent).
        Returns a list of tensors, one for each batch item, with variable samples.
        """
        b, _ = agent_mask_batch.shape
        list_of_coalition_vectors = []

        for i in range(b): # Iterate over each item in the batch
            agent_mask = agent_mask_batch[i]
            active_agent_indices = torch.where(agent_mask > 0)[0]
            
            if len(active_agent_indices) == 0:
                list_of_coalition_vectors.append(torch.empty(0, n_agents, device=self.device))
                continue

            coalition_vectors = []
            for agent_idx in active_agent_indices:
                other_active_agents = active_agent_indices[active_agent_indices != agent_idx]
                
                # Sample a random subset of the *other active* agents
                num_others = len(other_active_agents)
                if num_others > 0:
                    random_others_mask = torch.bernoulli(torch.full((num_others,), 0.5, device=self.device)).bool()
                    subset_others = other_active_agents[random_others_mask]
                else:
                    subset_others = torch.tensor([], dtype=torch.long, device=self.device)

                # Create coalition WITH agent i
                coalition_with = torch.zeros(n_agents, device=self.device)
                coalition_with[agent_idx] = 1
                coalition_with[subset_others] = 1
                
                # Create coalition WITHOUT agent i
                coalition_without = coalition_with.clone()
                coalition_without[agent_idx] = 0
                
                coalition_vectors.extend([coalition_with, coalition_without])
            
            list_of_coalition_vectors.append(torch.stack(coalition_vectors))
            
        return list_of_coalition_vectors

    def _generate_stratified_masks(self, n_agents, agent_mask_batch):
        """
        Generates coalitions using stratified sampling based on coalition size.
        Returns a list of tensors, one for each batch item, with a fixed number of samples.
        """
        b, _ = agent_mask_batch.shape
        list_of_coalition_vectors = []
        
        active_indices_list = [torch.where(m > 0)[0] for m in agent_mask_batch]

        for i in range(b):
            num_active = len(active_indices_list[i])
            active_indices = active_indices_list[i]
            
            if num_active == 0:
                list_of_coalition_vectors.append(torch.zeros(self.sample_num, n_agents, device=self.device))
                continue

            # Sample coalition sizes uniformly from 0 to num_active
            sampled_sizes = torch.randint(0, num_active + 1, (self.sample_num,), device=self.device)
            
            coalition_vectors = []
            for k in sampled_sizes:
                coalition_mask_vector = torch.zeros(n_agents, device=self.device)
                if k > 0:
                    # Randomly choose k agents from the active set
                    perm = torch.randperm(num_active, device=self.device)
                    chosen_indices = active_indices[perm[:k]]
                    coalition_mask_vector[chosen_indices] = 1.0
                coalition_vectors.append(coalition_mask_vector)
            
            list_of_coalition_vectors.append(torch.stack(coalition_vectors))

        return list_of_coalition_vectors

    def forward(self, input_tensor, agent_temporal_mask):
        b, n_a, t, e = input_tensor.size()
        input_reshaped = input_tensor.permute(0, 2, 1, 3).contiguous().reshape(b * t, n_a, -1)
        agent_mask_reshaped = agent_temporal_mask.permute(0, 2, 1).contiguous().reshape(b * t, n_a)

        coalition = np.arange(self.n_agents)
        np.random.shuffle(coalition)
        agent_embedding = self.agent_embedding(torch.tensor(coalition, device=self.device))[None, :, :].expand(b * t, n_a, self.emb_dim)
        input_with_embedding = input_reshaped + agent_embedding

        # --- Select Coalition Generation Method ---
        if self.coalition_method == 'random':
            list_of_coalition_vectors = self._generate_random_masks(b * t, n_a)
        elif self.coalition_method == 'structured':
            list_of_coalition_vectors = self._generate_structured_masks(n_a, agent_mask_reshaped)
        elif self.coalition_method == 'stratified':
            list_of_coalition_vectors = self._generate_stratified_masks(n_a, agent_mask_reshaped)
        else:
            raise ValueError(f"Unknown coalition method: {self.coalition_method}")

        # --- Process each item in the batch individually to handle variable samples ---
        avg_rewards_list = []
        avg_weights_list = []
        avg_scores_list = []

        for i in range(b * t):
            coalition_vectors = list_of_coalition_vectors[i]
            num_samples = coalition_vectors.shape[0]

            if num_samples == 0:
                # Handle cases with no active agents or zero samples
                avg_rewards_list.append(torch.zeros(n_a, e, device=self.device))
                avg_weights_list.append(torch.zeros(n_a, n_a, device=self.device))
                avg_scores_list.append(torch.zeros(self.phi.n_head, n_a, n_a, device=self.device))
                continue

            # Create attention masks from coalition vectors for this specific item
            attn_masks = coalition_vectors.unsqueeze(-1) * coalition_vectors.unsqueeze(-2)
            attn_masks = attn_masks + torch.eye(n_a, device=self.device).unsqueeze(0)
            attn_masks = (attn_masks > 0).float()
            
            # Expand the single input item to match the number of samples
            input_expanded = input_with_embedding[i].unsqueeze(0).repeat(num_samples, 1, 1)
            
            # Run the attention model
            marginal_rewards, _ = self.phi(input_expanded, input_expanded, input_expanded, attn_masks)

            # Average the results for this item and store them
            avg_rewards_list.append(marginal_rewards.mean(dim=0))
            avg_weights_list.append(self.phi.agent_weights.mean(dim=0))
            avg_scores_list.append(self.phi.agent_scores.mean(dim=0))

        # Stack the final averaged results, which are now guaranteed to be the same size
        avg_shapley_reward = torch.stack(avg_rewards_list)
        self.phi.agent_weights = torch.stack(avg_weights_list)
        self.phi.agent_scores = torch.stack(avg_scores_list)
        
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
