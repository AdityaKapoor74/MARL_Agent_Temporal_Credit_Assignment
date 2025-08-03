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

# class ShapelyAttention(nn.Module):
# 	"""
# 	Approximates Shapley values for agent importance using multi-head attention.

# 	This module computes the marginal contribution of each agent by attending over
# 	randomly sampled coalitions of agents. It serves as the agent-axis attention
# 	mechanism within the main STAS transformer body.
# 	"""
# 	def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0):
# 		super().__init__()
# 		self.emb_dim = emb_dim
# 		self.device = device
# 		self.n_agents = n_agents
# 		self.sample_num = sample_num
# 		# The core multi-agent attention mechanism
# 		self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)
# 		self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)
	
# 	def get_attn_mask(self, shape):
# 		"""Generates a random binary mask to represent a coalition of agents."""
# 		# Create a random mask for sampling a coalition
# 		mask = torch.bernoulli(torch.full((shape, shape), 0.5))
# 		# Ensure an agent always attends to itself
# 		mask = mask - torch.diag(torch.diag(mask)) + torch.eye(shape)
# 		return mask.to(self.device)

# 	def forward(self, input, agent_temporal_mask):
# 		"""
# 		Computes the Shapley-inspired agent representations for a batch of timesteps.

# 		Args:
# 			input (torch.Tensor): A tensor of input sequences with shape
# 								  (batch, n_agents, seq_len, emb_dim).
# 			agent_temporal_mask (torch.Tensor): Mask for inactive agents/timesteps.

# 		Returns:
# 			torch.Tensor: The final Shapley-inspired representations, averaged over samples.
# 		"""
# 		b, n_a, t, e = input.size()
# 		# Reshape for batch processing of all timesteps
# 		input = input.permute(0, 2, 1, 3).contiguous().reshape(b*t, n_a, -1)
		
# 		# Add a randomized agent embedding to break symmetry
# 		coalition = np.arange(self.n_agents)
# 		np.random.shuffle(coalition)
# 		agent_embedding = self.agent_embedding(torch.tensor(coalition).to(self.device))[None, :, :].expand(b*t, n_a, self.emb_dim)
# 		input = input + agent_embedding
		
# 		shapley_reward = []
# 		# Monte Carlo approximation of Shapley values
# 		for _ in range(self.sample_num):
# 			attn_mask = self.get_attn_mask(n_a).unsqueeze(0).repeat(b*t, 1, 1)
# 			marginal_reward, _ = self.phi(input, input, input, attn_mask)
# 			shapley_reward.append(marginal_reward)

# 		# Average the representations over all sampled coalitions
# 		shapley_reward = sum(shapley_reward)/self.sample_num
# 		# Reshape back to the original batch format
# 		shapley_reward = shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)

# 		return shapley_reward

class ShapelyAttention(nn.Module):
    """
    Approximates Shapley values for agent importance using multi-head attention.
    This module computes the marginal contribution of each agent by attending over
    randomly sampled coalitions of agents. It serves as the agent-axis attention
    mechanism within the main STAS transformer body.
    
    Now properly incorporates agent_masks in coalition formation and attention.
    """
    def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.n_agents = n_agents
        self.sample_num = sample_num
        # The core multi-agent attention mechanism
        self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)
        self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)
    
    def get_masked_coalition(self, active_agents):
        """
        Generates a coalition mask that only includes active agents.
        
        Args:
            active_agents (torch.Tensor): Boolean tensor indicating which agents are active
            
        Returns:
            torch.Tensor: Coalition mask for active agents
        """
        n_active = active_agents.sum().item()
        if n_active == 0:
            # No active agents - return identity mask
            return torch.eye(self.n_agents, device=self.device)
        
        # Get indices of active agents
        active_indices = torch.where(active_agents)[0]
        
        # Create coalition among active agents only
        coalition_mask = torch.zeros(self.n_agents, device=self.device)
        
        # Randomly sample a subset of active agents for the coalition
        if n_active > 0:
            # Sample coalition size uniformly from 1 to n_active
            coalition_size = torch.randint(1, n_active + 1, (1,), device=self.device).item()
            
            # Randomly select agents for the coalition
            perm = torch.randperm(n_active, device=self.device)
            selected_active = active_indices[perm[:coalition_size]]
            coalition_mask[selected_active] = 1.0
        
        # Convert to attention mask
        attn_mask = coalition_mask.unsqueeze(-1) * coalition_mask.unsqueeze(0)
        
        # Ensure each agent attends to itself (diagonal = 1)
        attn_mask = attn_mask + torch.eye(self.n_agents, device=self.device)
        
        # Apply agent mask - inactive agents cannot attend or be attended to
        agent_mask_2d = active_agents.unsqueeze(-1) * active_agents.unsqueeze(0)
        attn_mask = attn_mask * agent_mask_2d.float()
        
        # Ensure inactive agents attend only to themselves (for numerical stability)
        inactive_mask = ~active_agents
        attn_mask[inactive_mask, :] = 0.0
        attn_mask[:, inactive_mask] = 0.0
        attn_mask[inactive_mask, inactive_mask] = 1.0
        
        return attn_mask
    
    def get_attn_mask(self, shape, active_agents=None):
        """
        Generates a random binary mask to represent a coalition of agents.
        Now incorporates agent masks to only form coalitions among active agents.
        
        Args:
            shape (int): Number of agents
            active_agents (torch.Tensor): Boolean mask indicating active agents
            
        Returns:
            torch.Tensor: Attention mask respecting active agents
        """
        if active_agents is not None:
            return self.get_masked_coalition(active_agents)
        else:
            # Fallback to original behavior if no mask provided
            mask = torch.bernoulli(torch.full((shape, shape), 0.5, device=self.device))
            mask = mask - torch.diag(torch.diag(mask)) + torch.eye(shape, device=self.device)
            return mask

    def forward(self, input, agent_temporal_mask):
        """
        Computes the Shapley-inspired agent representations for a batch of timesteps.
        
        Args:
            input (torch.Tensor): A tensor of input sequences with shape
                                (batch, n_agents, seq_len, emb_dim).
            agent_temporal_mask (torch.Tensor): Mask for inactive agents/timesteps with shape
                                              (batch, n_agents, seq_len).
                                              
        Returns:
            torch.Tensor: The final Shapley-inspired representations, averaged over samples.
        """
        b, n_a, t, e = input.size()
        
        # Reshape for batch processing of all timesteps
        input = input.permute(0, 2, 1, 3).contiguous().reshape(b*t, n_a, -1)
        
        # Reshape agent mask to match: (batch*seq_len, n_agents)
        agent_mask_reshaped = agent_temporal_mask.permute(0, 2, 1).contiguous().reshape(b*t, n_a)
        
        # Add a randomized agent embedding to break symmetry
        coalition = np.arange(self.n_agents)
        np.random.shuffle(coalition)
        agent_embedding = self.agent_embedding(torch.tensor(coalition, device=self.device))[None, :, :].expand(b*t, n_a, self.emb_dim)
        input = input + agent_embedding
        
        shapley_reward = []
        
        # Monte Carlo approximation of Shapley values
        for i in range(self.sample_num):
            # Generate attention masks for each batch*time step
            batch_attn_masks = []
            
            for bt_idx in range(b*t):
                # Get active agents for this batch*time step
                active_agents = agent_mask_reshaped[bt_idx].bool()
                
                # Generate coalition mask considering only active agents
                attn_mask = self.get_attn_mask(n_a, active_agents)
                batch_attn_masks.append(attn_mask)
            
            # Stack all attention masks
            batch_attn_mask = torch.stack(batch_attn_masks, dim=0)  # (b*t, n_a, n_a)
            
            # Forward pass through multi-agent attention
            marginal_reward, _ = self.phi(input, input, input, batch_attn_mask)
            
            # Apply agent mask to ensure inactive agents have zero contribution
            agent_mask_expanded = agent_mask_reshaped.unsqueeze(-1).expand(-1, -1, e)
            marginal_reward = marginal_reward * agent_mask_expanded
            
            shapley_reward.append(marginal_reward)
        
        # Average the representations over all sampled coalitions
        shapley_reward = sum(shapley_reward) / self.sample_num
        
        # Reshape back to the original batch format
        shapley_reward = shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)
        
        return shapley_reward
	


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
	def __init__(self, environment, version, ally_obs_shape, enemy_obs_shape, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, sample_num,
			device, emb_dropout=0.5):
		super().__init__()

		self.environment = environment
		self.version = version
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
		if self.version != "no_inverse_dynamics": 
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
		if self.version == "no_final_outcome":
			self.reward_prediction = nn.Sequential(
				nn.Linear(emb_dim * self.n_layer, emb_dim),
				nn.GELU(),
				nn.Linear(self.emb_dim, self.emb_dim),
				nn.GELU(),
				nn.Linear(emb_dim, 1),
			)
		else:
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
		if self.version == "no_inverse_dynamics":
			action_prediction = torch.zeros(b, n_a, t, self.emb_dim).to(self.device)
		else:
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
		if self.version == "no_final_outcome":
			# If no final outcome conditioning, just use the intermediate states.
			reward_prediction_embeddings = x_intermediate
		else:
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
