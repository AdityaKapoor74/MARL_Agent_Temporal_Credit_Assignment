"""
This file defines the core building blocks for the transformer-based architecture used in the TAR² model.

It includes standard transformer components like Multi-Head Attention and Position-wise Feed-Forward Networks,
as well as a custom MultiAgentAttention module for handling inter-agent interactions. A sophisticated
initialization function, `init_model`, is also provided to ensure stable training of deep transformer models.
"""
import torch
from torch import nn
import torch.nn.functional as F
import math

def init_model(model):
	"""
	Applies a specialized weight initialization scheme to the model.

	This function uses Kaiming initialization for linear layers, scaled by the depth of the
	transformer to prevent vanishing/exploding gradients. It applies smaller-scale initialization
	for attention-related weights and the final reward prediction layer to further stabilize training.

	Args:
		model (nn.Module): The model to be initialized.
	"""
	
	# Count the number of transformer layers to calculate a depth-based scaling factor.
	depth = 0
	for name, module in model.named_modules():
		if 'layers.' in name and ('EncoderLayer' in str(type(module)) or hasattr(module, 'self_attn')):
			depth += 1
	
	# The scaling factor is based on the principle of residual connections in deep networks.
	depth_scale = math.sqrt(2.0 / max(depth, 1)) if depth > 2 else 1.0
	
	def init_fn(m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
			m.weight.data *= depth_scale
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		
		elif isinstance(m, nn.Embedding):
			nn.init.normal_(m.weight, 0, 0.01)  # Smaller std for embeddings
		
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
	
	model.apply(init_fn)
	
	# Apply a smaller initialization scale specifically to attention weights.
	for name, module in model.named_modules():
		if hasattr(module, 'W_Q') and hasattr(module, 'W_K'):
			scale = 0.5 * depth_scale
			
			if hasattr(module, 'W_Q'):
				if isinstance(module.W_Q, nn.ModuleList):
					for q in module.W_Q:
						q.weight.data *= scale
				else:
					module.W_Q.weight.data *= scale
			
			for attr in ['W_K', 'W_V', 'linear']:
				if hasattr(module, attr):
					getattr(module, attr).weight.data *= scale
	
	# Use an even smaller scale for the final output layer to keep initial predictions small.
	for name, module in model.named_modules():
		if 'reward_prediction' in name.lower():
			if isinstance(module, nn.Linear):
				module.weight.data *= 0.05
	
	print(f"TAR² v2 initialization complete (depth={depth}, scale={depth_scale:.3f})")

class EncoderLayer(nn.Module):
	"""
	A standard transformer encoder layer.

	It consists of a multi-head self-attention mechanism followed by a position-wise
	feed-forward network. Layer normalization and residual connections are applied
	around each of the two sub-layers.
	"""
	def __init__(self, d_hidden, n_head, d_ff, dropout, layer_norm_epsilon=1e-12):
		super().__init__()
		self.self_attn = MultiHeadAttention(d_hidden, n_head, dropout)
		self.layer_norm1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
		self.pos_ffn = PoswiseFeedForwardNet(d_hidden, d_ff, dropout)
		self.layer_norm2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
	
	def forward(self, inputs, attn_mask):
		"""
		Forward pass for the encoder layer.

		Args:
			inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_hidden).
			attn_mask (torch.Tensor): Attention mask to prevent attending to certain positions.

		Returns:
			tuple: A tuple containing:
				- ffn_outputs (torch.Tensor): The output of the encoder layer.
				- attn_prob (torch.Tensor): The attention probabilities.
		"""
		# Self-attention block with residual connection and layer normalization
		att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
		ffn_inputs = self.layer_norm1(inputs + att_outputs)
		
		# Feed-forward block with residual connection and layer normalization
		ffn_outputs = self.pos_ffn(ffn_inputs)
		ffn_outputs = self.layer_norm2(ffn_inputs + ffn_outputs)
		
		return ffn_outputs, attn_prob

class MultiHeadAttention(nn.Module):
	"""
	Standard multi-head attention mechanism.

	This module projects the queries, keys, and values into multiple heads, performs
	scaled dot-product attention independently for each head, and then concatenates
	the results.
	"""
	def __init__(self, d_hidden, n_head, dropout):
		super().__init__()
		if d_hidden % n_head != 0:
			raise ValueError(f"The hidden size ({d_hidden}) is not a multiple of the number of attention heads ({n_head})")
		
		self.d_hidden = d_hidden
		self.n_head = n_head
		self.attn_head_size = int(self.d_hidden / self.n_head)
		self.all_head_size = self.attn_head_size * self.n_head

		self.W_Q = nn.Linear(self.d_hidden, self.all_head_size)
		self.W_K = nn.Linear(self.d_hidden, self.all_head_size)
		self.W_V = nn.Linear(self.d_hidden, self.all_head_size)
		self.scaled_dot_attn = ScaledDotProductAttention(dropout, self.attn_head_size)
		self.linear = nn.Linear(self.all_head_size, self.d_hidden)
		self.dropout = nn.Dropout(dropout)

		# Store attention weights and scores for analysis
		self.temporal_weights = None
		self.temporal_scores = None

	def transpose_for_scores(self, x):
		"""Reshapes the input tensor to be compatible with multi-head attention."""
		new_x_shape = x.size()[:-1] + (self.n_head, self.attn_head_size)
		x = x.view(new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, Q, K, V, attn_mask):
		"""
		Forward pass for multi-head attention.

		Args:
			Q (torch.Tensor): Query tensor.
			K (torch.Tensor): Key tensor.
			V (torch.Tensor): Value tensor.
			attn_mask (torch.Tensor): Attention mask.

		Returns:
			tuple: A tuple containing:
				- output (torch.Tensor): The final attended output.
				- attn_prob (torch.Tensor): The attention probabilities.
		"""
		# Project Q, K, V into multi-head representations
		q_s = self.transpose_for_scores(self.W_Q(Q))
		k_s = self.transpose_for_scores(self.W_K(K))
		v_s = self.transpose_for_scores(self.W_V(V))

		# Expand mask to match the number of heads
		attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

		# Compute scaled dot-product attention
		context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
		
		# Store weights and scores for later analysis
		self.temporal_weights = attn_prob.mean(dim=1)
		self.temporal_scores = self.scaled_dot_attn.scores

		# Concatenate heads and apply final linear layer
		context = context.permute(0, 2, 1, 3).contiguous()
		context = context.view(context.size()[:-2] + (self.all_head_size,))
		output = self.linear(context)
		output = self.dropout(output)
		
		return output, attn_prob

class MultiAgentAttention(nn.Module):
	"""
	A specialized multi-head attention for inter-agent interactions.

	This module uses a separate query projection (W_Q) for each agent, allowing
	each agent to learn a unique way of querying other agents. This is a key
	component of the ShapelyAttention module.
	"""
	def __init__(self, d_hidden, n_head, n_agents, dropout, device):
		super().__init__()
		if d_hidden % n_head != 0:
			raise ValueError(f"The hidden size ({d_hidden}) is not a multiple of the number of attention heads ({n_head})")
		
		self.d_hidden = d_hidden
		self.n_head = n_head
		self.n_agents = n_agents
		self.attn_head_size = int(self.d_hidden / self.n_head)
		self.all_head_size = self.attn_head_size * self.n_head

		# Each agent gets its own query projection matrix
		self.W_Q = nn.ModuleList([nn.Linear(self.d_hidden, self.all_head_size) for _ in range(self.n_agents)])
		self.W_K = nn.Linear(self.d_hidden, self.all_head_size)
		self.W_V = nn.Linear(self.d_hidden, self.all_head_size)
		self.scaled_dot_attn = ScaledDotProductAttention(dropout, self.attn_head_size)
		self.linear = nn.Linear(self.all_head_size, self.d_hidden)
		self.dropout = nn.Dropout(dropout)

		# Store attention weights and scores for analysis
		self.agent_weights = None
		self.agent_scores = None

	def transpose_for_scores(self, x):
		"""Reshapes the input tensor to be compatible with multi-head attention."""
		new_x_shape = x.size()[:-1] + (self.n_head, self.attn_head_size)
		x = x.view(new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, Q, K, V, attn_mask):
		"""
		Forward pass for multi-agent attention.

		Args:
			Q (torch.Tensor): Query tensor (from each agent).
			K (torch.Tensor): Key tensor (from all agents).
			V (torch.Tensor): Value tensor (from all agents).
			attn_mask (torch.Tensor): Attention mask.

		Returns:
			tuple: A tuple containing:
				- output (torch.Tensor): The final attended output.
				- attn_prob (torch.Tensor): The attention probabilities.
		"""
		# Apply agent-specific query projections and concatenate
		Q_projected = torch.cat([self.W_Q[i](Q[:, i, :]).unsqueeze(dim=1) for i in range(self.n_agents)], dim=1)
		q_s = self.transpose_for_scores(Q_projected)
		
		# Project K and V
		k_s = self.transpose_for_scores(self.W_K(K))
		v_s = self.transpose_for_scores(self.W_V(V))

		attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

		context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
		
		self.agent_weights = attn_prob.mean(dim=1)
		self.agent_scores = self.scaled_dot_attn.scores

		context = context.permute(0, 2, 1, 3).contiguous()
		context = context.view(context.size()[:-2] + (self.all_head_size,))
		output = self.linear(context)
		output = self.dropout(output)
		
		return output, attn_prob

class ScaledDotProductAttention(nn.Module):
	"""Computes the scaled dot-product attention."""
	def __init__(self, dropout, attn_head_size):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.attn_head_size = attn_head_size
		self.scores = None
	
	def forward(self, Q, K, V, attn_mask):
		"""
		Args:
			Q (torch.Tensor): Queries. Shape (batch*heads, seq_len_q, head_size).
			K (torch.Tensor): Keys. Shape (batch*heads, seq_len_k, head_size).
			V (torch.Tensor): Values. Shape (batch*heads, seq_len_v, head_size).
			attn_mask (torch.Tensor): Mask to apply to the attention scores.

		Returns:
			tuple: A tuple containing the context vector and attention probabilities.
		"""
		# Compute raw attention scores
		scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
		
		# Apply the mask (e.g., for padding or causal attention)
		scores.masked_fill_(attn_mask == 0, -1e9)
		self.scores = scores * attn_mask
		
		# Convert scores to probabilities
		attn_prob = nn.Softmax(dim=-1)(scores)
		attn_prob = self.dropout(attn_prob)
		
		# Compute the weighted sum of values
		context = torch.matmul(attn_prob, V)
		
		return context, attn_prob

class PoswiseFeedForwardNet(nn.Module):
	"""A position-wise feed-forward network, a standard component of transformers."""
	def __init__(self, d_hidden, d_ff, dropout):
		super().__init__()
		self.dense1 = nn.Linear(d_hidden, d_ff)
		self.dense2 = nn.Linear(d_ff, d_hidden)
		self.active = F.gelu
		self.dropout = nn.Dropout(dropout)

	def forward(self, inputs):
		"""
		Args:
			inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_hidden).

		Returns:
			torch.Tensor: Output tensor of the same shape.
		"""
		output = self.dense1(inputs)
		output = self.active(output)
		output = self.dense2(output)
		output = self.dropout(output)
		return output
