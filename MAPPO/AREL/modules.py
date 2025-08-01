"""
This file defines the core building blocks for the transformer-based architecture used in the AREL model.

It includes standard transformer components like multi-head self-attention and feed-forward networks,
implemented in both "wide" and "narrow" configurations. It also defines a specialized
TransformerBlock_Agent for handling inter-agent attention.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .util import mask_
import random, math

def init(module, weight_init, bias_init, gain=1):
	"""
	Applies a specified weight and bias initialization to a module.

	Args:
		module (nn.Module): The module to initialize.
		weight_init: The function to use for weight initialization.
		bias_init: The function to use for bias initialization.
		gain (float): A scaling factor for the weight initialization.
	"""
	if isinstance(module, nn.LayerNorm):
		nn.init.ones_(module.weight)
		if module.bias is not None:
			nn.init.zeros_(module.bias)
	elif isinstance(module, nn.Linear):
		weight_init(module.weight.data, gain=gain)
		if module.bias is not None:
			bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	"""
	A helper function for orthogonal initialization of linear layers.

	Args:
		m (nn.Module): The module to initialize.
		gain (float): The gain for the orthogonal initialization.
		activate (bool): If True, calculates gain based on ReLU activation.
	"""
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class SelfAttentionWide(nn.Module):
	"""
	Multi-head self-attention where the embedding dimension is maintained per head.

	In this "wide" implementation, each head operates on the full embedding dimension,
	and the projections are concatenated before the final linear layer.
	"""
	def __init__(self, emb, heads=8, mask=False):
		super().__init__()
		self.emb = emb
		self.heads = heads
		self.mask = mask

		self.tokeys = init_(nn.Linear(emb, emb * heads, bias=False))
		self.toqueries = init_(nn.Linear(emb, emb * heads, bias=False))
		self.tovalues = init_(nn.Linear(emb, emb * heads, bias=False))
		self.unifyheads = init_(nn.Linear(heads * emb, emb))

		self.mask_value = torch.tensor(torch.finfo(torch.float).min, dtype=torch.float)
		self.attn_weights = None
		self.attn_scores = None

	def forward(self, x, masks=None, agent=False, temporal_only=False):
		b, t, e = x.size()
		h = self.heads
		assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

		# Project inputs to queries, keys, and values for each head
		keys = self.tokeys(x).view(b, t, h, e)
		queries = self.toqueries(x).view(b, t, h, e)
		values = self.tovalues(x).view(b, t, h, e)

		# Fold heads into the batch dimension for efficient computation
		keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
		queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
		values = values.transpose(1, 2).contiguous().view(b * h, t, e)

		# Scale queries and keys for stabilized dot-product
		queries = queries / (e ** (1/4))
		keys = keys / (e ** (1/4))

		# Compute raw attention scores
		dot = torch.bmm(queries, keys.transpose(1, 2))
		self.attn_scores = dot

		# Apply masks if provided (e.g., for padding or causal attention)
		if masks is not None:
			dot.masked_fill_(~masks.bool(), self.mask_value.to(x.device))
		
		if self.mask:  # Apply causal mask for temporal attention
			mask_(dot, maskval=float('-inf'), mask_diagonal=False)

		# Convert scores to probabilities and apply zero-out for masked elements
		dot = F.softmax(dot, dim=-1)
		if masks is not None:
			dot.masked_fill_(~masks.bool(), 0.0)
		
		self.attn_weights = dot.reshape(-1, h, t, t).mean(dim=1).detach()

		# Compute weighted sum of values
		out = torch.bmm(dot, values).view(b, h, t, e)

		# Unify heads
		out = out.transpose(1, 2).contiguous().view(b, t, h * e)
		return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):
	"""
	Multi-head self-attention where the embedding dimension is split across heads.

	In this "narrow" implementation, the embedding dimension is divided by the number
	of heads, and each head operates on a smaller chunk of the embedding.
	"""
	def __init__(self, emb, heads=8, mask=False):
		super().__init__()
		assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'
		self.emb = emb
		self.heads = heads
		self.mask = mask
		s = emb // heads

		self.tokeys = init_(nn.Linear(s, s, bias=False))
		self.toqueries = init_(nn.Linear(s, s, bias=False))
		self.tovalues = init_(nn.Linear(s, s, bias=False))
		self.unifyheads = init_(nn.Linear(heads * s, emb))

		self.attn_weights = None
		self.attn_scores = None

	def forward(self, x, masks=None, agent=False, temporal_only=False):
		b, t, e = x.size()
		h = self.heads
		assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
		s = e // h

		# Reshape input to split embedding across heads
		x = x.view(b, t, h, s)
		keys = self.tokeys(x)
		queries = self.toqueries(x)
		values = self.tovalues(x)

		# Fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
		queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
		values = values.transpose(1, 2).contiguous().view(b * h, t, s)

		queries = queries / (e ** (1/4))
		keys = keys / (e ** (1/4))

		dot = torch.bmm(queries, keys.transpose(1, 2))
		self.attn_scores = dot

		if masks is not None:
			dot.masked_fill_(~masks.bool(), -1e9)

		if self.mask:
			mask_(dot, maskval=float('-inf'), mask_diagonal=False)

		dot = F.softmax(dot, dim=-1)
		if masks is not None:
			dot.masked_fill_(~masks.bool(), 0.0)
			
		self.attn_weights = dot.reshape(-1, h, t, t).mean(dim=1).detach()

		out = torch.bmm(dot, values).view(b, h, t, s)
		out = out.transpose(1, 2).contiguous().view(b, t, s * h)
		return self.unifyheads(out)

class TransformerBlock(nn.Module):
	"""
	A standard transformer block for temporal processing.

	Combines a self-attention layer with a feed-forward network, using residual
	connections and layer normalization.
	"""
	def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True):
		super().__init__()
		self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide else SelfAttentionNarrow(emb, heads=heads, mask=mask)
		self.norm1 = nn.LayerNorm(emb)
		self.norm2 = nn.LayerNorm(emb)
		self.ff = nn.Sequential(
			init_(nn.Linear(emb, ff_hidden_mult * emb), activate=True),
			nn.GELU(),
			init_(nn.Linear(ff_hidden_mult * emb, emb), activate=True)
		)
		self.do = nn.Dropout(dropout)

	def forward(self, x, masks=None, temporal_only=False):
		attended = self.attention(x, masks, temporal_only=temporal_only)
		x = self.norm1(attended + x)
		x = self.do(x)
		fedforward = self.ff(x)
		x = self.norm2(fedforward + x)
		x = self.do(x)
		return x

class TransformerBlock_Agent(nn.Module):
	"""
	A transformer block adapted for agent-axis attention.

	This block reshapes the input tensor to perform attention across the agent
	dimension for each timestep.
	"""
	def __init__(self, emb, heads, mask, seq_length, n_agents, ff_hidden_mult=4, dropout=0.0, wide=True):
		super().__init__()
		self.n_a = n_agents
		self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide else SelfAttentionNarrow(emb, heads=heads, mask=mask)
		self.norm1 = nn.LayerNorm(emb)
		self.norm2 = nn.LayerNorm(emb)
		self.ff = nn.Sequential(
			init_(nn.Linear(emb, ff_hidden_mult * emb), activate=True),
			nn.GELU(),
			init_(nn.Linear(ff_hidden_mult * emb, emb), activate=True)
		)
		self.do = nn.Dropout(dropout)

	def forward(self, x, masks=None):
		_, t, e = x.size()
		# Reshape to (batch*seq_len, n_agents, emb_dim) to perform attention over agents
		x = x.view(-1, self.n_a, t, e).transpose(1, 2).contiguous().view(-1, self.n_a, e)
		
		attended = self.attention(x, masks, agent=True)
		x = self.norm1(attended + x)
		x = self.do(x)
		fedforward = self.ff(x)
		x = self.norm2(fedforward + x)
		x = self.do(x)
		
		# Reshape back to original format
		x = x.view(-1, t, self.n_a, e).transpose(1, 2).contiguous().view(-1, t, e)
		return x
