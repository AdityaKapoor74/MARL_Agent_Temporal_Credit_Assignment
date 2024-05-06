# from former import util
from .util import mask_

import torch
from torch import nn
import torch.nn.functional as F

import random, math

def init(module, weight_init, bias_init, gain=1):
	if isinstance(module, nn.LayerNorm):
		init.ones_(module.weight)
		if module.bias is not None:
			init.zeros_(module.bias)
	elif isinstance(module, nn.Linear):
		# weight_init(module.weight.data, gain=gain)
		weight_init(module.weight.data)
		if module.bias is not None:
			bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	# return init(m, nn.init.kaiming_uniform_, lambda x: nn.init.constant_(x, 0), gain=gain)
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class SelfAttentionWide(nn.Module):
	def __init__(self, emb, heads=8, mask=False):
		"""

		:param emb:
		:param heads:
		:param mask:
		"""

		super().__init__()

		self.emb = emb
		self.heads = heads
		self.mask = mask
		self.dot = 0

		self.tokeys = init_(nn.Linear(emb, emb * heads, bias=False))
		self.toqueries = init_(nn.Linear(emb, emb * heads, bias=False))
		self.tovalues = init_(nn.Linear(emb, emb * heads, bias=False))

		self.softmax = nn.Softmax(dim=-1)

		self.unifyheads = init_(nn.Linear(heads * emb, emb))

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)

		self.attn_weights = None
		self.attn_scores = None

	def forward(self, x, masks=None, agent=False, temporal_only=False):

		b, t, e = x.size() # b, n_a, e
		h = self.heads
		assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

		keys    = self.tokeys(x)   .view(b, t, h, e)
		queries = self.toqueries(x).view(b, t, h, e)
		values  = self.tovalues(x) .view(b, t, h, e)

		# compute scaled dot-product self-attention

		# - fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
		queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
		values = values.transpose(1, 2).contiguous().view(b * h, t, e)

		queries = queries / (e ** (1/4))
		keys    = keys / (e ** (1/4))
		# - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
		#   This should be more memory efficient

		# - get dot product of queries and keys, and scale
		dot = torch.bmm(queries, keys.transpose(1, 2))

		self.attn_scores = dot

		if masks is not None:
			shape = dot.shape
			n_agents = masks.shape[-1]
			if agent:
				t_ = masks.shape[1]
				dot = (torch.where(masks.reshape(-1, t_, 1, 1, n_agents).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), self.mask_value)).reshape(*shape)
				dot = (torch.where(masks.reshape(-1, t_, 1, n_agents, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), self.mask_value)).reshape(*shape)
			elif temporal_only:
				t_ = masks.shape[-1]
				dot = (torch.where(masks.reshape(-1, 1, 1, t_).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), self.mask_value)).reshape(*shape)
				dot = (torch.where(masks.reshape(-1, 1, t_, 1).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), self.mask_value)).reshape(*shape)
			else:
				dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, 1, t).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), self.mask_value)).reshape(*shape)
				dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, t, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), self.mask_value)).reshape(*shape)

		assert dot.size() == (b*h, t, t)

		if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
			mask_(dot, maskval=float('-inf'), mask_diagonal=False)

		# dot = F.softmax(dot, dim=2)
		dot = self.softmax(dot) # bxh, t, t
		
		# if an agent is dead, the row corresponding to it will all have -1e9 thus softmax would give uniform attention weight to each which should ideally by 0.0
		shape = dot.shape
		n_agents = masks.shape[-1]
		if agent:
			t_ = masks.shape[1]
			dot = (torch.where(masks.reshape(-1, t_, 1, 1, n_agents).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), 0.0)).reshape(*shape)
			dot = (torch.where(masks.reshape(-1, t_, 1, n_agents, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), 0.0)).reshape(*shape)
		elif temporal_only:
			t_ = masks.shape[-1]
			dot = (torch.where(masks.reshape(-1, 1, 1, t_).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), 0.0)).reshape(*shape)
			dot = (torch.where(masks.reshape(-1, 1, t_, 1).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), 0.0)).reshape(*shape)
		else:
			dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, 1, t).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), 0.0)).reshape(*shape)
			dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, t, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), 0.0)).reshape(*shape)
		
		self.attn_weights = dot.reshape(-1, h, t, t).mean(dim=1).detach()

		# - dot now has row-wise self-attention probabilities
		# apply the self attention to the values
		out = torch.bmm(dot, values).view(b, h, t, e)

		# swap h, t back, unify heads
		out = out.transpose(1, 2).contiguous().view(b, t, h * e)

		return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):

	def __init__(self, emb, heads=8, mask=False):
		"""

		:param emb:
		:param heads:
		:param mask:
		"""

		super().__init__()

		assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

		self.emb = emb
		self.heads = heads
		self.mask = mask

		s = emb // heads
		# - We will break the embedding into `heads` chunks and feed each to a different attention head

		self.tokeys    = init_(nn.Linear(s, s, bias=False))
		self.toqueries = init_(nn.Linear(s, s, bias=False))
		self.tovalues  = init_(nn.Linear(s, s, bias=False))

		self.softmax = nn.Softmax(dim=-1)

		self.unifyheads = init_(nn.Linear(heads * s, emb))

		self.attn_weights = None
		self.attn_scores = None

	def forward(self, x, masks=None, agent=False, temporal_only=False):

		b, t, e = x.size()
		h = self.heads
		assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

		s = e // h
		x = x.view(b, t, h, s)

		keys    = self.tokeys(x)
		queries = self.toqueries(x)
		values  = self.tovalues(x)

		assert keys.size() == (b, t, h, s)
		assert queries.size() == (b, t, h, s)
		assert values.size() == (b, t, h, s)

		# Compute scaled dot-product self-attention

		# - fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
		queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
		values = values.transpose(1, 2).contiguous().view(b * h, t, s)

		queries = queries / (e ** (1/4))
		keys    = keys / (e ** (1/4))
		# - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
		#   This should be more memory efficient

		# - get dot product of queries and keys, and scale
		dot = torch.bmm(queries, keys.transpose(1, 2))

		self.attn_scores = dot

		if masks is not None:
			shape = dot.shape
			n_agents = masks.shape[-1]
			if agent:
				t_ = masks.shape[1]
				dot = (torch.where(masks.reshape(-1, t_, 1, 1, n_agents).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), -1e9)).reshape(*shape)
				dot = (torch.where(masks.reshape(-1, t_, 1, n_agents, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), -1e9)).reshape(*shape)
			elif temporal_only:
				t_ = masks.shape[-1]
				dot = (torch.where(masks.reshape(-1, 1, 1, t_).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), -1e9)).reshape(*shape)
				dot = (torch.where(masks.reshape(-1, 1, t_, 1).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), -1e9)).reshape(*shape)
			else:
				dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, 1, t).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), -1e9)).reshape(*shape)
				dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, t, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), -1e9)).reshape(*shape)

		assert dot.size() == (b*h, t, t)

		if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
			mask_(dot, maskval=float('-inf'), mask_diagonal=False)

		dot = self.softmax(dot)
		# if an agent is dead, the row corresponding to it will all have -1e9 thus softmax would give uniform attention weight to each which should ideally by 0.0
		shape = dot.shape
		n_agents = masks.shape[-1]
		if agent:
			t_ = masks.shape[1]
			dot = (torch.where(masks.reshape(-1, t_, 1, 1, n_agents).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), 0.0)).reshape(*shape)
			dot = (torch.where(masks.reshape(-1, t_, 1, n_agents, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, t_, h, n_agents, n_agents), 0.0)).reshape(*shape)
		elif temporal_only:
			t_ = masks.shape[-1]
			dot = (torch.where(masks.reshape(-1, 1, 1, t_).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), 0.0)).reshape(*shape)
			dot = (torch.where(masks.reshape(-1, 1, t_, 1).repeat(1, h, 1, 1).bool(), dot.reshape(-1, h, t_, t_), 0.0)).reshape(*shape)
		else:
			dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, 1, t).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), 0.0)).reshape(*shape)
			dot = (torch.where(masks.permute(0, 2, 1).reshape(-1, n_agents, 1, t, 1).repeat(1, 1, h, 1, 1).bool(), dot.reshape(-1, n_agents, h, t, t), 0.0)).reshape(*shape)

		self.attn_weights = dot.reshape(-1, h, t, t).mean(dim=1).detach()
		# - dot now has row-wise self-attention probabilities

		# apply the self attention to the values
		out = torch.bmm(dot, values).view(b, h, t, s)

		# swap h, t back, unify heads
		out = out.transpose(1, 2).contiguous().view(b, t, s * h)

		return self.unifyheads(out)

class TransformerBlock(nn.Module):

	def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True):
		super().__init__()

		self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

		self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
					else SelfAttentionNarrow(emb, heads=heads, mask=mask)
		self.mask = mask

		self.norm1 = nn.LayerNorm(emb)
		self.norm2 = nn.LayerNorm(emb)

		self.ff = nn.Sequential(
			init_(nn.Linear(emb, ff_hidden_mult * emb), activate=True),
			nn.GELU(),
			init_(nn.Linear(ff_hidden_mult * emb, emb), activate=True)
		)

		self.do = nn.Dropout(dropout)

	def forward(self, x, masks=None, temporal_only=False):
		b, t, e = x.shape
		x = x + self.pos_embedding(torch.arange(t, device=x.device))[None, :, :].expand(b, t, e)

		attended = self.attention(x, masks, temporal_only=temporal_only)

		x = self.norm1(attended + x)

		x = self.do(x)

		fedforward = self.ff(x)

		x = self.norm2(fedforward + x)

		x = self.do(x)

		return x



class TransformerBlock_Agent(nn.Module):

	def __init__(self, emb, heads, mask, seq_length, n_agents, ff_hidden_mult=4, dropout=0.0, wide=True):
		super().__init__()

		self.n_a = n_agents

		self.agent_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=self.n_a)

		self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
					else SelfAttentionNarrow(emb, heads=heads, mask=mask)
		self.mask = mask

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

		x = x.view(-1, self.n_a, t, e).transpose(1, 2).contiguous().view(-1, self.n_a, e)
		x = x + self.agent_embedding(torch.arange(self.n_a, device=x.device))[None, :, :].expand(x.shape[0], self.n_a, e)

		attended = self.attention(x, masks, agent=True)

		x = self.norm1(attended + x)

		x = self.do(x)

		fedforward = self.ff(x)

		x = self.norm2(fedforward + x)

		x = self.do(x)

		x = x.view(-1, t, self.n_a, e).transpose(1, 2).contiguous().view(-1, t, e)

		return x