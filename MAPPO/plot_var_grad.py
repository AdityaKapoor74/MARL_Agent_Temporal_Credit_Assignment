import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = '../../../tests/TEAM COLLISION AVOIDANCE/eval_policy'

prd = np.load(os.path.join(data_dir, "prd_above_threshold_ascend.npy"))
# shared = np.load(os.path.join(data_dir, "shared.npy"))

print(prd.shape)


legend_entries =[
'PRD-MAPPO',
'MAPPO',
]

colors = [
'red',
'blue',
]

alphas = [1, 1]

trim_inds = [None, None]

xlabel = 'Episodes'
ylabel = 'Variance'
title = 'Team Collision Avoidance, Grad Var vs. Episodes (Lower is better)'
filename = 'Team_Collision_Avoidance_Grad_Var'
legend_loc = 'lower right'

def trim(list_of_arrays,trim_ind=None):
	lengths = []
	for a in list_of_arrays:
		lengths.append(a.shape[0])
	
	min_length = np.min(lengths)
	trimmed = []
	for a in list_of_arrays:
		if trim_ind is None:
			trimmed.append(a[:min_length])
		else:
			trimmed.append(a[:trim_ind])

	return trimmed


# def plot_with_var(data_mat, legend, color, alpha):

# 	means = np.mean(data_mat,axis=1)
# 	stds  = np.std(data_mat,axis=1)

# 	ucb = means + stds
# 	lcb = means - stds
	
# 	x = np.arange(data_mat.shape[0]) * 1000

# 	plt.plot(x, means, label=legend, color=color, alpha=alpha)
# 	plt.fill_between(x, lcb, ucb, color=color, alpha=0.1)

def plot_var(data_mat, legend, color, alpha):

	# means = np.mean(data_mat,axis=1)
	# stds  = np.std(data_mat,axis=1)
	var = np.var(data_mat, axis=-2)
	print(var.shape)
	stds  = np.std(var,axis=-1)

	ucb = var + stds
	lcb = var - stds
	
	x = np.arange(data_mat.shape[0]) * 1000

	plt.plot(x, var, label=legend, color=color, alpha=alpha)
	plt.fill_between(x, lcb, ucb, color=color, alpha=0.1)


plt.figure()

plot_var(prd, legend_entries[0], colors[0], alphas[0])
# plot_var(shared, legend_entries[1], colors[1], alphas[1])

plt.legend(loc=legend_loc)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title,fontdict={'fontsize':12,'fontweight':'bold'})
# plt.xlim(0, 30000)
# plt.ylim(-800, 0)
# plt.yscale('log')
plt.tight_layout()

plt.savefig(filename+".pdf", bbox_inches='tight')