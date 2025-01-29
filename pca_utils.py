import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA

from typing import Optional

def prep_for_pca(obj, increased_neurons = None, decreased_neurons = None, all_neurons = False, increased: Optional[bool] = None):
	colors = np.full(obj.formatted_data_array['firing_rate']['baseline'].shape[0], 'gray', dtype=object)
	if all_neurons:
		neurons_of_interest = np.arange(0, obj.formatted_data_array['firing_rate']['baseline'].shape[0])
		neurons_to_return = [neurons_of_interest]
	
	else:
		if increased is None:
			neurons_of_interest = increased_neurons + decreased_neurons
			neurons_to_return = [increased_neurons, decreased_neurons]
			colors[increased_neurons] = 'red'
			colors[decreased_neurons] = 'blue'
		elif increased:
			neurons_of_interest = increased_neurons
			neurons_to_return = [increased_neurons]
		else:
			neurons_of_interest = decreased_neurons
			neurons_to_return = [decreased_neurons]
		neurons_of_interest = np.sort(np.array(neurons_of_interest))
  
	baseline_data = obj.formatted_data_array['firing_rate']['baseline'][neurons_of_interest]  # Neurons x Trials x Time
	stimulus_data = obj.formatted_data_array['firing_rate']['stimulus'][neurons_of_interest] # Neurons x Trials x Time

	combined_data = np.concatenate([baseline_data, stimulus_data], axis=2)  # Neurons x Trials x (Baseline_Time + Stimulus_Time)

	# Average firing rates across trials
	trial_avaraged = np.mean(combined_data, axis=1)  # Neurons x Time
	time_avaraged = np.mean(combined_data, axis=2)  # Neurons x Time
	return neurons_to_return, combined_data, trial_avaraged, time_avaraged, colors[neurons_of_interest]

def plot_pca_explained_variance(pca):
	fig, axs = plt.subplots(1, 2, figsize=(12, 6))
	axs[0].plot(np.arange(pca.explained_variance_ratio_.shape[0]) + 1, pca.explained_variance_ratio_, marker='o')
	axs[0].set_xlabel('Principal Component')
	axs[0].set_ylabel('Explained Variance Ratio')
	axs[0].set_title('Explained Variance Ratio of Principal Components', fontsize=10)
	axs[1].plot(np.arange(pca.explained_variance_ratio_.shape[0]) + 1, np.cumsum(pca.explained_variance_ratio_),marker='o')
	axs[1].set_xlabel('Principal Component')
	axs[1].set_ylabel('Cumulative Explained Variance Ratio')
	axs[1].set_title('Cumulative Explained Variance Ratio of Principal Components', fontsize=10)
	plt.suptitle('Explained Variance - PCA Analysis of All Neurons')
	plt.tight_layout()
	plt.show()

def plot_pca_projections(obj,pca_projection, colors_neurons = [], colors = None, axis_pc = [[0,1], [0,2], [1,2]]):
	# Create a color array
	legend = False
	if colors is None:
		colors = np.full(len(pca_projection), 'gray', dtype=object)  # Default gray
		if len(colors_neurons) > 0:
			colors = np.full(len(pca_projection), 'gray', dtype=object)  # Default gray
			colors[colors_neurons[0]] = 'red'   # Red for increased neurons
			colors[colors_neurons[1]] = 'blue'  # Blue for decreased neurons
			# Create legend handles
			red_patch = mpatches.Patch(color='red', label='Increase')
			blue_patch = mpatches.Patch(color='blue', label='Decrease')
			gray_patch = mpatches.Patch(color='gray', label='No Change')
			legend = True
		else:
			# Choose a gradient property (e.g., PC1 values)
			colors = np.concatenate([obj.baseline_times,obj.stimulus_times])   # Use PC1 for color gradient 	
	else:
		# Create legend handles
		red_patch = mpatches.Patch(color='red', label='Increase')
		blue_patch = mpatches.Patch(color='blue', label='Decrease')
		gray_patch = mpatches.Patch(color='gray', label='No Change')
		legend = True
	# Scatter plot
	fig, axs = plt.subplots(1, 3, figsize=(10, 3))
	for i in range(3):
		ax = axs[i]
		scatter_0 = ax.scatter(pca_projection[:, axis_pc[i][0]], pca_projection[:, axis_pc[i][1]], s=10, c=colors, cmap='coolwarm')
		ax.set_xlabel(f"PC{axis_pc[i][0]+1}")
		ax.set_ylabel(f"PC{axis_pc[i][1]+1}")
		if not legend:
			fig.colorbar(scatter_0, ax=ax)

	plt.suptitle("Neurons in PCA Space")
	if legend:
		plt.legend(handles=[red_patch, blue_patch, gray_patch], loc='upper right', bbox_to_anchor=(1.7, 1))
	plt.tight_layout()
	plt.show()
 
def plot_pca_loadings(pca, pca_comp = [0,1,2], neurons = True):
	time_bins = np.arange(pca.components_[0].shape[0])
 	# Plot the loadings for PC1 and PC2
	plt.figure(figsize=(10, 5))
	if neurons:
		plt.xlabel("Neuron ID")
	else:
		plt.xlabel("Time")
		time_bins = time_bins - 50
  
	for i in range(len(pca_comp)):
		pc_loadings = pca.components_[pca_comp[i]]  
		plt.plot(time_bins, pc_loadings, label=f"PC{pca_comp[i]+1} Loadings", alpha=0.7)

	# Highlight significant contributions (optional)
	plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
	plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

	# Labels and legend
	plt.ylabel("Loading Magnitude")
	plt.title("PCA Loadings")
	plt.legend(loc = 'upper right', bbox_to_anchor=(1.2, 1))

	plt.show()

def apply_pca(obj, increased_neurons, decreased_neurons,
              n_components = 10, all_neurons = True, increased = True,
              axis_pc = [[0,1],[0,2],[1,2]],
              loading_pc = [0,1,2]):
	neurons_of_interest, _, trial_avaraged, time_avaraged, colors = prep_for_pca(obj, increased_neurons= increased_neurons,
                                                                              decreased_neurons = decreased_neurons,
                                                                              all_neurons = all_neurons, increased = increased)
	
	if all_neurons is True:
		colors = None
	pca_fr = PCA(n_components=n_components)
	pca_fr_projection = pca_fr.fit_transform(trial_avaraged)
	plot_pca_explained_variance(pca_fr)
	plot_pca_projections(obj,pca_fr_projection, colors = colors, colors_neurons = [increased_neurons, decreased_neurons],
                      axis_pc=axis_pc)
	plot_pca_loadings(pca_fr, neurons = False, pca_comp = loading_pc)
 
	pca_neuron = PCA(n_components=n_components)
	pca_neuron_projection = pca_neuron.fit_transform(trial_avaraged.T)
	plot_pca_explained_variance(pca_neuron)
	plot_pca_projections(obj,pca_neuron_projection, 
                      axis_pc=axis_pc)
	plot_pca_loadings(pca_neuron, pca_comp = loading_pc)
	return pca_fr, pca_neuron