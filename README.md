# MPhil-Thesis-Learning-Non-Parametric-Structural-Decompositions-in-Gaussian-Processes
GitHub repository for my MPhil thesis, "Learning Non-Parametric Structural Decompositions in Gaussian Processes", based on the codebase the paper "Additive Gaussian Processes Revisited" (Lu et al. (2022)): https://github.com/amzn/orthogonal-additive-gaussian-processes

This code serves as visualisations for understanding the unconstrained additive GP and OAK models, using the codebase from the paper "Additive Gaussian Processes Revisited" (Lu et al. (2022)): https://github.com/amzn/orthogonal-additive-gaussian-processes.

## Contents
The code for generating the models and visualisations used throughout my thesis, categorised by relevant chapter and order of appearance.

### Chapter 2:
- agp_grid.ipynb: Generate a synthetic two-dimensional dataset and visualise its component breakdowns when modelled using the unconstrained additive GP.
- oak_grid.ipynb: Generate a synthetic two-dimensional dataset and visualise its component breakdowns when modelled using the OAK model.

### Chapter 3:
- agp_models_complexity.ipynb: Visualise the prior and posterior samples from a one-dimensional unconstrained additive GP model with different hyperparameters.
- agp_models_1D_decomposition.ipynb: Comparison between the unconstrained additive GP and OAK decompositions in one dimension.
- first_order_signal_variance.ipynb: Verfiying the first-order signal variance learned by the unconstrained model by fitting a squared-exponential GP with lengthscale equal to the value chosen by the model for both first-order components.
- interaction_signal_variance.ipynb: Verfiying the second-order signal variance learned by the unconstrained model by fitting a squared-exponential GP with lengthscales equal to the values chosen by the model.
- leakage_visualisation.ipynb: Decomposing all leakage occuring within a two dimensional unconstrained additive GP model.
- covariance_decomposition.ipynb: Decompose a synthetic two-dimensional dataset's variance into the sum of component variances and covariance.
- learn_base_kernel_magnitudes.ipynb: Generate a synthetic two-dimensional dataset and visualise its component breakdowns when modelled using the unconstrained additive GP with learnable base kernel magnitude hyperparameters.
- agp_models_complexity_2D.ipynb: Visualise the prior and posterior samples from the unconstrained additive GP and OAK models modelling a two-dimensional dataset.

### Chapter 4:
- posterior_mean_mismatch.ipynb: Learn a synthetic one-dimensional dataset using OAK models with different Gaussian measures, and visualise the component differences.
- posterior_variance_inflation.ipynb: Learn a synthetic one-dimensional dataset using OAK models with different Gaussian measures which show the potential for variance inflation.
- measure_mismatch_grid.ipynb: Example plots from different synthetic one-dimensional datasets showing the different impacts a mismatched measure can have.
- sobol_indices.ipynb: Quantify the impact of mismatched measures using Sobol indices.
- sobol_indices_grid.ipynb: Component breakdown illustrating the reason for potential unreliable Sobol indices caused by mismatched measures.
- mismatched_measures_complexity.ipynb: Visualise the prior and posterior samples from the OAK model with different orthogonality measures.
- kl_divergence.ipynb: Visualisation of the unreliability of normalising flow quantified using the KL-divergence between distributions and the standard Gaussian before and after normalising flow.
- real_world_results.ipynb: Train real-world datasets usign various measure-alignment techniques and compare them in terms of predictive performance, interpretability, and parsimony.

### Appendices:
- identifiability_problem_data_impact.ipynb: Quantify the correlation between properties of the training data and the first-order posterior mean offset.
- learn_base_kernel_magnitudes_nmll.ipynb: Generate a synthetic one-dimensional dataset and visualise its component breakdowns to justify the improved NMLL potential when using learnable base kernel variance hyperparameters.
- normalising_flow_real_world.ipynb: Visualise the unreliability of normalising flow on real-world datasets using KS tests, histograms and Q-Q plots.

### Other:
- imports.py: Helper file containing all relevant Python libraries, and functions from the codebase needed to run my code.
- funcs/helper_functions.py: Common helper functions required in several of the interaction notebooks.
