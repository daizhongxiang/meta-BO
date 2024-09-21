# Paper title: On Provably Robust Meta-Bayesian Optimization

This directory contains the code for the experiment: Hyperparameter Tuning for Convolutional Neural Networks (CNNs), in the paper "On Provably Robust Meta-Bayesian Optimization" submitted to UAI 2022. The implemented alogrithms here include (1) GP-UCB, (2) RM-GP-UCB, (3) RM-GP-TS, (4) RGPE, and (5) TAF. For multitask BO (MTBO), we used the implementation in the RoBO package: https://github.com/automl/RoBO (after fixing some compatibility issues).

## Requirements

Key dependencies (excluding commonly used packages such as scipy, numpy, tensorflow, keras, etc.)
(1) GPy
(2) scipydirect: this package uses the DIRECT method to optimize the acquisition function
    (a) install scipydirect: pip install scipydirect
    (b) replace the content of the script "PYTHON_PATH/lib/python3.5/site-packages/scipydirect/__init__.py" with the content of the script "scipydirect_for_rm_gp_ucb.py" in the "dependencies" folder; this step is required since we modified the interface of the scipydirect minimize function


## Training and Evaluation

Instructions to generate the meta-tasks:
(1) generate the meta-observations:
    (a) call "run_generate_meta_obs_mnist.py":
    (b) call "run_generate_meta_obs_svhn.py":
    (c) call "run_generate_meta_obs_cifar_10.py":
    (d) call "run_generate_meta_obs_cifar_100.py":
(2) format the meta-observations to be used for meta-BO by calling "format_meta_tasks.py"

Instructions to run different BO/meta-BO algorithms:
(1) GP-UCB: "run_no_meta_mnist.py", "run_no_meta_svhn.py", "run_no_meta_cifar_10.py", "run_no_meta_cifar_100.py"
(2) RM-GP-UCB: "run_meta_mnist.py", "run_meta_svhn.py", "run_meta_cifar_10.py", "run_meta_cifar_100.py"
(2) RM-GP-TS: "run_meta_mnist_ts.py", "run_meta_svhn_ts.py", "run_meta_cifar_10_ts.py", "run_meta_cifar_100_ts.py"
(3) RGPE: "run_meta_mnist_rgpe.py", "run_meta_svhn_rgpe.py", "run_meta_cifar_10_rgpe.py", "run_meta_cifar_100_rgpe.py"
(4) TAF: "run_meta_mnist_taf.py", "run_meta_svhn_taf.py", "run_meta_cifar_10_taf.py", "run_meta_cifar_100_taf.py"


## Results

Instructions to analyze the results:
(1) run "analyze_results.ipynb" to analyze & visualize the results

Description of the directories:
(1) "meta_tasks": contains the generated meta-tasks and meta-observations
(2) "initializations": contains the initializations used by all methods (to facilitate fair comparisons)
(3) "results": is the place where all results are saved
(4) "dependencies": contains the dependencies required by the scripts
