<img align="left" width="100" height="75" src="https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/microsoft-azure-640x401.png">

# Optimizing an ML Pipeline in Azure



## Overview

This project is part of the Udacity Azure ML Nanodegree.

In this project, first we build and optimize an Azure ML pipeline using the Python SDK and a provided [Scikit-learn Logistic Regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). 
The hyperparameters of this model are optimized using HyperDrive. After that, a second model is built and optimized using Azure Auto ML on the on the same dataset.

Then, the results obtained by both models are compared.

A diagram illustrating the steps of this project is shown below:

![](https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/project_summary.JPG)
source: Nanodegree Program Machine Learning Engineer with Microsoft Azure

## Summary

### Dataset and Problem Statement

The dataset used in this project contains data collected during direct marketing campaigns (phone calls) of a Portuguese banking institution. 
This is a subset of the original public dataset available at [UCI Machine Learning repository]( https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 
In this website a detailed description of each feature can be found.

It consists of 32950 entries, 20 features containing information about `client`, information relative to the `marketing campaign`, and `social and economic` metrics. 

This is a classification problem wich goal is to predict if the client will subscribe (yes/no) to a bank term deposit (variable `y`).

**original source of the data**: 
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, 
June 2014 (https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf )

### Solution and Result

For this classification problem to approaches were used:

1. Apply a Scikit-learn Logistic Regression model, optimizing its hyperparameters using HyperDrive.
2. Use Azure Auto ML to build and optimize a model on the same dataset

As a result, the best model, considering `accuracy` as primary metric was the one obtained using AutoML. It was a Voting Ensemble model. 
However, the accuracy obtained both applying Logistic Regression with HyperDrive and AutoML were close, respectively, 0.9109 and 0.9174.
The main advantage of the AutoML is explainability, so we choose further which features have more weight on the predictions. In addition, 
we can observe the value of other metrics that in the case of an imbalanced dataset as this one can say more than accuracy.


## Scikit-learn Pipeline

### Summary of the Pipeline

<img align="center" width="800" height="600" src="https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/HyperDrive_pipeline.JPG">


An overview of the Scikit-learn/HyperDrive experiment is illustrate in the image above.

The script `train.py` included:

1. Loading dataset from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv 
2. Cleaning and transforming data (e.g. drop NaN values, one hot encode, and encode from string to number using dictionary). 
3. Calling the SKlearn Logistic Regression model using parameters (`C` (float): Inverse of regularization strength. float. Smaller values, stronger regularization) and 
`max_iter`(int): Maximum number of iterations taken for the solvers to converge)

Roughly, the following steps were taken in the notebook:

1. Initialize our `Workspace`
2. Create an `Experiment`
3. Define resources, i.e., create `AmlCompute` as training compute resource

Specify a compute configuration means defining the `type of machine` to be used and the `scalability behaviors`. Also, it is necessary to define the name of the cluster 
which must be unique within the workspace. This name is used to address the cluster later.

For this project we use a CPU cluster with following parameters:

* `type of the machine`:
    * `vm_size`: Defines the size of the virtual machine. We use here "STANDARD_D2_V2" (more details [here](https://docs.microsoft.com/en-us/azure/cloud-services/cloud-services-sizes-specs#dv2-series))

* `Scalability behaviors`:
    * `min_nodes`: Sets minimum size of the cluster. Setting the minimum to 0 the cluster will shut down all nodes while not in use. If you use another value you are able to have faster start-up times, but you will also be billed when the cluster is not in use.
    * `max_nodes`: Sets the maximum size of the cluster. Larger number allows for more concurrency and a greater distributed processing of scale-out jobs.

4. `Hyper parameter tunning` which means defining parameters to be used by HyperDrive. Part of it involves specifying a parameter sampler, a policy for early termination, and creating an estimator for the `train.py` script.
5. Submit the `HyperDriveConfig` to run the experiment using parameters defined in the previous step.
6. Use method ` get_best_run_by_primary_metric()` on the run to select the best hyperparameters for the Sklearn Logistic Regression model
7. Save the best model.
### What are the benefits of the parameter sampler chosen?
In the `random sampling algorithm used` in this project, parameter values are chosen from a set of discrete values or a distribution over a continuous range.
Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space.
The other two available techniques (Grid Sampling and Bayesian) are indicated if you have a budget to exhaustively search over the search space. In addition, Bayesian does not allow using early termination.
With Random sampling we can always do an initial search with random sampling and then refine the search space to improve results
### What are the benefits of the early stopping policy you chosen?
In general, an `early stopping policy` automatically terminate poorly performing runs which improves computational efficiency.
The `early termination policy` we used [`Bandit Policy`]( https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?preserve-view=true&view=azure-ml-py#&preserve-view=truedefinition ). This policy is based on `slack factor/slack amount` and `evaluation interval`. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

This allows more aggressive savings than Median Stopping policy if we apply a smaller allowable slack.

Parameter `slack_factor` which is the slack allowed with respect to the best performing training run, need to be defined while `evaluation_interval` and `delay_interval` are optional.

`evaluation_interval` says when the policy is applied. If the `evaluation_interval` is not defined the default value is one, i.e., policy is applied every time the training script reports the primary metric.

Specifying `delay_interval` avoids premature termination of training runs by allowing all configurations to run for a minimum number of intervals. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
