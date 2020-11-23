<img align="left" width="100" height="75" src="https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/microsoft-azure-640x401.png">

# Optimizing an ML Pipeline in Azure



## Overview

This project is part of the Udacity Azure ML Nanodegree.

In this project, first we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn Logistic Regression model. 
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
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014 (https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf )

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
