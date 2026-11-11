# Project Assignment — Machine Learning 2026

*Informatics Engineering Department*

## 1. Background

In this assignment, you are required to apply the analytical and machine learning methods learned throughout the course to the synthetic data in the **UCI AI4I 2020 Predictive Maintenance Dataset**:
<https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>

In the dataset, consider a factory that wants to identify machine failures from operating measurements. Missed failures and false alarms have different consequences: a missed failure may leave a problem undetected, while a false alarm triggers an unnecessary inspection.

The main goal of this project is to determine if a model can detect machine failures without generating too many false alarms. This is a classification dataset using synthetic observations.

## 2. Dataset Description (from UCI)

The dataset consists of **10 000 data points** stored as rows with seven input features:

1. **UID**: unique identifier ranging from 1 to 10000
2. **product ID**: consisting of a letter L, M, or H
3. **air temperature** [K]
4. **process temperature** [K]
5. **rotational speed** [rpm]
6. **torque** [Nm]
7. **tool wear** [min]

There is a generic failure indication label, **`machine failure`**, which indicates whether the machine has failed in this particular data point because any of the following failure modes is true.

There are five specific failure labels:

1. tool wear failure (**TWF**)
2. heat dissipation failure (**HDF**)
3. power failure (**PWF**)
4. overstrain failure (**OSF**)
5. random failures (**RNF**)

## 3. Objectives

The objective of this study is to address two related classification tasks:

1. **Binary Classification:** Develop a model to distinguish between failure and non-failure (label `machine failure`).
2. **Multiclass Classification:** Construct a model to predict the exact failure of a data point. Here you can choose and justify your approach, i.e., you can choose the types of failures you want to predict.

In this project, you must use **at least two different machine learning methods**:

- **Neural Networks (NN) are mandatory**, under severe grade penalty if not used.
- The second method is up to you to choose, **except for Naïve Bayes, Logistic Regression, KNN, and simple Decision Trees** used in the Challenge. Justify the choice.

The overall goal is to evaluate and compare the performance of different classification approaches in both the binary and multiclass settings, thereby gaining insights into the factors that influence machine failure.

## 4. Practical Assignment

### 4.1 Data import and analysis

Develop scripts for feature data import. Organise data into subsets relating to each source type you intend to test, e.g., create training, validation, and testing sets. If present, remove or find ways to handle missing data (for example, using the mean or the median of a certain feature), check for duplicates, inconsistencies, etc. In sum, do a general data quality check before proceeding with the next steps.

You need to understand the data you are working with. To that end, you should explore it. Use different types of data visualisation tools (histograms, pie charts, box plots, correlation plots) and analyse the data. Consider using feature selection techniques and see how they affect the performance of the machine learning algorithms. Notice the class imbalance and the different features' domains in this dataset and find strategies to cope. Make sure you know your features. Do any preprocessing, data encoding or normalisation that you deem necessary, always justifying your decisions. Do not forget to include in the final report all your findings, challenges, and steps taken.

**Mandatory guidelines for preparing the data:**

- Use a **75/25** train/test split
- Use **`random_state = 42`**
- Use **stratification** on the target class

### 4.2 Experimental analysis and evaluation

You should be able to design experiments to run the machine learning algorithms on the given data and evaluate their results. Define the appropriate performance metrics and justify your choices.

- **Run the experiments multiple times.** Present average results and standard deviations of the metrics used. You may also decide to cross-validate.
- In the end, choose the best classifier and evaluate it on a testing set (hold-out).
- Manually inspecting the predictions of your algorithms can give you precious insights into where they might be failing (and why) and what you can do to improve them (e.g., what makes the algorithm fail in this particular case? What special characteristic does it have that makes it so hard? How can I make the algorithm better deal with those cases?).
- Go back and forth on the hyperparameter choices of the different machine learning algorithms until you are satisfied with the results. Keep track of the evolution of the performance of your algorithm during this process, and show these trends in your final report to justify all the involved issues (choosing parameters, model fit, etc.).
- Try to understand how the methods perform differently on your data.

**Mandatory reporting:**

- For every required model, report **accuracy, precision, recall and F1 for class 1** on **validation and test** data.
- If a metric is undefined, use **0** as the convention.
- It is strongly suggested to present **test confusion matrices as counts**, with **actual classes in rows** and **predicted classes in columns**.
- Explain the change in **missed failures** and **false alarms**.

### 4.3 Results and Discussion

Present and discuss the final results obtained in your project assignment.

- Other authors have already studied this type of problem and related datasets. **Compare your results with results from other sources**, and do not copy from them.
- State your **final recommendation** and **one limitation**.
- Refer to your results and discuss any **disagreement between validation and test**.
- **Do not change your selections based on test results.**

### 4.5 Libraries and Language

Your code must be written in **Python**. You are free to use Colab, Jupyter or a script. For Neural Networks, PyTorch, Keras, or Scikit-learn (for more basic architectures) are recommended. In any case, you may use any Python-based framework or library.

## 5. Submission and Assessment

| Item | Date |
| --- | --- |
| Submission deadline | **17 November 2026, 23:59** |
| Individual written defence | **18 November 2026** (morning classes) |

### What to submit (per group of three)

- **A zip with code**, containing:
  - an executed notebook
  - any supporting scripts
  - the original data, or clear download instructions
  - saved split indices
  - dependencies
  - a README with the execution order
- **Results report**: maximum **three-page PDF**. This is the project report; no additional long report is required.

### Assessment

| Component | Weight | Minimum |
| --- | --- | --- |
| Submitted artifacts | 50% | 30% |
| Individual written defence | 50% | 30% |

The written defence is limited open-book: **one A4 handwritten page** (both sides).

### AI-use and contribution declaration

Include a brief **AI-use and contribution declaration in the notebook**. Identify the tools used, their role, how you checked generated material, and each member's contribution. If no AI tools were used, state this.
