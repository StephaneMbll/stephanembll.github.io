# Scikit-Learn

Throughout this notebook, the words *****data***** and ******target****** are used respectively for X and Y  

# Module 1:

## Introduction

Machine learning: building predictive models (builds prediction rules from data)

rows = samples / columns = features / annotated = associated with label ou target class / predict Y from X (supervised learning)

regression si target class est continue

tabular data : 2D table 

## Tabular Data Exploration

```python
target.value_counts()

pandas.crosstab(index= .., columns = ..) # relations entre features

seaborn.pairplot() / scatterplot()

dataset.describe()
```

**seaborn** pairplot ou scatter plot : how each var differs according to target

Is target variable imbalanced ? Redundant columns ?

## Fitting a model with scikit learn

Séparer data et target

### K-nearest neighbors

‘To predict the target of a new sample, a k-nearest neighbors takes into account its K closest samples in the training set and predicts the majority target of these samples.”

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
_ = model.fit(data, target)
```

fit : Learning algorithm + model states (used to predict)

Average success rate :  [ target == target_predicted].mean()]

### Train/test data split

Always test generalization on different set than training set

Identify numerical/categorical datas: data.dtype

```python
X_train, X_test, Y_train, Y_test = 
train_test_split(data_numeric, target, test_size=0.25, shuffle=True/False)
```

### Logistic Regression

Linear models family

In short, linear models find a set of weights to combine features linearly and predict the target (a*f1 + b*f2 usw)

Dummy Classifier : make predictions ignoring input features (’constant’ strategy for specific target)

# Preprocessing

Why normalize ?

> Models that rely on the distance between a pair of samples, for instance
k-nearest neighbors, should be trained on normalized features to make each
feature contribute approximately equally to the distance computations.
> 

> Many models such as logistic regression use a numerical solver (based on
gradient descent) to find their optimal parameters. This solver converges
faster when the features are scaled.
> 

```python
from sklearn.preprocessing import StandardScaler

scaler.fit(data_train)
data_train_scaled = scaler.transform(data_train)
```

****************************StandardScaler****************************: Shifts and scales each feature individually so that they all have a 0-mean and a unit standard deviation. For each feature, we subtract its mean and divide by its standard deviation.

## Pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(data_train, target_train)
model.predict(data_target)
```

Pipeline exposes same method as final predictor

### Visualiser pipeline

```python
from sklearn import set_config
set_config(display='diagram')
model
```

## Scoring

Obviously score after model fitting

```python
model.score(X_test, Y_test)
```

Scoring computes ****************accuracy**************** for **********************classifiers**********************.

⚠️ Whereas ********************************cross-validation******************************** evaluates the ******************************************************generalization performances****************************************************** of a model.

## Cross Validation

********************************Cross Validation********************************: Mean (?) evaluation of a model by repeating the split (train/test) such that the training and testing sets are different for each evaluation

**********K-Fold Cross Validation**********: K being number of splits of the dataset, so you get K final scores (dataset is divided in train/test K times)

Fort the sake of simplicity, we choose to summarize the cross_validation scores by their mean and their standard deviation across differents splits.

(The more splits you make, the more likely it is to draw repeated datas)

```python
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(n_splits = 5, shuffle = True)
test_scores = cross_val_score(model, data, target, cv=cv)

cv_results = cross_validate(model, data_categorical, target)
scores = cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")

# The cross_validate function differs from cross_val_score in two ways:
#It allows specifying multiple metrics for evaluation.
#It returns a dict containing fit-times, score-times (and optionally training scores as well as fitted estimators) in addition to the test score.
```

Full data should not be used for scoring a model → train-test split to test generalization

Estimate variability of the estimate → cross-validation

The goal of cross-validation is not to train a model, but rather to estimate approximately the generalization performance of a model that would have been trained to the full training set, along with an estimate of the variability (+ compute standard deviation)

## Categorical Features

Categorical variables have discrete values, typically represented by string labels (but not only) taken from a finite list of possible choices.

```python
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
```

### Encode categorical into numerical features

************************************Ordinal categories************************************: encode each category with a number

```python
from sklearn.preprocessing import OrdinalEncoder

education_column = data_categorical[["education"]]

encoder = OrdinalEncoder()
education_encoded = encoder.fit_transform(education_column)
```

Categories are encoded for each feature independantly

⚠️ Predictive models can assume that values are ordered (0 < 1 < 2 …)

*By default*: lexicographical strategy (alphabetical order)

**********************************One Hot Encoding**********************************: For a given feature, it will create as many new columns as there are possible categories. For a given sample, the value of the column corresponding to the category will be set to `1` while all the columns of the other categories will be set to `0`.

⚠️ Prevents downstream models to make assumption about categories ordering.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
education_encoded = encoder.fit_transform(education_column)
```

[**************************************Sparse matrices************************************** are efficient when most elements are 0]

Associating features names and encoding

```python
feature_names = encoder.get_feature_names_out(input_features=["education"])
education_encoded = pd.DataFrame(education_encoded, columns=feature_names)
```

********BUT:********

OneHot Encoding flattens dataset features:  per row, one large array containing X (features) * nb of categories per X feature (e.g from 8 to 102 features) 

![screenshot_26-10-2022_17:34:26.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/77b11251-3885-4391-932f-ee1d34ec3160/screenshot_26-10-2022_173426.png)

### Choosing encoding

**********************************If linear models:********************************** OneHotEncoder

****************************************If tree based models****************************************: OrdinalEncoder

⚠️ Linear model and `OrdinalEncoder` are used together only for ordinal categorical features, i.e. features that have a specific ordering. Otherwise, your model will perform poorly.

If rare category in a feature → can be a probleme during cross validation

→ either provide keyword ‘categories’ ******OR****** use parameter **************handle_unknown**************

⚠️ ****************************OneHotEncoding**************************** training is much longer than ******OrdinalEncoder****** (generates 10* more features)

## Numerical and categorical features together

Define 2 selectors, one for numerical and one for categorical

```python
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)
```

Dispatch data to processors according to their type → ColumnTransformer

```python
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()
preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

#Or, if some datas dont need transform

preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_columns)],
    remainder="passthrough")

#And then simply pipe the ColumnTransformer

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))&
```

## Decision trees based models

👍 pas besoin de scale les numerical features

👍 Ordinal encoding is fine for categorical, even if resulting in arbitrary ordering

********************************************Gradient-boosting tree :******************************************** useful  whenever the dataset has a large number of samples and limited number of informative features (e.g. less than 1000) with a mix of numerical and categorical variables.

—> Numerical variables dont need scaling + categorical var as OrdinalEncoder (avoid high dim representations)

```python
from sklearn.ensemble import HistGradientBoostingClassifier
model = make_pipeline(preprocessor, HistGradientBoostingClassifier())
```

`HistGradientBoostingClassifier` is expressive and robust enough to deal with misleading ordering of integer coded categories (which was not the case for linear models).

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11d80305-76b1-473c-8dd5-34d9b68d1a85/Untitled.png)

## Imputer

****************************SimpleImputer**************************** completes missing variables according to given strategy (cst or stat)

## SVM

******************************************************************Support Vector Machine Classifier******************************************************************: linear, except use with kernel

# Module 2

## Errors and Overfitting Underfitting

**Overfitting**: 

- model too complex, not enough data
- testing error >> training error

************************Underfitting************************: 

- model too simple for the data
- even training error is large

```jsx
score = mean_absolute_error(target_train, target_predicted)
```

Empirical error = training error (predict on training set)

Generalization error = testing error (predict on test set)

Different model families = different complexity & inductive bias

## Validation and Learning Curves

******************Validation Curve:****************** Plot testing and training score according to variation of a  hyperparameter

```python
from sklearn.model_selection import validation_curve
train_scores, test_scores = validation_curve(
    regressor, data, target, param_name="max_depth", param_range=max_depth,
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
# Convert the scores into errors
train_errors, test_errors = -train_scores, -test_scores
plt.plot(max_depth, train_errors.mean(axis=1), label="Training error")
plt.plot(max_depth, test_errors.mean(axis=1), label="Testing error")
plt.legend()
plt.xlabel("Maximum depth of decision tree")
plt.ylabel("Mean absolute error (k$)")
_ = plt.title("Validation curve for decision tree")
#In this specific case we're testing depth of a decision tree
```

********************************Learning Curve:******************************** Plot testing and training score according to variation of the samples number

```python
from sklearn.model_selection import learning_curve
results = learning_curve(
    regressor, data, target, train_sizes=train_size"s, cv=cv,
    scoring="neg_mean_absolute_error", n_jobs=2)
train_size, train_scores, test_scores = results[:3]
# Convert the scores into errors
train_errors, test_errors = -train_scores, -test_scores

plt.errorbar(train_size, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training error")
plt.errorbar(train_size, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Testing error")
plt.legend()

plt.xscale("log")
plt.xlabel("Number of samples in the training set")
plt.ylabel("Mean absolute error (k$)")
_ = plt.title("Learning curve for decision tree")
```

## Bias versus Variance d’un modèle

******************************************************Biais élevé == Underfitting******************************************************

****************Variance élevée == Overfitting****************

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0afb550d-e980-4059-a368-cf1d6ee34a1e/Untitled.png)

# Hyperparameter Tuning

## Manual Tuning

Change hyperparameter on a created model:

```python
#model is a pipeline of a scaler into a logreg called classifier
model.set_params(classifier__C=1e3)
#ex utile for evaluation
for val in value_list:
	model.set_params(xxx)
	cross_validate() 
```

## Automated Tuning

### Grid-Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate': (0.01, 0.1, 1, 10),
    'classifier__max_leaf_nodes': (3, 10, 30)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=2, cv=2)

#Fit
model_grid_search.fit(data_train, target_train)
```

⚠️ Quickly computationally expensive the more param_grid you search

```python
print(f"The best set of parameters is: "
      f"{model_grid_search.best_params_}")
#Wanna see all results
cv_results = pd.Dataframe(model_grid_search.cv_results_)
						.sort_values("mean_test_score", ascending=False)
```

### Randomized-Search

Grid can be too regular (optimal hyper can be between given values): “Rather, stochastic search will sample hyperparameter 1 independently from hyperparameter 2 and find the optimal region.”

—> Random search is typically beneficial compared to grid search to optimize 3 or more hyperparameters.

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'classifier__l2_regularization': loguniform(1e-6, 1e3),
    'classifier__learning_rate': loguniform(0.001, 10),
    'classifier__max_leaf_nodes': loguniform_int(2, 256),
    'classifier__min_samples_leaf': loguniform_int(1, 100),
    'classifier__max_bins': loguniform_int(2, 255),
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    cv=5, verbose=1,
)
model_random_search.fit(data_train, target_train)

#Noms de variables trop grands

def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

cv_results = cv_results.rename(shorten_param, axis=1)
cv_results
```

Augment number of iterations (n_iter) to find potentially better sets of parameters 

### Evaluation

**********************************************Nested cross-validation**********************************************: An inner cross validation for selection of the hyperparameters (GridSearchCV z.B), and an outer cross validation for evaluation of generalization from tuned fitted model

```python
model.fit() 
#essaie diff params puis cv sur chacun pr eval
cv_results = pd.DataFrame(model_grid_search.cv_results_)
#into
cv_results = cross_validate(model, ...)
```

```python
nystroem_regression.fit(data, target) #or any model
target_predicted = nystroem_regression.predict(data)
mse = mean_squared_error(target, target_predicted)

ax = sns.scatterplot(data=full_data, x="input_feature", y="target",
                     color="black", alpha=0.5)
ax.plot(data, target_predicted)
_ = ax.set_title(f"Mean squared error = {mse:.2f}")
```

## Linear Models

Y = Ax + B with **A** = coef and **B** = intercept

Simple and fast baselines for:

- regression (**********************************linear regression**********************************)
- classification ******************************(logistic regression******************************)

Underfit when nb of features << nb of samples

Excellent with large nb of features

- Optimal parameters can be found founding an equation (fit on training)

two metrics: (i) the mean squared error and (ii) the mean absolute error.

For non linear features-target relationship

Indeed, there are 3 possibilities to solve this issue:

1. choose a model that can natively deal with non-linearity,
2. engineer a richer set of features by including expert knowledge which can
be directly used by a simple linear model,  (expand data if we know their structure to break linearity)

use a "kernel" to have a locally-based decision function instead of a
global linear decision function

******************Kernels:****************** Instead of learning a weight per feature as we previously emphasized, a weight will be assigned to each sample. However, not all samples will be used. This is the base of the support vector machine algorithm.

```python
from sklearn.svm import SVR
svr = SVR(kernel="poly", degree=3) #kernel: "poly" or "linear"
```

Kernel methods such as SVR are very efficient for small to medium datasets.

For larger datasets with `n_samples >> 10_000`, it is often computationally more efficient to perform explicit feature expansion using `PolynomialFeatures` or other non-linear transformers from scikit learn such as [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) or [Nystroem](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html)

**PolynomialFeatures** create additional features to encode non linear interactions between features (can easily overfit)

```python
from sklearn.preprocessing import PolynomialFeatures

#transforms the features

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression(),
)

#include_bias = False pour pas créer une colonne parfaitement corrélée au parametre intercept
```

Linear model stores value in **.coef_** and **.intercept**

**Underfitting** : faible nombre de variables d’entrées par rapport aux données d’entrainement

**Overfitting:** inclus trop de caractéristiques

Comment choisir caractéristiques importantes / qui n’entraîneront pas de biais ?

### Ridge

Pour introduire la régularisation (essaie de tirer les coeffs vers 0)

parametre alpha controle force de régularisation 

Looking specifically to weights values, we observe that increasing the value
of `alpha` will decrease the weight values. A negative value of `alpha` would
actually enhance large weights and promote overfitting

**.get_features_names_out** to generate features names

a ridge model will enforce all weights to have a similar magnitude, while the overall magnitude of the weights is shrunk towards zero with respect to the linear regression model.

Use of a StandardScaler before ridge to make it easier to find optimal parameters

we should include search of the hyperparameter `alpha`
 within the
cross-validation (RidgeCV)

```python
import numpy as np
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-2, 0, num=21)
ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                      RidgeCV(alphas=alphas, store_cv_values=True))

best_alphas = [est[-1].alpha_ for est in cv_results["estimator"]]

mse_alphas = [est[-1].cv_values_.mean(axis=0) for est in cv_results["estimator"]]
cv_alphas = pd.DataFrame(mse_alphas, columns=alphas)
cv_alphas = cv_alphas.aggregate(["mean", "std"]).T
cv_alphas
```

## Logistic Regression

binary outcome but use logistic to predict probability

Decision function boundary

We see that a small `C` will shrink the weights values toward zero. It means that a small `C` provides a more regularized model. Thus, `C` is the inverse of the `alpha` coefficient in the `Ridge` model

Linear classification expects data to be linearly separable, sinon not expressive enough → feature augmentation or kernel based method

## Wrap-up

In this module, we saw that:

- the predictions of a linear model depend on a weighted sum of the values of
the input features added to an intercept parameter;
- fitting a linear model consists in adjusting both the weight coefficients and
the intercept to minimize the prediction errors on the training set;
- to train linear models successfully it is often required to scale the input
features approximately to the same dynamic range;
- regularization can be used to reduce over-fitting: weight coefficients are
constrained to stay small when fitting;
- the regularization hyperparameter needs to be fine-tuned by cross-validation
for each new machine learning problem and dataset;
- linear models can be used on problems where the target variable is not
linearly related to the input features but this requires extra feature
engineering work to transform the data in order to avoid under-fitting.

# Classification Trees

Faudra faire le module 6 il a l’air stylé 

M5.02 : a - a - c

M5.03 : b - b - b

M5.04 : b - a

M5.05 : c - b - b - b 

M6.01 : a - a - ac

M6.02 : acd - bc - a

M6.03 : ad - c - ab

M6.04 : c - c - be - bc - a - bc -
