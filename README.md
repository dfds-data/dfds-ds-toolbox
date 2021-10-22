# Introduction 
This repo is intended to contain a packaged toolbox of some neat, frequently-used data science code snippets and functions. The intention is that the classes should be compatible with the [sklearn](https://scikit-learn.org/stable/) library.

Have a look at https://dfds-dstoolbox-docs.s3.eu-central-1.amazonaws.com/index.html for user guide.

Already implemented:
* Feature selector for regression problems
* Model selector for regression and classification problems
* Profiling tool for generating stats files of the execution time of a function 

To be implemented in the future:

* Preprocessing 
    * Imbalanced datasets
    * Outlier detection & handling
    * Missing value imputation

* Feature generation
    * Binning
    * Type variables, create multiple features
    * Timestamp, seasonality variables
    * Object: onehot, grouping, etc.

* Performance analysis (plots, summary, error analysis)

More ideas might arise in the future and should be added to the list.

A guide on how to install the package and some working examples of how to use the classes can be found in later sections. 
# Getting Started
## Install locally
Make a virtual environment:
```shell
python -m venv venv
venv\Scripts\activate.bat
```
Install dependencies
```shell
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
pip install -e .
```

Run tests to see everything working
```shell
pytest
```

## Install this library in another repo

Make sure your virtual environment is activated, then install the required packages
```shell
python -m pip install --upgrade pip
pip install keyring artifacts-keyring
```
If you want to install the package `ds_toolbox` version 3.0, you should run
```shell
pip install --extra-index-url=https://dfds.pkgs.visualstudio.com/_packaging/smartdata/pypi/simple/ ds_toolbox==0.3.0
```
For more information see [this guide in the wiki](https://dfds.visualstudio.com/Smart%20Data/_wiki/wikis/Smart-Data.wiki/2779/Installing-a-package-from-the-smartdata-artifact-feed)

# Versions
* 0.1.0 => Inclussion of feat selector and model selector
* 0.2.0 => Fix bugs to install
* 0.3.0 => Plotting updates (Univariate, Pred_Real, AUC, Lift)
Remember to add descriptions of any new versions, please include working examples

# Working examples
For the working examples in this session we will focus on a regression problem and use the boston dataset from sklearn.

```
from sklearn import datasets
boston = datasets.load_boston()
```


## Feature selection
Start by importing the feature selection class.

```
from feature_selection.feat_selector import RegFeatureSelector
```

The ```RegFeatureSelector``` class takes a dataframe as input and therefore, the first thing to do is convert the boston data into a pandas dataframe.

```
model_cols = list(boston['feature_names'])
df = pd.DataFrame(
    data = np.c_[boston['data'], boston['target']], 
    columns = model_cols + ['target']
)
```

For training and testing purposes, we will create a train and test dataframe.

```
test = df.sample(frac = 0.2, replace = False)
train = df[~df.index.isin(test.index)]
```

The ```RegFeatureSelector``` class implements different strategies for feature selection:

* variance
* l1
* rf_feature_importance
* rf_top_features
* stepwise

The list of strategies can be extracted from the class object

```
available_strategies = RegFeatureSelector()._available_strategies
```

Now we are ready to perform the feature selection

```
for strategy in available_strategies:
    print('\nStrategy=',strategy)
    fs = RegFeatureSelector(strategy = strategy)
    X_adj = fs.fit_transform(train[model_cols], train['target'])
    selected_cols = list(X_adj.columns)
    print('selected_cols = ',len(selected_cols), sorted(selected_cols))
```

This code snippet will, for each strategy, print out a list of chosen features, and transform the dataset accordingly.

## Model selection
Start by importing the switcher class from ```model_selection```.

```
from model_selection.model_selection import ClfSwitcher, RegSwitcher
```

In this module there are two switcher classes; one for classification (```ClfSwitcher```) and one for regression (```RegSwitcher```). The switcher classes each implement a base class that allows us to switch between estimators.

In order to test out different models, and potentially also different model parameters, we can make use of fx the [```GridSearchCV```](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) module implemented in sklearn.

```
from sklearn.model_selection import GridSearchCV
```

First, a parameter grid is defined, containing the models to test and each of their range of settings.

```
parameters = [
    {
        'estimator': [RandomForestRegressor()],
        'estimator__n_estimators':[150, 200], 
        'estimator__max_depth':[2, 3]
    },
    {
        'estimator':[LinearRegression()],
        'estimator__fit_intercept': [True, False]
    }
]
```

Now the defined parameter grid can be searched, here using 3-fold cross validation

```
gs = GridSearchCV(
    RegSwitcher(), 
    parameters, 
    cv = 3, 
    n_jobs = 3, 
    scoring = 'neg_mean_squared_error'
)
gs.fit(train[model_cols], train['target'])
```

Using the functionality of the [```GridSearchCV```](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) module, the results can be accesed in the following way

```
# To get the best performing estimator
gs.best_estimator_ 

# To evaluate and compare all combinations of estimator and parameter settings
pd.DataFrame(gs.cv_results_)
```

## Working with pipelines
The feature selection and model selection modules can also be tested together using the [```pipeline```](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) module from sklearn.

```
from sklearn.pipeline import Pipeline
```

First, define the pipeline to be test, in this case including both the feature selector and the switcher class

```
pipeline = Pipeline(
    [
        ('feature_selector', RegFeatureSelector()), 
        ('model', RegSwitcher())
    ]
)
```

Next step is to define the estimators and parameters to be tested; in this example we will test a random forest regressor and a linear regression model, and then include all available strategies for the feature selector. In principle we could add as many configurations and models as we want to this parameter grid, but for simplicity we will just be testing these two.

```
parameters = [
    # RandomForestRegressor model and parameters to test
    {
        'model__estimator': [RandomForestRegressor()],
        'model__estimator__n_estimators': [150, 200], 
        'model__estimator__max_depth': [2, 3],
        'feature_selector__strategy': available_strategies
    },

    # LinearRegression model and parameters to test
    {
        'model__estimator': [LinearRegression()],
        'model__estimator__fit_intercept': [True, False],
        'feature_selector__strategy': available_strategies
    }
]
```

Just as in previous example, this can now be tested using grid search

```
gs = GridSearchCV(
    pipeline, 
    parameters, 
    cv = 3, 
    n_jobs = 3, 
    scoring = 'neg_mean_squared_error'
)
gs.fit(train[model_cols], train['target'])
```

Again, using the functionality of the [```GridSearchCV```](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) module, the results can be accesed as described above.

## Profiling
In the process of speeding up our pipelines, it is really handy to get more information about the execution time of the individual parts of our code (profiling), which can help us to spot bottlenecks. An approach one could follow is using the built-in module in python [```cProfile```](https://docs.python.org/3/library/profile.html). An example of implementation can be found in  with the ```@profileit()``` decorator:

We can then add the decorator to whatever function we want to profile:
```python
from ds_toolbox.profiling.profiling import profileit
from pathlib import Path

@ profileit(profiles_path=Path("profiles/"), profile_name="main_script_profile")
def main():
...

if __name__ == "__main__":
    main()
```
And visualize the profiles with tools like [```snakeviz```](https://jiffyclub.github.io/snakeviz/):
```cmd
snakeviz main_script_profile.stats
``` 

## Logging
The logging module allows you to easily log in console or/and in a debug file with coloured message levels and timestamps:

Usage:
```python
>>> from ds_toolbox.logging.logging import init_logger
>>> logger = init_logger()
>>> logger.info("This message will not be logged.")
>>> logger.critical("Something BAD happened.")
```
               
[2021-10-13 14:57:32,394] <span style="color:red">CRITICAL</span> test_log - Something BAD happened.  

The default value of `stream_level` is set to `"WARNING"`. If you want to log the messages on a different level than `"WARNING"` use:
```python
>>> logger = init_logger(stream_level="INFO")
>>> logger.debug("This message will not be logged.")
>>> logger.info("Starting some work.")
```
If you want to save additionally the log the messages into a file use:
```python
>>> logger = init_logger(stream_level="INFO", debug_file="path/log_files.log")
>>> logger.debug("Logging something to a file.")
>>> logger.info("Logging something to both terminal and file.")
```


# Contribute
We want this library to be useful across many data science projects.
If you have some standard utilities that you keep using in your projects, please add them here and make a PR.

## Documentation
### Website

The full documentation of this package is available at https://dfds-dstoolbox-docs.s3.eu-central-1.amazonaws.com/index.html
To build the documentation locally run: 
```shell
cd docs/
sphinx-apidoc -o . ../ds_toolbox/ ../*tests*
make html
```
Now, you can open the documentation site in `docs/_build/index.html`.

### Style
We are using Googles [Python style guide](https://google.github.io/styleguide/pyguide.html#381-docstrings) convention for docstrings. 
This allows us to make an up-to-date documentation website for the package. 

In short, every function should have a short one-line description, optionally a longer description afterwards and a list of parameters. 
For example
```python
def example_function(some_parameter: str, optional_param: int=None) -> bool:
    """This function does something super smart.
    
    Here I will dive into more detail about the smart things.
    I can use several lines for that. 
    
    Args:
        some_parameter: Name of whatever
        optional_param: Number of widgets or something. Only included when all the starts align. 
        
    Returns:
         An indicator describing if something is true. 
    """
```

There are many other style issues that we can run into, but if you follow the Google style guide, you will probably be fine. 
