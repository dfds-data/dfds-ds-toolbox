# Introduction

This repo is intended to contain a packaged toolbox of some neat,
frequently-used data science code snippets and functions. The intention is that
the classes should be compatible with the
[sklearn](https://scikit-learn.org/stable/) library.

Have a look at
https://dfds-dstoolbox-docs.s3.eu-central-1.amazonaws.com/index.html for user
guide.

Already implemented:

- Feature selector for regression problems
- Model selector for regression and classification problems
- Profiling tool for generating stats files of the execution time of a function

To be implemented in the future:

- Preprocessing

  - Imbalanced datasets
  - Outlier detection & handling
  - Missing value imputation

- Feature generation

  - Binning
  - Type variables, create multiple features
  - Timestamp, seasonality variables
  - Object: onehot, grouping, etc.

- Performance analysis (plots, summary, error analysis)

More ideas might arise in the future and should be added to the list.

A guide on how to install the package and some working examples of how to use
the classes can be found in later sections.

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

Make sure your virtual environment is activated, then install the required
packages

```shell
python -m pip install --upgrade pip
pip install keyring artifacts-keyring
```

If you want to install the package `ds_toolbox` version 3.0, you should run

```shell
pip install --extra-index-url=https://dfds.pkgs.visualstudio.com/_packaging/smartdata/pypi/simple/ ds_toolbox==0.3.0
```

For more information see
[this guide in the wiki](https://dfds.visualstudio.com/Smart%20Data/_wiki/wikis/Smart-Data.wiki/2779/Installing-a-package-from-the-smartdata-artifact-feed)

# Versions

- 0.1.0 => Inclussion of feat selector and model selector
- 0.2.0 => Fix bugs to install
- 0.3.0 => Plotting updates (Univariate, Pred_Real, AUC, Lift) Remember to add
  descriptions of any new versions, please include working examples
- 0.7.0 => Following new coding guidelines. Functions and variables have been
  renamed.
-

# Contribute

We want this library to be useful across many data science projects. If you have
some standard utilities that you keep using in your projects, please add them
here and make a PR.

## Documentation

### Website

The full documentation of this package is available at
https://dfds-dstoolbox-docs.s3.eu-central-1.amazonaws.com/index.html To build
the documentation locally run:

```shell
cd docs/
sphinx-apidoc -o . ../ds_toolbox/ ../*tests*
make html
```

Now, you can open the documentation site in `docs/_build/index.html`.

### Style

We are using Googles
[Python style guide](https://google.github.io/styleguide/pyguide.html#381-docstrings)
convention for docstrings. This allows us to make an up-to-date documentation
website for the package.

In short, every function should have a short one-line description, optionally a
longer description afterwards and a list of parameters. For example

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

There are many other style issues that we can run into, but if you follow the
Google style guide, you will probably be fine.

### Examples

To show the intended use and outcome of some of the included methods, we have
included a gallery of plots in `examples/`. To make a new example create a new
file and name it something like `plot_<whatever>.py`. Start this file with a
docstring, for example

```python
"""
Univariate plots
================

For a list of features separate in bins and analysis the target distribution in both Train and Test
"""
```

and after this add the python code needed to create the example plot.
