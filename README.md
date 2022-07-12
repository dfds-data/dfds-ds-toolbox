# Introduction

This repo is intended to contain a packaged toolbox of some neat,
frequently-used data science code snippets and functions. The intention is that
the classes should be compatible with the
[sklearn](https://scikit-learn.org/stable/) library.

Have a look at https://dfds-ds-toolbox.readthedocs.io for user guide.

Already implemented:

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

We use poetry as the package manager and build tool. Make sure you have poetry
installed locally, then run

```shell
poetry install
```

Run tests to see everything working

```shell
poetry run pytest
```

## Install this library in another repo

Make sure your virtual environment is activated, then install the required
packages

```shell
python -m pip install --upgrade pip
```

If you want to install the package `dfds_ds_toolbox` version 0.8.0, you should
run

```shell
pip install dfds_ds_toolbox==0.8.0
```

# Versions

See changelog at
[GitHub](https://github.com/dfds-data/dfds-ds-toolbox/releases).

# Contribute

We want this library to be useful across many data science projects. If you have
some standard utilities that you keep using in your projects, please add them
here and make a PR.

## Releasing a new version

When you want to release a new version of this library to
[PyPI](https://pypi.org/project/dfds-ds-toolbox/), there is a few steps you must
follow.

1. Update the version in `setup.py`. We follow
   [Semantic Versioning](https://semver.org/), so think about if there is any
   breaking changes in your release when you increment the version.
2. Draft a new release in
   [Github](https://github.com/dfds-data/dfds-ds-toolbox/releases/new). You can
   follow this link or click the "Draft a new release button" on the "releases"
   page.
   1. Here you must add a tag in the form "v<VERSION>", for example "v0.9.2".
      The title should be the same as the tag.
   2. Add release notes. The easiest is to use the button "Auto-generate release
      notes". That will pull titles of completed pull requests. Modify as
      needed.
3. Click "Publish release". That will start a
   [Github Action](https://github.com/dfds-data/dfds-ds-toolbox/actions) that
   will build the package and upload to PyPI. It will also build the
   documentation website.

## Documentation

### Website

The full documentation of this package is available at
https://dfds-ds-toolbox.readthedocs.io

To build the documentation locally run:

```shell
pip install -r docs/requirements.txt
cd docs/
sphinx-apidoc -o . ../dfds_ds_toolbox/ ../*tests*
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
