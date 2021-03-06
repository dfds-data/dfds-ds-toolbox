Working examples
================

See :doc:`auto_examples/index` for examples of the plotting capabilities.


Model selection
---------------
Start by importing the switcher class from `model_selection`.::

    from dfds_ds_toolbox.model_selection.model_selection import ClfSwitcher, RegSwitcher

In this module there are two switcher classes; one for classification (:py:class:`ClfSwitcher`)
and one for regression (:py:class:`RegSwitcher`). The switcher classes each implement a base class that
allows us to switch between estimators.

In order to test out different models, and potentially also different model parameters,
we can make use of fx the `GridSearchCV`_ module implemented in sklearn.::

    from sklearn.model_selection import GridSearchCV


First, a parameter grid is defined, containing the models to test and each of their range of settings.::

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


Now the defined parameter grid can be searched, here using 3-fold cross validation.::

    gs = GridSearchCV(
        RegSwitcher(),
        parameters,
        cv = 3,
        n_jobs = 3,
        scoring = 'neg_mean_squared_error'
    )
    gs.fit(train[model_cols], train['target'])


Using the functionality of the `GridSearchCV`_ module, the results can be accesed in the following way::

    # To get the best performing estimator
    gs.best_estimator_

    # To evaluate and compare all combinations of estimator and parameter settings
    pd.DataFrame(gs.cv_results_)


Working with pipelines
----------------------
The model selection modules can also be tested together with transformers using the `pipeline`_ module from sklearn.::

    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import VarianceThreshold

First, define the pipeline to be test, in this case including both the feature selector and the switcher class::

    pipeline = Pipeline(
        [
            ('features', VarianceThreshold()),
            ('model', RegSwitcher())
        ]
    )

Next step is to define the estimators and parameters to be tested; in this example we will test a random forest
regressor and a linear regression model.
In principle we could add as many configurations and models as we want to this parameter grid,
but for simplicity we will just be testing these two.::

    parameters = [
        # RandomForestRegressor model and parameters to test
        {
            'features__threshold': [0.0, 0.1],
            'model__estimator': [RandomForestRegressor()],
            'model__estimator__n_estimators': [150, 200],
            'model__estimator__max_depth': [2, 3],
        },

        # LinearRegression model and parameters to test
        {
            'features__threshold': [0.0, 0.1],
            'model__estimator': [LinearRegression()],
            'model__estimator__fit_intercept': [True, False],
        }
    ]


Just as in previous example, this can now be tested using grid search::

    gs = GridSearchCV(
        pipeline,
        parameters,
        cv = 3,
        n_jobs = 3,
        scoring = 'neg_mean_squared_error'
    )
    gs.fit(train[model_cols], train['target'])


Again, using the functionality of the `GridSearchCV`_ module, the results can be accessed as described above.

Profiling
---------
In the process of speeding up our pipelines, it is really handy to get more information about the execution time of the
individual parts of our code (profiling), which can help us to spot bottlenecks.
An approach one could follow is using the built-in module in python `cProfile <https://docs.python.org/3/library/profile.html>`_.
An example of implementation can be found in  with the ``@profileit()`` decorator:

We can then add the decorator to whatever function we want to profile::

    from dfds_ds_toolbox.profiling.profiling import profileit
    from pathlib import Path

    @profileit(path=Path("profiles/"), name="main_script_profile")
    def main():
    ...

    if __name__ == "__main__":
        main()

And visualize the profiles with tools like `snakeviz <https://jiffyclub.github.io/snakeviz/>`_:

.. code-block:: bash

    snakeviz main_script_profile.stats

Logging
-------
The logging module allows you to easily log in console or/and in a debug file with coloured message levels and timestamps:

Usage::

    >>> from dfds_ds_toolbox.logging.logging import init_logger
    >>> logger = init_logger()
    >>> logger.info("This message will not be logged.")
    >>> logger.critical("Something BAD happened.")

    [2021-10-13 14:57:32,394] <span style="color:red">CRITICAL</span> test_log - Something BAD happened.

The default value of ``stream_level`` is set to ``"WARNING"``. If you want to log the messages on a different level than ``"WARNING"`` use::

    >>> logger = init_logger(stream_level="INFO")
    >>> logger.debug("This message will not be logged.")
    >>> logger.info("Starting some work.")

If you want to save additionally the log the messages into a file use::

    >>> logger = init_logger(stream_level="INFO", debug_file="path/log_files.log")
    >>> logger.debug("Logging something to a file.")
    >>> logger.info("Logging something to both terminal and file.")



.. _GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
.. _pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
