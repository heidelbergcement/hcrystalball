.. _glossary:

Glossary
==========

There are many disciplines involved in the world of time series forecasting and many use different names for the same/similar things.
To make it easier for further package navigation, these are important
terms mentioned throughout the package.

.. list-table::
   :header-rows: 1
   :widths: 40, 100, 50

   * - Term
     - Meaning
     - Synonyms

   * - Wrapper
     - Convenient adapter to third party estimators following Sklearn API enabling time series forecasting
     -
   * - Horizon
     - Number of datapoints within defined data frequency that are predicted
     - Steps-ahead
   * - Cross-validation
     - Procedure to run model's fitting and predicting on several test/train data splits in order to assess model ability to predict out-of-sample
     - CV
   * - Split
     - One of the train/test data subsamples (n_splits) created during Cross-validation (based on splitting strategy)
     -
   * - Target
     - Data we are trying to predict in form array_like, (1d)
     - Target vector, Target variable, Dependent variable, Label, Response variable, Explained variable, Outcome variable, Output variable, Endogenous variable, Y
   * - Feature
     - Measurable property (e.g. is_holiday) of the target (e.g. sales) in form of pandas.Dataframe
     - Exogenous variable, Covariate, Regressor, Independent variable, Explanatory variable, Predictor variable, Input, X
