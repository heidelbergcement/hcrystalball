=========
Changelog
=========

Version 0.1.12
=============
- Update failing param grid for exponential smoothing models
- Update depencencies
- Silently skip fialing autosarimaxin large scale CV

Version 0.1.11
=============
- Fix issues with unstable models picked during CV
- Improve logging (hcb_verbose flags)
- Update docs
- Update depencencies

Version 0.1.10
=============
- Add random states to examples and model grid configuration
- Fix damped_trend in statsmodels model definition

Version 0.1.9
=============

- Add support for python 3.6
- Fix loading of country code column for model selector
- Improve compatibility with sktime

Version 0.1.8
=============

- Add before_days, after_days, bridge_days features to the holiday transformer for better modeling of after, before and between holiday effects
- Add Theta model

Version 0.1.7
=============

- Add support for simultaneous usage of multiple holidays
- Add progress bars for model selection
- Improve plotting
- Column names created by Seasonality and Holiday transformers start with _ to prevent name clashes

Version 0.1.6
=============

- Fix country column handling in ModelSelector
- Improved experience of trying examples through Binder (pre-cached docker image)
- Adding templates for GitHub issues

Version 0.1.5
=============

- More informative error messages when trying to use wrappers without installed dependent packages
- Updating examples and dependencies to enable wider usage with core installation

Version 0.1.4
=============

- Adding Windows distribution
- Other internal fixes (CI, CD, conda distribution, pre-commit, code cleanup, test coverage)

Version 0.1.3
=============

- Python packaging fixes

Version 0.1.0
=============

- A time-series forecasting library developed by the data science team of Heidelbergcement using a scikit-learn compatible API and popular ML libraries as backend
- First public release
