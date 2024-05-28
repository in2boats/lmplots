# lmplots
## Linear Regression Diagnostic Plots

For a linear regression model, it generates following diagnostic plots:

1. _Residual plot_
2. _Q-Q plot of residuals_
3. _Scale location plot_
4. _Residuals vs Leverage plot_
5. _Cook's distance plot_
6. _Cook's distance againt leverage plot_
    
Args:
        model (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
        must be instance of statsmodels.regression.linear_model object

Raises:
        TypeError: if instance does not belong to above object
        

This class is a Python implementation of `plot.lm()` function from `R`.

For some details check: https://medium.com/@biman.pph/linear-regression-diagnostic-plots-using-python-a-comprehensive-guide-178aaa24dc13
