# lmplots
## Linear Regression Diagnostic Plots

For a linear regression model, it generates following diagnostic plots:

        a.*residual plot*
        b. qq plot of residuals
        c. scale location plot
        d. leverage
        e. Cook's distance plot
        f. Cook's distance againt leverage plot

        
        Args:
            model (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object
        
