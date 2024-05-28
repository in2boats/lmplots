import statsmodels
import numpy as np
import seaborn as sns
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
import matplotlib.pyplot as plt
from typing import Type

class lmPlots():
    '''
    Diagnostic plots for linear regression models from Statsmodels
    Similar to the plots from the plot.lm function in R
    '''
    
    def __init__(self,
                 model: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual plot
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
        """
        
        if isinstance(model, statsmodels.regression.linear_model.RegressionResultsWrapper) is False:
            raise TypeError("model must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object")

        self.model = maybe_unwrap_results(model)

        self.y = self.model.model.endog
        self.fitted = self.model.fittedvalues
        self.x = self.model.model.exog
        self.xnames = self.model.model.exog_names

        self.residuals = np.array(self.model.resid)
        influence = self.model.get_influence()
        self.studresid = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_d = influence.cooks_distance[0]
        self.nparams = len(self.model.params)
        self.nresids = len(self.studresid)
        
    def __call__(self, which=None, plot_context='seaborn-v0_8', **kwargs):
        # print(plt.style.available)
        # GH#9157
        if plot_context not in plt.style.available:
            plot_context = 'default'
        with plt.style.context(plot_context):
            if which == None:
                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
                self.residplot(ax=ax[0,0])
                self.qqplot(ax=ax[0,1])
                self.scalelocplot(ax=ax[1,0])
                self.levplot(
                    ax=ax[1,1],
                    high_leverage_threshold = kwargs.get('high_leverage_threshold'),
                    cooks_threshold = kwargs.get('cooks_threshold'))
                plt.show()
            elif which == 1:
                fig, ax = plt.subplots()
                self.residplot(ax=ax)
                plt.show()
            elif which == 2:
                fig, ax = plt.subplots()
                self.qqplot(ax=ax)
                plt.show()
            elif which == 3:
                fig, ax = plt.subplots()
                self.scalelocplot(ax=ax)
                plt.show()
            elif which == 4:
                fig, ax = plt.subplots()
                self.levplot(ax=ax,  high_leverage_threshold = kwargs.get('high_leverage_threshold'),
                    cooks_threshold = kwargs.get('cooks_threshold'))
                plt.show()
            
            elif which == 5:
                fig, ax = plt.subplots()
                self.cookdplot(ax=ax)
                plt.show()   
            elif which == 6:
                fig, ax = plt.subplots()
                self.cookdlevplot(ax=ax)
                plt.show() 
            else:
                raise ValueError('which parameter value is not implemented')

        return fig, ax
    
    def residplot(self,ax=None):
    
        if ax == None:
            fig, ax = plt.subplots()
        sns.residplot(x=self.fitted, y=self.residuals, lowess=True,
                      scatter_kws={'alpha': 0.5},
                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, ax=ax)
        
        # annotations
        residual_abs = np.abs(self.residuals)
        abs_resid = np.flip(np.argsort(residual_abs), 0)
        abs_resid_top_3 = abs_resid[:3]
        for i in abs_resid_top_3:
            ax.annotate(i, xy=(self.fitted[i], self.residuals[i]), color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals');
    
        return ax
    
    
    def qqplot(self,ax=None):
    
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.studresid)
        fig = QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)
        
        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.studresid)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for i, x, y in self.__qq_top_resid(QQ.theoretical_quantiles, abs_norm_resid_top_3):
            ax.annotate(i, xy=(x, y), ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax
    
    def __qq_top_resid(self, quantiles, top_residual_indices):
        """
        Helper generator function yielding the index and coordinates
        """
        offset = 0
        quant_index = 0
        previous_is_negative = None
        for resid_index in top_residual_indices:
            y = self.studresid[resid_index]
            is_negative = y < 0
            if previous_is_negative == None or previous_is_negative == is_negative:
                offset += 1
            else:
                quant_index -= offset
            x = quantiles[quant_index] if is_negative else np.flip(quantiles, 0)[quant_index]
            quant_index += 1
            previous_is_negative = is_negative
            yield resid_index, x, y
            
    def scalelocplot(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        
        residual_norm_abs_sqrt = np.sqrt(np.abs(self.studresid))

        ax.scatter(self.fitted, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(x=self.fitted, y=residual_norm_abs_sqrt, scatter=False, ci=False,
                    lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                    ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(i, xy=(self.fitted[i], residual_norm_abs_sqrt[i]),
                        color='C3')

        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax
    
    def levplot(self, ax=None, high_leverage_threshold=False, cooks_threshold='baseR'):
        if ax is None:
            fig, ax = plt.subplots()
        
        
        ax.scatter(self.leverage, self.studresid, alpha=0.5);

        sns.regplot(x=self.leverage, y=self.studresid,scatter=False,
                    ci=False, lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                    ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_d), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(i, xy=(self.leverage[i], self.studresid[i]),color = 'C3')

        factors = []
        if cooks_threshold == 'baseR' or cooks_threshold is None:
            factors = [1, 0.5]
        elif cooks_threshold == 'convention':
            factors = [4/self.nresids]
        elif cooks_threshold == 'dof':
            factors = [4/ (self.nresids - self.nparams)]
        else:
            raise ValueError("threshold_method must be one of the following: 'convention', 'dof', or 'baseR' (default)")
        for i, factor in enumerate(factors):
            label = "Cook's distance" if i == 0 else None
            xtemp, ytemp = self.__cooks_dist_line(factor)
            ax.plot(xtemp, ytemp, label=label, lw=1.25, ls='--', color='red')
            ax.plot(xtemp, np.negative(ytemp), lw=1.25, ls='--', color='red')

        if high_leverage_threshold:
            high_leverage = 2 * self.nparams / self.nresids
            if max(self.leverage) > high_leverage:
                ax.axvline(high_leverage, label='High leverage', ls='-.', color='purple', lw=1)

        ax.axhline(0, ls='dotted', color='black', lw=1.25)
        ax.set_xlim(0, max(self.leverage)+0.01)
        ax.set_ylim(min(self.studresid)-0.1, max(self.studresid)+0.1)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        plt.legend(loc='best')
        return ax
    
    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y
    
    def cookdplot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
    
        ax.vlines(range(len(self.cooks_d)),0,self.cooks_d)

        # annotations
        cookd_top_3 = np.flip(np.argsort(self.cooks_d), 0)[:3]
        for i in cookd_top_3:
            ax.annotate(i, xy=(i, self.cooks_d[i]),color = 'C3')
        
        ax.set_title("Cook's Distance" , fontweight="bold")
        ax.set_xlabel('Obs. Number')
        ax.set_ylabel("Cook's distance")
        return ax
    
    def cookdlevplot(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
    
    
        g = self.leverage/(1-self.leverage)
        ax.scatter(g, self.cooks_d, alpha=0.5)
    
        sns.regplot(x=g, y=self.cooks_d,scatter=False,
                    ci=False, lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                    ax=ax)
        # Label axis with Leverage values
        athat = pretty(self.leverage)
        ax.set_xticks(athat/(1-athat), labels = athat)
    
        bvals = pretty(np.sqrt(self.nparams*self.cooks_d/g))
    
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        for bval in bvals:
            bi2 = bval**2
        
            if ymax > bi2*xmax:
                xi = xmax 
                yi = bi2*xi
                ax.plot([0, xi], [0, yi],'k--' )
                ax.annotate(bval, xy=(xi, yi))
            else:
                yi = ymax 
                xi = yi/bi2
                ax.plot([0, xi], [0, yi],'k--' )
                ax.annotate(bval, xy=(xi, yi))
            
        # annotations
        cookd_top_3 = np.flip(np.argsort(self.cooks_d), 0)[:3]
        for i in cookd_top_3:
            ax.annotate(i, xy=(g[i], self.cooks_d[i]),color = 'C3')
        
        ax.set_title("Cook's Distance vs Leverage $h_{ii}/(1-h_{ii})$" , fontweight="bold")
        ax.set_xlabel("Leverage $h_{ii}$")
        ax.set_ylabel("Cook's Distance")
        return ax
    
def nicenumber(x, round):
    exp = np.floor(np.log10(x))
    f   = x / 10**exp

    if round:
        if f < 1.5:
            nf = 1.
        elif f < 3.:
            nf = 2.
        elif f < 7.:
            nf = 5.
        else:
            nf = 10.
    else:
        if f <= 1.:
            nf = 1.
        elif f <= 2.:
            nf = 2.
        elif f <= 5.:
            nf = 5.
        else:
            nf = 10.

    return nf * 10.**exp

def pretty(x, n=5):
    high = max(x)
    low = min(x)
    r1 = nicenumber(high - low, False)
    d     = nicenumber(r1 / (n+1), True)
    miny  = np.floor(low  / d) * d
    maxy  = np.ceil (high / d) * d
    return np.arange(miny, maxy+0.5*d, d)