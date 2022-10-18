from curses.ascii import EM
from datetime import datetime, date, timedelta
from multiprocessing.dummy import JoinableQueue
from black import err
from scipy.stats._continuous_distns import _distn_names

from more_itertools import sample
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import scipy
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula
import seaborn as sns
import warnings
import scipy.stats as st




class empirical_copula:
    def __init__(self, margin1, margin2, samplesize):

        if (samplesize < len(margin2)) and (samplesize < len(margin1)):
            self.numdata = samplesize
        else:
            raise ValueError("samplesize must be equal or smaller to len of margin(s)")
        self.margin1 = margin1[0:samplesize]
        self.margin2 = margin2[0:samplesize]

    def get_count(
        self, list_to_check1, list_to_check2, variable_to_check1, variable_to_check2
    ):
        '''
        this function calculates all sample pairs smaller than order statistic x_(i),y_(j) 
        returns: count of pairs smaller than one pair of order statistics (variable_to_check_1 and _2)
        '''

        list_to_check1_bool = np.array((list_to_check1 <= variable_to_check1)) * 1
        list_to_check2_bool = np.array((list_to_check2 <= variable_to_check2)) * 1
        #with the dot product one can single out all sample pairs smaller than given pair of orderstatistic (if one of the samples is larger than dot-contribution amounts to zero)
        count = np.dot(list_to_check1_bool, list_to_check2_bool)

        return count

    def get_empirical_copula(self):
        n = self.numdata
        C = np.ones((n, n))
        margin1_ordered = self.margin1.sort()
        margin2_ordered = self.margin2.sort()
        for i in range(n):
            print(i)
            for j in range(n):
                C[i][j] = (
                    1/n*get_count(
                        self.margin1,
                        self.margin2,
                        margin1_ordered[i],
                        margin2_ordered[j],
                    )
                )

        return C


class fitting:
    def __init__(self, margin1, margin2, samplesize):
        if (samplesize < len(margin2)) and (samplesize < len(margin1)):
            self.numdata = samplesize
        else:
            raise ValueError("samplesize must be equal or smaller to len of margin(s)")
        self.margin1 = margin1[0:samplesize]
        self.margin2 = margin2[0:samplesize]
        tau, _ = scipy.stats.kendalltau(self.margin1, self.margin2)
        self.tau = tau

    def get_theta(self, theta):
        """
        returns: difference of kendalls tau of margins and a tau of a to be fitted gumbel copula
        """
        return abs(self.tau - (theta - 1) / theta)

    def calibrate_copula(self):
        """
        calibrates the copula with a pseudo mle estimator over kendalls tau (via relation tau=(theta-1)/theta)
        in such a way one does not have to calculate empirical copula (costly), emp copula still can be used for backtesting later
        returns: theta value of Gumbel copula (closest fit to computed tau of margins)
        """
        bound = (1, 10)
        res = scipy.optimize.minimize_scalar(
            fun=self.get_theta, bounds=bound, method="bounded"
        )  # truncated newton for bounded problems
        theta = res.x
        return theta

    def best_fit_distribution(self, data, bins=200, ax=None):
        """
        in order to draw samples from margins it is easier to use parametric margins. these are fitted to the margin market data
        note that the copula however is fitted with the observed data
        code used from https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
        returns: list of all named fit distributions of scipy.stats, ordered by goodness of fit parameter (decreasing)
        """
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Best holders
        best_distributions = []

        # Estimate distribution parameters from data
        for ii, distribution in enumerate(
            [d for d in _distn_names if not d in ["levy_stable", "studentized_range"]]
        ):

            print("{:>3} / {:<3}: {}".format(ii + 1, len(_distn_names), distribution))

            distribution = getattr(st, distribution)

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                    except Exception:
                        pass

                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))

            except Exception:
                pass
        best_distributions = sorted(best_distributions, key=lambda x: x[2])
        best_dist = best_distributions[0]
        best_dist_name = best_dist[0]
        best_dist_params = best_dist[1]
        return best_dist_name, best_dist_params


class monte_carlo_pricer_copula:
    def __init__(
        self,
        estimated_copula,
        margin1_fit_name,
        margin2_fit_name,
        margin1_fit_params,
        margin_2_fit_params,
        strike,
        n,
    ):
        self.estimated_copula = estimated_copula
        self.margin1_fit_name = margin1_fit_name
        self.margin2_fit_name = margin2_fit_name
        self.margin1_fit_params = margin1_fit_params
        self.margin2_fit_params = margin_2_fit_params
        self.strike = strike
        self.numsims = n

    def get_simulations(self):
        """
        #Algorithm to sample from Gumbel copula, already implemented by package statsmodels, however nice to have
        #based off Paper by Marius Hofert "Sampling Archimedean Copulas"
        levy_stable_sample=st.levy_stable.rvs(alpha=1/theta, beta=1, loc=0, scale=np.cos(np.pi/(2*theta))**theta, size=1, random_state=None)
        copula_samples=[]
        for i in range(1000):
            uniform_samples=st.uniform.rvs(size=2)
            gumbel_copula_sample_1=np.exp(-(-np.log(uniform_samples[0])/levy_stable_sample)**(1/theta))
            gumbel_copula_sample_2=np.exp(-(-np.log(uniform_samples[1])/levy_stable_sample)**(1/theta))
            copula_samples.append(zip(gumbel_copula_sample_1,gumbel_copula_sample_2))
            #diese sims noch in die inversen margins stecken!
            #über ppf methode der jeweiligen verteilung
        """
        arg1 = self.margin1_fit_params[:-2]
        loc1 = self.margin1_fit_params[-2]
        scale1 = self.margin1_fit_params[-1]

        arg2 = self.margin2_fit_params[:-2]
        loc2 = self.margin2_fit_params[-2]
        scale2 = self.margin2_fit_params[-1]
        # oder das der pricing engine übergeben (ist eleganter)
        marginals = [
            self.margin1_fit_name(loc=loc1, scale=scale1, *arg1),
            self.margin2_fit_name(loc=loc2, scale=scale2, *arg2),
        ]  # die params müssen anders übergeben werden!, vllt direkt schon funktion der besten verteilung übergeben! (und nicht nur name phne param)

        joint_dist = CopulaDistribution(
            copula=self.estimated_copula, marginals=marginals
        )
        sample = joint_dist.rvs(self.numsims, random_state=20210801)
        h = sns.jointplot(x=sample[:, 0], y=sample[:, 1], kind="scatter")
        _ = h.set_axis_labels("USD/EUR", "ROCHE Stock (USD)", fontsize=16)
        plt.show()
        return sample

    def vanilla_quanto(self, FX, stockprice):
        """
        returns: simulated price of quanto option
        """
        return 1 / FX * max(stockprice - self.strike, 0)

    def montecarlo_price(self):
        """
        returns: mean of simulated prices
        """
        simulations = self.get_simulations()
        FX_prices = simulations[:, 0]
        StockPrices = simulations[:, 1]

        option_prices = []
        plt.plot(StockPrices)
        plt.show()
        for i in range(len(FX_prices)):
            
            print(StockPrices[i]-self.strike)
            option_prices.append(
                self.vanilla_quanto(FX_prices[i], StockPrices[i])
            )
        return np.mean(option_prices)


def main():
    today = date.today()
    start_date = today - timedelta(4000)
    end_date = today

    FX_rate = "USDEUR=X"
    data_fx = yf.download(FX_rate, start_date, end_date)
    data_fx_open = np.array(data_fx["Open"])

    stock = "ROG.SW"  # roche

    data_stock = yf.download(stock, start_date, end_date)
    data_stock_open = np.array(data_stock["Open"])

    # Empirical Copula not needed as of now, one can however in the future compute it and calculate the difference to the fitted copula
    # copula=empirical_copula(data_stock_open,data_fx_open,300)

    fit = fitting(data_fx_open, data_stock_open, samplesize=300)
    
    theta = fit.calibrate_copula()
    estimated_copula = GumbelCopula(theta=theta)
    margin1_fit_name, margin1_fit_params = fit.best_fit_distribution(fit.margin1)
    margin2_fit_name, margin2_fit_params = fit.best_fit_distribution(fit.margin2)
   
    data_stock_open = data_stock_open[0:300]#just to get a sensible strike value (in this case arbitrarily set to mean of first 300 market data points (also used for calibration))
    strike = np.mean(data_stock_open)
    pricing_engine = monte_carlo_pricer_copula(
        estimated_copula=estimated_copula,
        margin1_fit_name=margin1_fit_name,
        margin1_fit_params=margin1_fit_params,
        margin2_fit_name=margin2_fit_name,
        margin_2_fit_params=margin2_fit_params,
        strike=strike,
        n=1000,
    )
    price_quanto = pricing_engine.montecarlo_price()
    print(
        "The price of a FX quanto call on",
        stock,
        " with strike ",
        strike,
        " and underlying payout currency being denoted in EUR amounts to",
        price_quanto,
    )

    return None


if __name__ == "__main__":
    main()
