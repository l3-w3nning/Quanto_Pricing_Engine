# Quanto_Pricing_Engine
This is one of my pet projects that price a "vanilla" quanto option with payout structure FXRate*max(stockprice-strike,0). 
Until now the pricing "engine" only features one pricing method, namely the copula approach, where a parametric mutual distribution is approximated via a speicif choice of a copula. In this case I opted for a Archimedean Copula (Gumbel) and fitted the copula parameter theta by simple least squares optimization over the relation tau=(theta-1)/theta with the observed market data of a pair of a selected stock and a FX Rate.
The code consists of 3 classes: 1) the first class calculates the calculation steps of the empirical copulas (for now only out of curiosity). This class can later be used to check in how far the estimated parametric copula deviates from the empirical copula (which converges to the true mutual distribution for large samples).
Secondly I implemented a fiting class which not only fits the Gumbel copula but also fits the marginals to later better sample from them in the monte carlo pricing "engine". The script tries to fit all named distritbutions from the stats library and returns the sorted list of fitted distributions (ranked by a goodness of fit parameter)
Lastly I implemented a MonteCarlo Pricer class, which samples from the Copula/Margins and returns an estimated price of the Quanto option.

In the future I want to backtest this approach with "practiticioner" approaches (namely a Black-Scholes and a DSW(Dimitroff-Szimayer-Wagner)-pricing algorithm)
