import os
import warnings
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import scipy
from scipy.special import factorial
import math

def fitContinuousDistribution(data, bins=100):
    y, x = np.histogram(data, bins=bins, density=True)
    # Get middles of the bins
    x = (x + np.roll(x, -1))[:-1] / 2.0
    distribs = [stats.alpha,stats.anglit,stats.arcsine,stats.beta,stats.betaprime,stats.bradford,stats.burr,stats.cauchy,stats.chi,stats.chi2,stats.cosine,
            stats.dgamma,stats.dweibull,stats.erlang,stats.expon,stats.exponnorm,stats.exponweib,stats.exponpow,stats.f,stats.fatiguelife,stats.fisk,
            stats.foldcauchy,stats.foldnorm,stats.frechet_r,stats.frechet_l,stats.genlogistic,stats.genpareto,stats.gennorm,stats.genexpon,
            stats.genextreme,stats.gausshyper,stats.gamma,stats.gengamma,stats.genhalflogistic,stats.gilbrat,stats.gompertz,stats.gumbel_r,
            stats.gumbel_l,stats.halfcauchy,stats.halflogistic,stats.halfnorm,stats.halfgennorm,stats.hypsecant,stats.invgamma,stats.invgauss,
            stats.invweibull,stats.johnsonsb,stats.johnsonsu,stats.kstwobign,stats.laplace,stats.levy,
            stats.logistic,stats.loggamma,stats.loglaplace,stats.lognorm,stats.lomax,stats.maxwell,stats.mielke,stats.nakagami,stats.ncx2,stats.ncf,
            stats.nct,stats.norm,stats.pareto,stats.pearson3,stats.powerlaw,stats.powerlognorm,stats.powernorm,stats.rdist,stats.reciprocal,
            stats.rayleigh,stats.rice,stats.recipinvgauss,stats.semicircular,stats.t,stats.triang,stats.truncexpon,stats.truncnorm,stats.tukeylambda,
            stats.uniform,stats.vonmises,stats.vonmises_line,stats.wald,stats.weibull_min,stats.weibull_max,stats.wrapcauchy]
    i = 0
    minVal = (1000.0, 0)
    best_params = []
    for d in distribs:
         # fit dist to data
         with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            params = d.fit(data)
            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            pdf = d.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))
            print("Index [{}] {} SSE: {}, params: {} ".format(i, distribs[i], sse, params))
            if sse < minVal[0]:
                minVal = (sse, i)
                best_params = params
            i += 1
    print(minVal)
    params = best_params
    loc = params[-2]
    scale = params[-1]
    print(loc)
    print(scale)


def fitPaymentTypeDistribution(data):
    factorised, categories = pd.factorize(data)
    print(factorised)
    #distribs = [stats.alpha, stats.binom]
    y, _ = np.histogram(factorised, bins=2)
    x = [0.0, 1.0]
    params = stats.binom.fit(factorised)
    print(params)
if __name__ == "__main__":
    fare = pd.read_csv('./data/clean_1_fare_data_4_2013.csv', skipinitialspace=True)

    fare_amount = np.array(fare.fare_amount)
    tip_amount = np.array(fare.tip_amount)
    total_amount = np.array(fare.total_amount)
    payment_type = np.array(fare.payment_type)

    fitContinuousDistribution(fare_amount)
    fitContinuousDistribution(tip_amount)
    fitContinuousDistribution(total_amount)
