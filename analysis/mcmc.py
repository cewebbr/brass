"""
MCMC functions for analysis of moral and emotional categories
Copyright (C) 2023  Henrique S. Xavier
Contact: hsxavier@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as pl

def count_onehot_pairs(onehot_df):
    """
    Count the number of occurences of each binary pair.

    Parameters
    ----------
    onehot_df : DataFrame
        Two-column (A and B) table with binary values; columns 
        are events and rows are instances.

    Returns
    -------
    counts : array, shape (1, 4)
        Number of instances that match the data pattern (in this 
        order): (0, 0), (0, 1), (1, 0), (1, 1).
    """
    
    # Create index with all possible pair combinations:
    index = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)], names=onehot_df.columns)
    init  = pd.DataFrame(index=index)

    # Join the counts:
    counts_series = onehot_df.value_counts()
    complete = init.join(counts_series).fillna(0).astype(int)
    counts = complete['count'].sort_index().values.reshape((1,4))

    return counts


def create_multinomial_model(counts, alpha=1, prior_name='pair_probs'):
    """
    Create a pymc Bayesian model for the counts, assuming the `counts`
    come from a Multinomial distribution whose probabilities are 
    extracted from a Dirichlet prior.

    Parameters
    --------a--
    counts : array, shape (n, k)
        Observed counts from n categorical samplings (i.e. experiments), 
        each one containing k possible categories.
    alpha : float
        Constant concentration parameter for the Dirichlet prior (all variables
        go to the same power of `alpha - 1`). `alpha = 1` correponds to an 
        uniform distribution subject to the total probability constraint.
    prior_name : str
        How to call the vector of probabilities that will be sampled by the 
        MCMC chain.

    Returns
    -------
    model : pymc.Model
        The model.
    """
    # Data characteristics:
    n_cells      = len(counts[0])   
    total_counts = counts.sum()

    # Create model:
    with pm.Model() as model:
        
        # Prior (distribuição multivariada uniforme sujeita ao vínculo $\sum p_i = 1$):
        pair_probs = pm.Dirichlet(prior_name, np.ones(n_cells) * alpha)
        
        # Cria Likelihood:
        gen_counts = pm.Multinomial('gen_counts', n=total_counts, p=pair_probs, shape=counts.shape, observed=counts)
    
    return model


def sample_bayesian_model(model, n_draws=1000, n_chains=4, n_tune=1000, seed=None, discard_tuned_samples=True, progressbar=True):
    """
    Run an MCMC on the specified model.

    Parameters
    ----------
    model : pymc.Model
        The model to be sampled with MCMC.
    n_draws : int
        Number of points in the MCMC, after burning in.
    n_chains : int
        Number of parallel chains to generate.
    n_tune : int
        Number of point in the chain used to burn in.
    seed : int or None
        Pseudo random number generator seed.
    discard_tuned_samples : bool
        Whether to throw away the burn-in points or not. 
        Either way, they are not concatenated with the 
        burned-in chain.

    Returns
    -------
    idata : arviz.InferenceData
        The chain and associated statistics and parameters.
    """
    with model:
        idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains, random_seed=seed, discard_tuned_samples=discard_tuned_samples, progressbar=progressbar)

    return idata


def idata2df(idata, par_name='pair_probs', par_dim_names=['p00', 'p01', 'p10', 'p11']):
    """
    Transform a MCMC set of chains into a Pandas DataFrame.

    Parameters
    ----------
    idata : Arviz InferenceData
        The chain, the output from pymc model sample() method.
    par_name : str
        The name of the model parameter that was sampled. It is expected 
        to be a vector.
    par_dim_names : list of str
        The names on each dimension in the model parameter mentioned
        above.

    Returns
    -------
    chain_df : DataFrame
        Values of the model parameter vector sampled by the parallel 
        chains, stacked. The burn-in data is already not present
        in `idata`. 
    """
    
    # Get data specs:
    n_chains = idata['posterior'].dims['chain']
    n_draws  = idata['posterior'].dims['draw']
    n_cells  = idata['posterior'].dims[par_name + '_dim_0']
    
    # Stack parallel chains into a single sampling set:
    stacked_chains = np.array(idata['posterior'][par_name]).reshape((n_chains * n_draws, n_cells))
    
    # Create DataFrame:
    chain_df = pd.DataFrame(data=stacked_chains, columns=par_dim_names)

    return chain_df


def add_derived_probs(chain_df):
    """
    Compute marginal and conditional probabilities from joint 
    probabilities of two binary events A and B; and compute the
    differences P(Y|X) - P(Y). Add these columns to `chain_df`.

    Parameters
    ----------
    chain_df : DataFrame
        MCMC chain with sampled parameters:
        p00 = P(A=0,B=0)
        p01 = P(A=0,B=1)
        p10 = P(A=1,B=0)
        p11 = P(A=1,B=1)

    Returns
    -------
    chain_df : DataFrame
        The input is modified inplace and returned.
        Extra columns added are:
        Pa      = P(A=1)
        Pb      = P(B=1)
        P(a|b)  = P(A=1|B=1)
        P(a|~b) = P(A=1|B=0)
        P(b|a)  = P(B=1|A=1)
        P(b|~a) = P(B=1|A=0)
        sPa     = P(A=1|B=1)-P(A=1)
        sPb     = P(B=1|A=1)-P(B=1)
    """
    
    # Compute marginal probabilities of events A and B:
    chain_df['Pa'] = chain_df['p10'] + chain_df['p11']
    chain_df['Pb'] = chain_df['p01'] + chain_df['p11']
    
    # Compute conditional probabilities:
    chain_df['P(a|b)'] = chain_df['p11'] / chain_df['Pb']
    chain_df['P(a|~b)'] = chain_df['p10'] / (1 - chain_df['Pb'])
    chain_df['P(b|a)'] = chain_df['p11'] / chain_df['Pa']
    chain_df['P(b|~a)'] = chain_df['p01'] / (1 - chain_df['Pa'])
    
    # Compute differences in probabilities:
    #chain_df['dPa'] = chain_df['P(a|b)'] - chain_df['P(a|~b)']
    #chain_df['dPb'] = chain_df['P(b|a)'] - chain_df['P(b|~a)']
    chain_df['sPa'] = chain_df['P(a|b)'] - chain_df['Pa']
    chain_df['sPb'] = chain_df['P(b|a)'] - chain_df['Pb']

    return chain_df


def plot_chain_probs(chain_df, bins=50, figsize=(15,8)):
    """
    Plot the distribution of marginal and conditional 
    probabilities estimated from an MCMC chain.

    Parameters
    ----------
    chain_df : DataFrame
        Table with samples (rows) of the quantities 'Pa', 
        'P(a|b)', 'P(a|~b)', 'sPa', 'Pb', 'P(b|a)', 'P(b|~a)', 
        and 'sPb' (columns in `chain_df`) obtained from an 
        MCMC sampling.
    figsize : tuple of floats
        Figure size.
    """
    pl.figure(figsize=figsize)
    for i,c in enumerate(['Pa', 'P(a|b)', 'P(a|~b)', 'sPa', 'Pb', 'P(b|a)', 'P(b|~a)', 'sPb']):
        pl.subplot(2, 4, i + 1)
        pl.title(c)
        chain_df[c].hist(bins=bins)


def compute_pvalue(series, threshold=0):
    """
    Compute the p-value for a variable sample `series` 
    (Series or array) and a `threshold`. Always return
    the probability that is smaller than 0.5.
    """
    p_value = (series < threshold).mean()
    
    if p_value > 0.5:
        p_value = 1 - p_value
    
    return p_value


def create_stats_summary_df(chain_df, featA, featB):
    """
    Compute summary statistics given a MCMC chain.

    Parameters
    ----------
    chain_df : DataFrame
        MCMC chain with columns 'Pa', 'Pb', 'sPa' and 'sPb'.
    featA : str
        Name of feature A.
    featB : str
        Name of feature B.

    Returns
    -------
    summary_stats_df : DataFrame
        Table with summary statistics about the probability of 
        occurences of features A and B.
    """
    
    # Hard-coded:
    cols = ['Direction', 'A', 'B', 'P(A)', 'dev P(A)', 'P(A|B)', 'dev P(A|B)', 'sP(A|B)', 'dev sP(A|B)', 'abs s nsigma', 's p-value']
    
    # Compute stats for A,B:
    cond_prob_mean = chain_df['P(a|b)'].mean()
    cond_prob_dev  = chain_df['P(a|b)'].std()    
    prob_gain_mean = chain_df['sPa'].mean()
    prob_gain_dev  = chain_df['sPa'].std()
    p_value = compute_pvalue(chain_df['sPa'])
    data_f = ['F', featA, featB, chain_df['Pa'].mean(), chain_df['Pa'].std(), cond_prob_mean, cond_prob_dev, prob_gain_mean, prob_gain_dev, np.abs(prob_gain_mean / prob_gain_dev), p_value]
    
    # Compute stats for B,A:
    cond_prob_mean = chain_df['P(b|a)'].mean()
    cond_prob_dev  = chain_df['P(b|a)'].std()    
    prob_gain_mean = chain_df['sPb'].mean()
    prob_gain_dev  = chain_df['sPb'].std()
    p_value = compute_pvalue(chain_df['sPb'])
    data_b = ['B', featB, featA, chain_df['Pb'].mean(), chain_df['Pb'].std(), cond_prob_mean, cond_prob_dev, prob_gain_mean, prob_gain_dev, np.abs(prob_gain_mean / prob_gain_dev), p_value]
    
    # Add to DataFrame:
    data = [data_f, data_b]
    summary_stats_df = pd.DataFrame(columns=cols, data=data)
    
    return summary_stats_df


def mcmc_sample_multinomial_pars(onehot_df, feat1, feat2, chain_size=10000, seed=None, progressbar=True):
    """
    Given observations of a pair of binary features, generate an 
    MCMC chain sampling the Posterior for the multinomial 
    parameters, i.e. the probabilities of ocurrences of each pair.
    
    Parameters
    ----------
    onehot_df : DataFrame
        Instances (rows) of observations of binary features (columns).
    feat1: str
        Name of the column containing data about one feature of the 
        pair.
    feat2: str
        Name of the column containing data about the other feature
        of the pair.
    chain_size : int
        Number of samples to generate in the chain, after the burn-in.
    seed : int or None
        Seed for the pseudo-random number generator.
    progressbar : bool
        Show sampling progress bar or not.
    
    Returns
    -------
    chain_df : DataFrame
        Chain of the sampled values of the Posterior parameters (and
        derived parameters).
    """
    
    # Count pair ocurrences: 
    obs_counts = count_onehot_pairs(onehot_df[[feat1, feat2]])
    
    # Sample multinomial:
    model = create_multinomial_model(obs_counts)
    idata = sample_bayesian_model(model, chain_size, seed=seed, progressbar=progressbar)
    
    # To DataFrame with derived parameters:
    chain_df = idata2df(idata)
    chain_df = add_derived_probs(chain_df)
    
    return chain_df
