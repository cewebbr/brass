"""
Bayesian estimator for the association between categorical variables
Copyright (C) 2023  Henrique S. Xavier
Contact: hsxavier@gmail.com

If you use this method, please cite the publication:
    Xavier, H. S. et al. (2023), "Applying a new category association 
    estimator to sentiment analysis on the Web"
    https://arxiv.org/abs/2311.05330
    
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


class AssocEstimator:
    """
    Create an association estimator for a pair of binary variables.
    
    Parameters
    ----------
    pair_counts : array, shape (k,), or None
        Observed counts for k possible categories. Currently, k must be 4 
        since a pair of binary variables can have the values: (0,0), (0,1), 
        (1,0), (1,1). If None, then `onehot_trials` must be provided.
    onehot_trials : DataFrame, array, or None
        Two-column (A and B) table with binary values; columns are variables 
        and rows are instances. If array, must have the shape (N, 2), where 
        N is the number of observations. Should only be provided if 
        `pair_counts` is not provided.
    prior_alpha : float or array
        Parameter alpha for the Dirichlet distribution, used as prior. It 
        must be an array of 4 positive values, each associated to the 
        probability of the results (0,0), (0,1), (1,0), (1,1), respectively.
        If float, all variables go to the same power of `prior_alpha - 1`, and 
        it is called "concentration parameter". `alpha = 1` correponds to an 
        uniform distribution subject to the total probability constraint.
    verbose : Whether to print log messages while employing this object's 
        methods.    
    """

    
    def __init__(self, pair_counts=None, onehot_trials=None, prior_alpha=1, 
                 verbose=False):
        
        # Check basic input:
        assert isinstance(verbose, bool), '`verbose` must be boolean.'
        
        # Save basic input:
        self.verbose = verbose
        
        # Only one data format can be specified:
        if (type(pair_counts) == type(None)) and (type(onehot_trials) == type(None)):
            raise Exception('Either `pair_counts` or `onehot_trials` must be specified.')
        if (type(pair_counts) != type(None)) and (type(onehot_trials) != type(None)):
            raise Exception('One cannot specify both `pair_counts` and `onehot_trials`.')
        
        # If X (trials) is specified, set pair counts:
        if type(onehot_trials) != type(None):
            self.pair_counts = self._count_onehot_pairs(onehot_trials)    
        else:
            self.pair_counts = pair_counts
                
        # Pair counts check and standarization:
        assert type(self.pair_counts) in {list, tuple, np.ndarray}, 'Input `pair_counts` must be list, tuple or array.'
        if (type(self.pair_counts) == list) or (type(self.pair_counts) == tuple):
            # Convert to array:
            self.pair_counts = np.array(self.pair_counts)                
        assert self.pair_counts.shape == (4,), 'Input `pair_counts` must have the shape (4,) and contain 4 int entries: the counts for (0,0), (0,1), (1,0), (1,1).'
        
        # Derived parameters:
        self.n_cells = len(self.pair_counts)
        self.total_counts = self.pair_counts.sum()
        
        # Standarization of the prior parameter:
        self.prior_alpha = self._std_prior_alpha(prior_alpha, self.n_cells)

        # Create bayesian model:
        self.model = self._create_multinomial_model(self.pair_counts, self.prior_alpha)
        
        
    def _check_onehot(self, onehot_trials):
        """
        Verify if `onehot_trials` have the necessary properties. It must be a
        table with 2 columns and N > 0 rows containing only 0s and 1s.
        """
        
        # Check type and shape:
        assert type(onehot_trials) in {pd.DataFrame, np.ndarray}, 'Input `onehot_trials` must be a numpy array or a Pandas DataFrame.'
        assert onehot_trials.shape[1] == 2, 'Input `onehot_trials` should have two columns, one for each binary variable.'
        assert onehot_trials.shape[0] > 0, 'Input `onehot_trials` should have at least one trial (i.e. one row).'
        
        # Standardize type to test its values:
        if type(onehot_trials) == np.ndarray:
            test_df = pd.DataFrame(data=onehot_trials, columns=['A', 'B'])
        else:
            test_df = onehot_trials
        # Check if values are binary:
        assert ((test_df == 1) | (test_df == 0)).all(axis=None), 'Input `onehot_trials` should contain only 0s and 1s.'
    
        
    def _count_onehot_pairs(self, onehot_df):
        """
        Count the number of occurences of each binary pair.

        Parameters
        ----------
        onehot_df : DataFrame
            Two-column (A and B) table with binary values; columns 
            are events and rows are instances.

        Returns
        -------
        counts : array, shape (4, )
            Number of instances that match the data pattern (in this 
            order): (0, 0), (0, 1), (1, 0), (1, 1).
        """
        
        # Check input:
        self._check_onehot(onehot_df)
        
        # Standardize input:
        assert type(onehot_df) in {np.ndarray, pd.DataFrame}, '`onehot_df` should be a numpy array or a Pandas dataframe.' 
        if type(onehot_df) == np.ndarray:
            onehot_df = pd.DataFrame(data=onehot_df, columns=['A', 'B'])
        
        # Create index with all possible pair combinations:
        index = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)], names=onehot_df.columns)
        init  = pd.DataFrame(index=index)

        # Join the counts:
        counts_series = onehot_df.value_counts()
        complete = init.join(counts_series).fillna(0).astype(int)
        counts = complete['count'].sort_index().values.reshape((1,4))

        return counts[0]


    def _std_prior_alpha(self, prior_alpha, n_cells):
        """
        Standardize Dirichlet distribution parameter `prior_alpha` to an array 
        of `n_cells` entries, in case `prior_alpha` is int, float, list or 
        tuple.
        
        Returns a numpy array of shape `(n_cells,)`.
        """
        
        # Check input type:
        assert isinstance(prior_alpha, (int, float, np.ndarray, list, tuple)), '`prior_alpha` should be integer or list-like.'
        # Constant alpha:
        if isinstance(prior_alpha, (int, float)):
            out_alpha = np.ones(n_cells) * prior_alpha
        # List or tuple:
        elif isinstance(prior_alpha, (list, tuple)):
            out_alpha = np.array(prior_alpha)
        # Array:
        else:
            out_alpha = prior_alpha
        # Security check:
        assert out_alpha.shape == (n_cells,), '`prior_alpha` must have shape (4,).'

        return out_alpha
    

    def _create_multinomial_model(self, pair_counts, prior_alpha, prior_name='pair_probs'):
        """
        Create a pymc Bayesian model for the counts, assuming the `counts`
        come from a Multinomial distribution whose probabilities are 
        extracted from a Dirichlet prior.
    
        Parameters
        ----------
        pair_counts : array, shape (k,)
            Observed counts for k possible categories. Currently, k must be 4 
            since a pair of binary variables can have the values: (0,0), (0,1), 
            (1,0), (1,1).
        prior_alpha : float or array
            Parameter alpha for the Dirichlet distribution, used as prior. It 
            must be an array of 4 positive values, each associated to the 
            probability of the results (0,0), (0,1), (1,0), (1,1), respectively.
            If float, all variables go to the same power of `prior_alpha - 1`, and 
            it is called "concentration parameter". `alpha = 1` correponds to an 
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
        total_counts = pair_counts.sum()
    
        # Create model:
        with pm.Model() as model:
            
            # Prior (distribuição multivariada uniforme sujeita ao vínculo $\sum p_i = 1$):
            pair_probs = pm.Dirichlet(prior_name, prior_alpha)
            
            # Cria Likelihood:
            obs_counts = pair_counts.reshape(1, len(pair_counts))
            gen_counts = pm.Multinomial('gen_counts', n=total_counts, p=pair_probs, shape=obs_counts.shape, observed=obs_counts)
        
        return model
        

    def _run_mcmc(self, model, n_draws=1000, n_chains=4, n_tune=1000, seed=None, discard_tuned_samples=True, progressbar=True):
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
        
        # Check input parameters:
        NoneType = type(None)
        assert isinstance(n_draws, int), '`n_draws` must be an int.'
        assert n_draws > 0, '`n_draws` must be positive.'
        assert isinstance(n_chains, int), '`n_chains` must be an int.'
        assert n_chains > 0, '`n_chains` must be positive.'
        assert isinstance(n_tune, int), '`n_tune` must be an int.'
        assert n_tune > 0, '`n_tune` must be positive.'
        assert isinstance(discard_tuned_samples, bool), '`discard_tuned_samples` must be boolean.'
        assert isinstance(progressbar, bool), '`progressbar` must be boolean.'
        assert isinstance(seed, (NoneType, int)), '`seed` must be None or an int.'

        #self.n_draws = n_draws
        #self.n_chains = n_chains
        #self.n_tune = n_tune
        #self.discard_tuned_samples = discard_tuned_samples
        #self.progressbar = progressbar
        #self.seed = seed

        with model:
            idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains, random_seed=seed, discard_tuned_samples=discard_tuned_samples, progressbar=progressbar)
    
        return idata


    def _idata2df(self, idata, par_name='pair_probs', par_dim_names=['p00', 'p01', 'p10', 'p11']):
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


    def _add_derived_probs(self, chain_df):
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
            delPa   = P(A=1|B=1)-P(A=1)
            delPb   = P(B=1|A=1)-P(B=1)
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
        chain_df['delPa'] = chain_df['P(a|b)'] - chain_df['Pa']
        chain_df['delPb'] = chain_df['P(b|a)'] - chain_df['Pb']
    
        return chain_df


    def sample_model(self, n_draws=2500, n_chains=4, n_tune=1000, seed=None, 
                     discard_tuned_samples=True, progressbar=True):
        """
        Given the observations of a pair of binary features, generate an 
        MCMC chain sampling the Posterior for the multinomial 
        parameters, i.e. the probabilities of ocurrences of each pair.
        
        Parameters
        ----------
        n_draws : int
            Number of points in an MCMC chain, after burning in.
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
        progressbar : bool
            Show sampling progress bar or not.

        Returns
        -------
        chain_df : DataFrame
            Chain of the sampled values of the Posterior parameters (and
            derived parameters).
        """
        
        idata = self._run_mcmc(self.model, n_draws, n_chains, n_tune, seed, discard_tuned_samples, progressbar)
        
        # To DataFrame with derived parameters:
        chain_df = self._idata2df(idata)
        chain_df = self._add_derived_probs(chain_df)
        
        return chain_df


    def plot_chain_probs(self, chain_df, bins=40, figsize=(15,8), alpha=0.4, label=None):
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
        if figsize is not None:
            pl.figure(figsize=figsize)
        for i,c in enumerate(['Pa', 'P(a|b)', 'P(a|~b)', 'delPa', 'Pb', 'P(b|a)', 'P(b|~a)', 'delPb']):
            pl.subplot(2, 4, i + 1)
            pl.title(c)
            chain_df[c].hist(bins=bins, alpha=alpha, label=label)


    def compute_pvalue(self, series, threshold=0):
        """
        Compute the p-value for a variable sample `series` (Series or array) 
        and a `threshold`. Always return the probability that is smaller than 
        0.5. 
        
        If `series` is a sampling of deltaP(A,B) = P(A=1|B=1) - P(A=1), then 
        the returned `p_value` value asserts if A and B can be considered 
        dependent (e.g. if `p_value < 0.05` or so).
        
        In case you run this test multiple times, be sure to correct for the
        Multiple Comparisons Effect.
        """
        p_value = (series < threshold).mean()
        
        if p_value > 0.5:
            p_value = 1 - p_value
        
        return p_value


    def summarize_stats(self, chain_df, featA, featB):
        """
        Compute summary statistics given a MCMC chain.
    
        Parameters
        ----------
        chain_df : DataFrame
            MCMC chain with columns 'Pa', 'Pb', 'delPa' and 'delPb'.
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
        cols = ['Direction', 'A', 'B', 'P(A)', 'dev P(A)', 'P(A|B)', 'dev P(A|B)', 'deltaP(A,B)', 'dev deltaP(A,B)', 'abs delta nsigma', 'delta p-value']
        
        # Compute stats for A,B:
        cond_prob_mean = chain_df['P(a|b)'].mean()
        cond_prob_dev  = chain_df['P(a|b)'].std()    
        prob_gain_mean = chain_df['delPa'].mean()
        prob_gain_dev  = chain_df['delPa'].std()
        p_value = self.compute_pvalue(chain_df['delPa'])
        data_f = ['F', featA, featB, chain_df['Pa'].mean(), chain_df['Pa'].std(), cond_prob_mean, cond_prob_dev, prob_gain_mean, prob_gain_dev, np.abs(prob_gain_mean / prob_gain_dev), p_value]
        
        # Compute stats for B,A:
        cond_prob_mean = chain_df['P(b|a)'].mean()
        cond_prob_dev  = chain_df['P(b|a)'].std()    
        prob_gain_mean = chain_df['delPb'].mean()
        prob_gain_dev  = chain_df['delPb'].std()
        p_value = self.compute_pvalue(chain_df['delPb'])
        data_b = ['B', featB, featA, chain_df['Pb'].mean(), chain_df['Pb'].std(), cond_prob_mean, cond_prob_dev, prob_gain_mean, prob_gain_dev, np.abs(prob_gain_mean / prob_gain_dev), p_value]
        
        # Add to DataFrame:
        data = [data_f, data_b]
        summary_stats_df = pd.DataFrame(columns=cols, data=data)
        
        return summary_stats_df
    

class VarGenerator:
    """
    A generator of samples of two binary variables A and B that can be 
    associated (if `delPa != 0`) or not.
    
    The constraints on the input probabilities `delPa`, `Pa` and `Pb` are
    checked internally.
    
    Parameters
    ----------
    delPa : float
        The probability boost on variable A given variable B was already 
        detected: delta P(A,B) = P(A=1|B=1) - P(A=1).
        Must be in a range more restricted than `-1 < delPa < 1`.
    Pa : float
        P(A=1), the probability of observing A=1.
    Pb : float
        P(B=1), the probability of observing B=1.
    """

    
    def __init__(self, delPa, Pa, Pb):
        
        self.PaNb, self.PaGb = self._compute_cond_probs(delPa, Pa, Pb)
        self.delPa = delPa
        self.Pa = Pa
        self.Pb = Pb
        
        
    def _compute_cond_probs(self, dP, Pa, Pb):
        """
        Return the conditional probabilities P(A=1|B=0) and P(A=1|B=1)
        given \Delta P(A,B), P(A) and P(B).
        """
        
        PaGb = dP + Pa                     # P(A|B)
        PaNb = (Pa - PaGb * Pb) / (1 - Pb) # P(A|notB)
    
        # Sanity checks:
        assert -1 <= dP <= 1
        assert  0 <= Pa <= 1
        assert  0 <= Pb <= 1
        assert  0 <= PaGb <= 1, 'P(A=1|B=1) is {:.4f}.'.format(PaGb)
        assert  0 <= PaNb <= 1, 'P(A=1|B=0) is {:.4f}.'.format(PaNb)
        
        return PaNb, PaGb
    
    
    def _gen_cond_sample(self, Pb, PaNb, PaGb, n_samples, seed=None):
        """
        Generate a sample of a pair of binary variables A and B given
        P(B=1), P(A=1|B=0) and P(A=1|B=1).
        
        Parameters
        ----------
        Pb : float
            P(B=1)
        PaNb : float
            P(A=1|B=0)
        PaGb : float
            P(A=1|B=1)
        n_samples : int
            Number of instances in the sample.
        seed : int or None
            Seed for the pseudo random number generator.
        
        Returns
        -------
        ass : array
            Instances of A.
        bss : array
            Associated instances of B.
        """
        
        # Init random number generator:
        rng = np.random.default_rng(seed)
    
        # Generate Bs:
        bss = (rng.random(n_samples) < Pb).astype(int)
    
        # Generate As for B=0:
        asb0 = (rng.random(n_samples) < PaNb).astype(int)
        # Generate As for B=1:
        asb1 = (rng.random(n_samples) < PaGb).astype(int)
        # Combine the parallel worlds:
        ass = np.where(bss == 1, asb1, asb0)
    
        return ass, bss
    
    
    def sample(self, n_samples, seed=None):
        """
        Generate a sample of a pair of binary variables A and B given
        \Delta P(A,B), P(A=1) and P(B=1).
        
        Parameters
        ----------
        n_samples : int
            Number of instances in the sample.
        seed : int or None
            Seed for the pseudo random number generator.
        
        Returns
        -------
        ass : array
            Instances of A.
        bss : array
            Associated instances of B.
        """
                
        # Generate sample:
        ass, bss = self._gen_cond_sample(self.Pb, self.PaNb, self.PaGb, n_samples, seed)
        
        return ass, bss