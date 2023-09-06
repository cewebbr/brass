"""
Functions for generating associated pairs of categorical variables.
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

def compute_cond_probs(dP, Pa, Pb):
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


def gen_cond_sample(Pb, PaNb, PaGb, n_samples, seed=None):
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


def gen_sample(dP, Pa, Pb, n_samples, seed=None):
    """
    Generate a sample of a pair of binary variables A and B given
    \Delta P(A,B), P(A=1) and P(B=1).
    
    Parameters
    ----------
    dP : float
        \Delta P(A,B) = P(A=1|B=1) - P(A=1)
    Pa : float
        P(A=1)
    Pb : float
        P(B=1)
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
    
    # Compute conditional probabilities:
    PaNb, PaGb = compute_cond_probs(dP, Pa, Pb)
    
    # Generate sample:
    ass, bss = gen_cond_sample(Pb, PaNb, PaGb, n_samples, seed)
    
    return ass, bss
