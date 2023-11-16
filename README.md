# BRASS - Categorical variable association estimator

BRASS is a Bayesian method for estimating the degree of association between categorical variables. It is described in the paper 
"_Applying a new category association estimator to sentiment analysis on the Web_" ([Xavier et al. (2023)](https://arxiv.org/abs/2311.05330)),
where it was applied to annotations of emotions identified on tweets in Portuguese.

## Structure of the project:

    .
    ├── README.md               <- This document
	├── LICENSE                 <- The license for this work
    ├── requirements.txt        <- Python packages required to run everything
	├── min_requirements.txt    <- Python packages required by brass.py
	├── brass.py                <- A Python module that implements BRASS as a class
    ├── data                    <- Where data would be stored
    ├── analysis                <- Analysis made for the paper
    |   └── plots               <- Where plots would be saved
    └── examples                <- Jupyter notebooks showing how to use BRASS
    

## Installation

The Python packages required to run BRASS and all the analysis done in the paper are specified in `requirements.txt`. If you just want to use the 
[BRASS module](brass.py), you can run the following command:

      pip install -r min_requirements.txt

Then, just copy the `brass.py` file to a local folder where it can be found.


## Citing this work

If you use the data or the code in this repository, please cite:

@unpublished{Xavier2023,
  author = "Henrique S. Xavier and Diogo Cortiz and Mateus Silvestrin and Ana Lu\'isa Freitas and Let\'icia Yumi Nakao Morello and Fernanda Naomi Pantale\~ao and Gabriel Gaudencio do R\\\^ego",
  title  = "Applying a new category association estimator to sentiment analysis on the Web",
  archivePrefix = {arXiv},
  eprint = {2311.05330},
  primaryClass = {stat.AP},
  month  = "11",
  year   = "2023"
}


## Contact

For more information, contact [Henrique S. Xavier](http://henriquexavier.net) (<https://github.com/hsxavier>).
