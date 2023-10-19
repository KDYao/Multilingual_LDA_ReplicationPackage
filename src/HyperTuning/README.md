 
# How to run opentuner?

Here is the user guide: https://opentuner.org/tutorial/gettingstarted/

The hyperparameters are: 
- Number of Topics (K)
- Dirichlet hyperparameter alpha: Document-Topic Density
- Dirichlet hyperparameter beta: Word-Topic Density

The source code for executing hyperparameters tuning is in `src/HyperTuning/HyperParamOptimization.py`

Based on the default arg set of OpenTuner, we append several args based on our testing environment. The added inputs arguments:

- LDAType (required): The type of LDA model. It should be either LDA1, LDA2, or LDA3.
- languages (required): The languages to be considered when analyzing. Separate languages by comma.
- numTopicsRange (default 5-10): The range of the selection of number of topics.
- alphaRange (default 1.0-30.0): The range of the selection of alpha.
- betaRange(default 0.01-5.00): The range of the selection of beta.
- metric (default 2): The evaluation metric index
    - COHERENCE = 1 
    - PERPLEXITY = 2 
    - LOGALIGNMENT = 3 
    - INTERCORPUSRATIO = 4 
    - AGGREGATED = 5

