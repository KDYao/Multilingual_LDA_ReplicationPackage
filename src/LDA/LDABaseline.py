"""
This file is the evaluation of 10 metrics of the same LDA model
We use the median metric value of 10 models as the baseline for comparison
The purpose of this experiment is to find
"""
import sys
import os
import pandas as pd
import statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.resources.utils.utils import parse_args_baseline_lda, getPath, getWorkers
from src.LDA.RunLDAOpts import OptionLDARunner
from src.LDA.LDAOptions import LDAOptions


class LDABaseLine:
    def __init__(self, num_topics=100, alpha=1.0, beta=0.01, out_dir='../../result/baseline'):
        # We use the default setting which is usually adopted by most studies
        # Where k=100, alpha=1.0 and beta=0.01
        # This is used as a baseline to compare with the tuning result
        # To make the result less bias by noise, we repeat the modeling 10 times
        # and use the median metric to represent the baseline results
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.out_dir = os.path.abspath(
            os.path.join(os.path.abspath(__file__), out_dir)
        )
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    def run(self, args, ldarunner, repeat=10):
        """
        Run LDA modeling according to the specific LDA type 10 times
        Parameters
        ----------
        args

        Returns
        -------
        """
        lda_type = args.LDAType
        args.numTopics = self.num_topics
        args.alpha = self.alpha
        args.beta = self.beta
        if lda_type in [x.name for x in LDAOptions]:
            res = ldarunner.run(lda_type=LDAOptions[lda_type], args=args, paraeval_runs=repeat)
        else:
            raise TypeError('Type %s not defined.' % lda_type)

        df = pd.DataFrame(res)
        df.to_csv(
            os.path.join(self.out_dir, 'LDABaseLine_{}.csv'.format(lda_type)), index=False
        )

        for column in df:
            try:
                print('Median_{k}: {v}'.format(
                    k=column,
                    v=str(statistics.median(df[column].apply(lambda x: float(x))))
                ))
            except ValueError:
                # For LDA2TEXT/LDA2LOG, log-alignment is set to None
                # We might need to remove that metric in the future
                continue
        #res = [dict(**x, **{'LDAType': lda_type}) for x in res]


def clean_LDA2_corpus(args):
    """
    This function removes the LDA2_SEP corpus
    Since LDA2_SEP corpus is based on the result of LDA2TEXT/LDA2LOG
    It's a dynamic corpus which needs to be cleaned every time before tuning
    Parameters
    ----------
    args

    Returns
    -------
    """

    if args.LDAType == LDAOptions.LDA2_SEP.name:
        f_lda2_corpus = os.path.join(*[root, 'documents', 'LDA2_SEP', 'LDA2_documents.pkl'])
        if os.path.isfile(f_lda2_corpus):
            os.remove(f_lda2_corpus)


if __name__ == '__main__':
    args, _ = parse_args_baseline_lda()
    lda_base = LDABaseLine()
    mallet_path = getPath('mallet')
    os.environ.update({
        'MALLET_HOME': os.path.dirname(os.path.dirname(mallet_path))
    })
    root = getPath('root')
    clean_LDA2_corpus(args=args)
    f_corpus_name = getPath('CORPUS_NAME')
    f_data_dict = os.path.join(*[root, 'post-processed-data', f_corpus_name])
    ldarunner = OptionLDARunner(root=root, f_data_dict=f_data_dict, mallet_path=mallet_path, workers=getWorkers())
    lda_base.run(args=args, ldarunner=ldarunner)
