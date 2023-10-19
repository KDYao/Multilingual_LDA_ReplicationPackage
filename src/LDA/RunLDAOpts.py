#!/usr/bin/env python
import itertools
import os
import sys
import datetime
import logging

# These settings are preserved for Pyinstaller
# Since this file will be running as executable file, we need to manually set the working directory
#hiddenimports = ["gensim", "numpy", "nltk", "python-Levenshtein", "beautifulsoup4"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable wrapper and save it locally, since its not supported by gensim since version 4.0
# from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
# from gensim.models.wrappers import LdaMallet
from src.LDA.LDAOptions import LDAOptions
from src.resources.utils.utils import getWorkers, EvaluationMetrics
import src.resources.utils.utils as ut
from src.LDA.LDA import LDA
from src.dataprocessing.BuildCorpus import CorpusBuilder

log = logging.getLogger(__name__)

class OptionLDARunner:
    def __init__(self, root, f_data_dict, mallet_path, num_topics=11, workers=4):
        self.root = root
        self.f_data_dict = f_data_dict
        self.num_topics = num_topics
        self.mallet_path = mallet_path
        self.workers = workers
        # Define the corpus path for each lda_type
        self.doc_dir = os.path.join(self.root, 'documents')
        # Corpus builder, in case not corpus picked not exists
        self.cor_builder = CorpusBuilder(root=self.root, f_data_dict=self.f_data_dict)

    def run(self, lda_type, args):
        """
        Run LDA according to selected types
        Parameters
        ----------
        lda_type: type of LDA model
        args: input arguments
        Returns
        -------

        """
        start_time = datetime.datetime.now()

        if hasattr(args, 'metric'):
            # If metric used for optimization is defined
            metric_type = EvaluationMetrics(args.metric).name
        else:
            metric_type = None

        # Repeat the test how many times
        # Setup according to the type of runs: baseline or paraeval
        # Disabled Now
        repeats = None
        exp_type = None

        # LDA1: Use all.pkl
        # Other LDA2s: Use every language.pkl
        languages = [x.strip() for x in args.languages.split(',')]

        # Generate corpus
        self.cor_builder.createCorpus(doc_dir=self.doc_dir, languages=languages)

        res = self.run_lda(metric_type=metric_type, languages=languages, lda_type=lda_type, args=args)
        # Log elapsed time
        log.debug('Elapsed time for an iteration of %s is %s', (lda_type.name, str(datetime.datetime.now()-start_time)))
        print('Elapsed time for an iteration of %s is %s' % (lda_type.name, str(datetime.datetime.now()-start_time)))
        return res

    def run_lda(self, languages, lda_type, args, metric_type=None, queue=None, exp_id=None, workers=None, exp_type=None):
        start_time = datetime.datetime.now()
        if workers is None:
            workers = self.workers

        if lda_type == LDAOptions.LDA2:
            documents = self.generate_lda2_docs(
                languages=languages,
                args=args,
                metric_type=metric_type
            )
            print('Elapsed time for generating LDA2TEXT and LDA2LOG docs: %s' % str(datetime.datetime.now() - start_time))
            # Since each topic has limited number of terms, we will skip the filtering frequent/infrequent terms here
            (corpus, dictionary) = self.cor_builder.generate_corpus_dictionary(
                input_documents=documents, isFilter=False
            )
            data_corpus = (documents, corpus, dictionary)
        else:
            data_corpus = None

        lda_runner = LDA(root=self.root, mallet_path=self.mallet_path,
                         topicNum=args.numTopics, alpha=args.alpha,
                         beta=args.beta, lda_type_str=lda_type.name, workers=workers,
                         metric_type=metric_type, exp_id=exp_id, exp_type=exp_type)

        f_corpus = None
        if lda_type == LDAOptions.LDA1:
            f_corpus = os.path.join(self.doc_dir, '%s.pkl' % '_'.join(languages))
        elif lda_type == LDAOptions.LDA3:
            f_corpus = tuple([os.path.join(self.doc_dir, x + '.pkl') for x in languages])

        if hasattr(args, 'saveLDAVis'):
            res = lda_runner.run(f_corpus=f_corpus, data_corpus=data_corpus, saveLDAVis=args.saveLDAVis)
        else:
            res = lda_runner.run(f_corpus=f_corpus, data_corpus=data_corpus)

        if queue is not None:
            queue.put(res)
        # Print result
        # According to Irace documentation, the target-runner script must return only one number.
        # print(lda_runner.aggregated)
        print('Elapsed time for LDA modeling of %s is %s' % (lda_type.name, str(datetime.datetime.now() - start_time)))
        return res

    def generate_lda2_docs(self, languages, args, metric_type):
        """
        Before running LDA2, first run LDA2TEXT and LDA2LOG with the given params and get top terms
        Parameters
        ----------
        languages: The languages to be considered

        Returns
        -------
        """
        top_terms = []
        for lan in languages:
            lda_runner = LDA(root=self.root, mallet_path=self.mallet_path,
                             topicNum=args.numTopics, alpha=args.alpha,
                             beta=args.beta, lda_type_str='LDA2_%s' % lan, isTemp=True, metric_type=metric_type)
            top_terms.append(lda_runner.run(
                f_corpus=os.path.join(self.doc_dir, lan + '.pkl'), topTopicsOnly=True
            ))

        # The total length should be same
        lda2_terms = []
        for idx in range(0, len(top_terms[0])):
            lda2_terms.append(list(itertools.chain.from_iterable([x[idx] for x in top_terms])))
        #return [terms_log + terms_text for terms_log, terms_text in zip(top_terms[0], top_terms[1])]
        return lda2_terms

    def generate_lda2_docs_from_models(self, f_models: list, f_docs: list, ldatypes):
        """
        Generate LDA2_SEP from previous runs
        Parameters
        ----------
        f_models
        f_docs
        ldatypes

        Returns
        -------

        """
        import pickle
        for f_mod, f_doc, lda_type in zip(f_models, f_docs, ldatypes):
            # Load pickled model files
            with open(f_mod, 'rb') as handle:
                ldaModel = pickle.load(handle)
            # Load pickled documents
            with open(f_doc, 'rb') as handle:
                doc, corpus, dic = pickle.load(handle)

            top_documments = LDA.format_topics_sentences(
                ldamodel=ldaModel, corpus=corpus)
            f_top_docs = os.path.join(
                    *[os.path.dirname(f_mod), 'top_docs_list.pkl'.format(lda_type.name)])
            # Top doc output
            with open(f_top_docs, 'wb') as pk_out:
                pickle.dump(top_documments, pk_out)


    def generate_lda2_docs_from_paramsets(self, f_opt_param, corpus_path_dict, metric_type):
        """
        Load the best paramsets from LDA2TEXT and LDA2LOG
        Rerun these two LDA2_SEP models and generate top topics at runtime
        Parameters
        ----------
        f_opt_param
        corpus_path_dict
        metric_type

        Returns
        -------

        """
        import json
        if not os.path.isfile(f_opt_param):
            raise FileNotFoundError('Optimal configuration file not found: %s' % f_opt_param)
        with open(f_opt_param, 'r') as json_file:
            opt_param_dict = dict(json.load(json_file))
        for ldatype in [LDAOptions.LDA2TEXT, LDAOptions.LDA2LOG]:
            param_dict = opt_param_dict[ldatype.name]
            lda_runner = LDA(root=self.root, mallet_path=self.mallet_path,
                             topicNum=param_dict['numTopics'], alpha=param_dict['alpha'],
                             beta=param_dict['beta'], lda_type_str=ldatype.name, workers=self.workers,
                             metric_type=metric_type)
            f_corpus = corpus_path_dict[ldatype.name]
            lda_runner.run(f_corpus, saveTopTopics=True)

if __name__ == '__main__':
    # TODO: Decide whether we need to reset topic_threshold in LdaMallet (default 0.0)
    # Setup Mallet environment
    mallet_path = ut.getPath('mallet')
    os.environ.update({
        'MALLET_HOME': os.path.dirname(os.path.dirname(mallet_path))
    })

    root = ut.getPath('root')
    f_corpus_name = ut.getPath('CORPUS_NAME')
    f_data_dict = os.path.join(*[root, 'post-processed-data', f_corpus_name])
    ldarunner = OptionLDARunner(root=root, f_data_dict=f_data_dict, mallet_path=mallet_path, workers=getWorkers())
    args, _ = ut.parse_args_train_lda()
    lda_type = args.LDAType
    if lda_type in [x.name for x in LDAOptions]:
        ldarunner.run(LDAOptions[lda_type], args)
    else:
        raise TypeError('Type %s not defined.' % lda_type)
