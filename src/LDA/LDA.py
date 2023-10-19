#!/usr/bin/env python
import itertools
import pickle
import os
import json
import statistics
import subprocess

import numpy as np
from gensim.models import CoherenceModel

# Disable wrapper and save it locally, since its not supported by gensim since version 4.0
# from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
#from gensim.models.wrappers import LdaMallet
from src.resources.gensim_mod.wrappers.ldaMallet import LdaMallet
from src.resources.gensim_mod.wrappers.ldaMallet import malletmodel2ldamodel
from src.resources.utils.utils import check_existance, EvaluationMetrics, get_project_root
from src.LDA.LDAOptions import LDAOptions

class LDA:
    def __init__(self, root, mallet_path, lda_type_str, alpha=50, beta=0.01, topicNum=11, workers=4, iterations=100,
                 topic_filtering_thresh=0.2, metric_type=None, exp_id=None, exp_type=None, isTemp=False):
        self.root = check_existance(root, 'd')
        # Number of Topics (K)
        # Dirichlet hyperparameter alpha: Document-Topic Density
        # Dirichlet hyperparameter beta: Word-Topic Density
        self.alpha = alpha
        self.beta = beta
        self.mallet_path = mallet_path
        self.topicNum = topicNum
        self.workers = workers
        self.iterations = iterations
        self.coherence = None
        self.perplexity = None
        # Self-defined: Log alignment measures the average of proportion of log terms/all terms in each topic
        self.log_alignment = None
        # Self-defined: The ratio of corpus that contain intercorpus information (contain both text and log)
        self.intercorpus_topics_ratio = None
        # The param used for tuning. It's the harmonic average of three metrics: coherence, perplexity, log-alignment
        self.aggregated = None
        self.topic_filtering_thresh = topic_filtering_thresh
        self.proj_root = get_project_root()
        if exp_id is not None:
            if metric_type:
                self.tmp_out = os.path.join(*[self.proj_root, "out", exp_type, metric_type,
                                              'model', lda_type_str, 'job_%d' % exp_id, ''])
            else:
                self.tmp_out = os.path.join(*[self.proj_root, "out", exp_type,
                                              'model', 'job_%d' % exp_id, lda_type_str, ''])
        else:
            if metric_type:
                self.tmp_out = os.path.join(*[self.proj_root, "out", metric_type, 'model', lda_type_str, ''])
            else:
                self.tmp_out = os.path.join(*[self.proj_root, "out", 'model', lda_type_str, ''])
        # Preserve for LDA2TEXT, LDA2LOG at runtime
        if isTemp: self.tmp_out = self.tmp_out + 'temp' + os.path.sep
        self.lda_type_str = lda_type_str
        if not os.path.isdir(self.tmp_out):
            os.makedirs(self.tmp_out)
        self.metric_type = metric_type
        self.exp_type = exp_type

        # We introduce another way to calculate aggregated metric
        # So RUN PERPLEXITY before AGGREGATED
        self._getPerplexityRange()

    def _getPerplexityRange(self):
        """
        This function reads perplexity range from perplexity runs
        Returns
        -------

        """
        self.min_max_perp = None
        if self.metric_type == EvaluationMetrics.AGGREGATED.name or self.exp_type == 'baseline':
            # Locate perplexity

            f_reference = os.path.join(*[self.proj_root, 'out', EvaluationMetrics.PERPLEXITY.name, 'log',
                                         'output_%s.log' % self.lda_type_str])
            if os.path.isfile(f_reference):
                min_max = []
                for opt in ('head', 'tail'):
                    cmd = """grep -io '"Perplexity": "[^"]*' {f} | grep -o '[^"]*$' | sort -V | {opt} -n1""".format(
                        f=f_reference, opt=opt
                    )
                    out = subprocess.check_output(cmd, shell=True).decode('utf-8')
                    min_max.append(float(out.strip()))
                self.min_max_perp = min_max

    @staticmethod
    def format_topics_sentences(ldamodel, corpus, threshold=0.0):
        """
        Find the topics with highest probabilities, then choose the top 10 terms from each topic
        The threshold is used to filter the topics that has lower than threshold probabilities
        Parameters
        ----------
        ldamodel
        corpus
        threshold

        Returns
        -------

        """
        out_doc = []
        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num, topn=10)
                    topic_keywords = [
                        '{term};TOPIC:{topicID}'.format(topicID=str(topic_num), term=word)
                        for word, prop in wp]
                    # 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords
                    # print((int(topic_num), round(prop_topic, 4), topic_keywords))
                    if prop_topic >= threshold:
                        out_doc.append(topic_keywords)
                    else:
                        out_doc.append([])
                    break
                else:
                    break
        return out_doc

    def _calculate_log_alignment(self, documents):
        """
        Calculate the average number of log alignment
        A log alignment value indicates how many percent tokens in a topic are log tokens
        Also check how many corpus actually has both log and text terms
        numIntercorpusTopics/numTopics
        Parameters
        ----------
        documents: document to be analyzed

        Returns
        -------
        average log alignment for all documents, and the proportion of documents with both Log&Text
        """
        log_alignment_vals = []
        for topic in documents:
            num_terms = len(topic)
            num_log_terms = len([x for x in topic if x.startswith('LOG')])
            # We consider Inter-corpus topics that should have both Log and Text terms
            if num_log_terms != 0 and num_log_terms != num_terms:
                log_alignment_vals.append(float(num_log_terms/num_terms))
        if log_alignment_vals:
            # Calculate log alignment in each topic
            return statistics.median(log_alignment_vals), len(log_alignment_vals) * 1.0/len(documents)
        else:
            return None, 0

    def _calculate_intercorpus_topic_coverage(self, model, top_words=10):
        """
        Check how many corpus actually has both log and text terms
        numIntercorpusTopics/numTopics
        Parameters
        ----------
        documents

        Returns
        -------
        """
        if hasattr(model, 'num_topics'):
            num_topics = model.num_topics
        else:
            num_topics = self.topicNum
        inter_corpus_topics = []
        all_topics = []
        for topic_id, topic_terms in model.show_topics(formatted=False, num_topics=num_topics, num_words=top_words):
            all_topics.append(topic_id)
            terms = [x[0] for x in topic_terms]
            text_terms_num = len([x for x in terms if x.startswith('TEXT:')])
            if text_terms_num == 0 or text_terms_num == top_words: continue
            else:
                inter_corpus_topics.append(topic_id)
        return float(len(inter_corpus_topics)/len(all_topics))

    def getAggregatedMetric(self):
        """
        An aggregated value that considers all three metrics: coherence; perplexity; log-alignment
        The cost is the harmonic mean of those three metrics of interest.

        By default, irace considers that the value returned by targetRunner (or by targetEvaluator,
        if used) should be minimized.

        Since we use harmonic mean, all values should be > 0
        We want higher coherence; However, we probably need lower perplexity within each topic;
        And we are not sure about the preference for log alignment, maybe see how close it is to 50% (both log/text)

        Currently we just use coherence

        Returns
        -------
        cost
        """
        if self.min_max_perp:
            min_perp, max_perp = self.min_max_perp
            perp_scaled = (self.perplexity - min_perp) / (max_perp - min_perp)
            if perp_scaled > 1:
                perp_scaled = 1
            elif perp_scaled < 0:
                perp_scaled = 0
            preplexity_norm = 1 - perp_scaled
        else:
            preplexity_norm = 1/self.perplexity
        vals = [x for x in (self.coherence, preplexity_norm, self.intercorpus_topics_ratio) if x]
        return statistics.harmonic_mean(vals)

    def run(self, f_corpus, data_corpus=None, saveTopTopics = False, saveLDAVis=False, topTopicsOnly=False):
        """
        Run ldaMallet and print cost of the algorithm
        Parameters
        ----------
        f_corpus: int or tuple
        the input sources of pickled corpora information (could be a tuple: LDA3 or a str: others)
        saveTopTopics: boolean optional
            Choose the topics that has highest probability for each document;
            And choose the top 10 terms for each topic
        Returns
        -------

        """
        # Top docs output
        f_top_docs = self.tmp_out + 'top_docs_list.pkl'
        # Check if it's used for lda2 top-topic documents building
        if isinstance(f_corpus, str) or (data_corpus is not None):
            if data_corpus:
                documents, corpus, dictionary = data_corpus
            else:
                # Skip running lda2log and log2text if top doc file already exists
                if saveTopTopics and os.path.isfile(f_top_docs): return

                with open(f_corpus, 'rb') as handle:
                    documents, corpus, dictionary = pickle.load(handle)

            model = LdaMallet(mallet_path=self.mallet_path, corpus=corpus, num_topics=self.topicNum, id2word=dictionary,
                              workers=self.workers, prefix=self.tmp_out, iterations=self.iterations,
                              alpha=self.alpha, beta=self.beta)
            ldaModel = malletmodel2ldamodel(model)
            if topTopicsOnly:
                return self.getTopTopics(model=ldaModel, corpus=corpus)

            coherencemodel = CoherenceModel(model=ldaModel, texts=documents, dictionary=dictionary, coherence='c_v')
            #coherencemodel = CoherenceModel(model=ldaModel, texts=documents, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence = coherencemodel.get_coherence()
            perplexity = model.get_perplexity()

        elif isinstance(f_corpus, tuple):
            # LDA3, the tuple contains the path for both log and text corpora pickle files
            corpus = []
            documents = []
            models = []
            # Load documents for both log and text
            for f in f_corpus:
                with open(f, 'rb') as handle:
                    f_documents, f_corpus, f_dictionary = pickle.load(handle)
                    corpus.append(f_corpus)
                    documents.append(f_documents)
                    language_type = os.path.basename(f).rstrip('.pkl')
                    # Corpus is None, setup and avoid training
                    model = LdaMallet(mallet_path=self.mallet_path, prefix=self.tmp_out + '%s_' % language_type, id2word=f_dictionary)
                    models.append(model)

            # Perform multi-lingual mallet
            model = LdaMallet.train_multilingual(
                mallet_path=self.mallet_path, num_topics=self.topicNum, workers=self.workers,
                iterations=self.iterations, alpha=self.alpha, beta=self.beta, prefix=self.tmp_out,
                models=models, corpus=corpus
            )
            ldaModel = malletmodel2ldamodel(model)

            # Update documents by merging log/text documents
            #documents = [x + y for x, y in zip(documents[0], documents[1])]
            documents_lda3 = []
            for idx in range(0, len(documents[0])):
                documents_lda3.append(list(itertools.chain.from_iterable([x[idx] for x in documents])))
            documents = documents_lda3

            # Update dictionaries
            dictionary = model.id2word

            coherencemodel = CoherenceModel(model=ldaModel,
                                            texts=documents,
                                            coherence='c_v')
            # Disable zero divide warning
            with np.errstate(divide='ignore'):
                coherence = coherencemodel.get_coherence()

            # # Update corpus by merging log/text corpus and update the token indexes for text corpus (+length of log tokens)
            # text_corpus = corpus.pop(1)
            # # Assign log_corpus to corpus
            # corpus = corpus[0]
            # log_num_terms = models[0].num_terms
            # for idx, text_corpus_row in enumerate(text_corpus):
            #     corpus[idx] += [(tp[0] + log_num_terms, tp[1]) for tp in text_corpus_row]

            prev_num_terms = 0
            corpus_lda3 = []
            for mod, cor in zip(models, corpus):
                num_terms = mod.num_terms
                for idx, corpus_row in enumerate(cor):
                    try:
                        corpus_lda3[idx] += [(tp[0] + prev_num_terms, tp[1]) for tp in corpus_row]
                    except IndexError:
                        corpus_lda3.append(corpus_row)
                prev_num_terms += num_terms
            corpus = corpus_lda3

            # get per-word likelihood bound
            log_likelihood = ldaModel.log_perplexity(corpus)
            perplexity = np.exp2(-log_likelihood)

        mallet_out = model.output
        with open(self.tmp_out + 'mallet_stdout.log', 'w') as wout:
            wout.write(mallet_out)
        with open(self.tmp_out + 'mallet_model.pkl', 'wb') as pk_out:
            pickle.dump(ldaModel, pk_out)

        # Calculate evaluation metrics
        self.coherence = coherence
        self.perplexity = perplexity
        self.log_alignment, _ = self._calculate_log_alignment(documents=documents)
        self.intercorpus_topics_ratio = self._calculate_intercorpus_topic_coverage(model=ldaModel)

        self.aggregated = self.getAggregatedMetric()

        res = {
            EvaluationMetrics.COHERENCE.name: str(coherence),
            EvaluationMetrics.PERPLEXITY.name: str(perplexity),
            EvaluationMetrics.LOGALIGNMENT.name: str(self.log_alignment),
            EvaluationMetrics.INTERCORPUSRATIO.name: str(self.intercorpus_topics_ratio),
            EvaluationMetrics.AGGREGATED.name: str(self.aggregated)
        }
        print(json.dumps(res))

        # Highest probability topic for each doc
        if saveTopTopics:
            self.getTopTopics(model=ldaModel, corpus=corpus, f_save=f_top_docs)

        if saveLDAVis:
            import pyLDAvis.gensim_models
            vis = pyLDAvis.gensim_models.prepare(topic_model=ldaModel, corpus=corpus, dictionary=dictionary)
            pyLDAvis.save_html(vis, os.path.join(self.tmp_out, '%s_ldavis.html' % self.lda_type_str))

        return res

    def get_top_terms(self, topN=10):
        """
        Load top terms per topic for all documents
        Returns
        -------

        """
        f_doc_topics = self.tmp_out + 'doctopics.txt'
        f_topic_keys = self.tmp_out + 'topickeys.txt'

        lst_topic_terms = []

        # Read top terms per topic
        # Noted some topic has less than 10 terms (default top N words from mallet output is 20)
        # mt_topic_keys = np.loadtxt(f_topic_keys, dtype="str", usecols=range(2, 2 + topN))
        with open(f_topic_keys, 'r') as fr:
            for idx, line in enumerate(fr):
                if line.strip() == '': continue
                words = line.strip('\n').split()[2:]
                top_terms = words[:min(topN, len(words))]
                lst_topic_terms.append(['{term};TOPIC:{topicID}'.format(topicID=str(idx), term=x) for x in top_terms])

        num_topics = len(lst_topic_terms)
        # Load document-topic
        mt_doc_topics = np.loadtxt(f_doc_topics, dtype="float", usecols=range(2, 2 + num_topics))
        max_theta_indexes = np.argmax(mt_doc_topics, axis=1)
        return [lst_topic_terms[x] for x in max_theta_indexes]


    def getTopTopics(self, model, corpus, f_save=None):
        """
        Return the top topics for each word
        Parameters
        ----------
        model
        corpus

        Returns
        -------

        """
        if self.lda_type_str.startswith('LDA2_'):
            try:
                top_documents = self.get_top_terms()
            except Exception:
                top_documents = LDA.format_topics_sentences(
                    ldamodel=model, corpus=corpus, threshold=self.topic_filtering_thresh)
            # Top doc output
            if f_save is not None:
                with open(f_save, 'wb') as pk_out:
                    pickle.dump(top_documents, pk_out)
            return top_documents