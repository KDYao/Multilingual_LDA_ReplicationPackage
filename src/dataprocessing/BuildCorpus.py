#!/usr/bin/env python
import numpy
import os
from nltk.stem import PorterStemmer
from src.dataprocessing.WordProcess import get_stopwords
import pickle
import dill
import itertools
from src.LDA.LDAOptions import LDAOptions
from src.resources.utils.utils import check_existance
from src.resources.gensim_mod.corpora.dictionary import Dictionary
from enum import Enum
from collections import defaultdict


class CorpusBuilder:
    def __init__(self, root: str, f_data_dict: str):
        self.root = check_existance(root, 'd')
        self.f_data_dict = check_existance(f_data_dict, 'f')

    def generate_corpus_dictionary(self, input_documents, isFilter=True):
        """
        Process documents and build corpus
        Parameters
        ----------
        input_documents: documents to be processed
        isFilter: filter the most/least frequent terms

        Returns
        -------

        """
        texts = input_documents
        # It seems the documents need to be processed first before running other functions
        # Few params such as self.numTopics are required in these functions
        dictionary = Dictionary(texts)
        if isFilter:
            dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=None)
        corpus = [dictionary.doc2bow(text) for text in texts]

        return (corpus, dictionary)


    def createCorpus(self, doc_dir, isFilter=True, languages=None):
        """
        Create corpus for option 1
        Parameters
        ----------
        doc_dir: Directory of the documents
        isFilter: filters extreme frequency tokens; by default we filter token frequency <2% or >50%
        Returns
        -------

        """
        f_out_all = [os.path.join(doc_dir, lan + '.pkl') for lan in languages]
        f_out_all.append(os.path.join(doc_dir, '%s.pkl' % '_'.join(languages)))

        # If all required post-processed files are ready, skip
        if all([os.path.isfile(x) for x in f_out_all]):
            print('All required post-processed docs ready; skip re-generating')
            return

        with open(self.f_data_dict, 'rb') as handle:
            data_dict = dill.load(handle)
        print("Done with collecting post content")
        if not languages:
            languages = data_dict.keys()

        # Iter all post ids
        # We know that each post should be multi-lingual
        # Just to be safe
        all_posts = []
        for lang_detail in data_dict.values():
            for post_id in lang_detail.keys():
                if post_id not in all_posts:
                    all_posts.append(post_id)

        documents = []
        process_terms = defaultdict(list)
        for post in all_posts:
            document = []
            for language in languages:
                if post in data_dict[language].keys():
                    terms = [language + ':' + x for x in data_dict[language][post]]
                else:
                    terms = []
                process_terms[language].append(terms)
                document.extend(terms)
            documents.append(document)
        (corpus, dictionary) = self.generate_corpus_dictionary(
            input_documents=documents, isFilter=isFilter
        )

        # Separate
        #FIXME: Change here
        isFilter =False
        for lan in languages:
            f_out = os.path.join(doc_dir, lan + '.pkl')
            documents_lan = process_terms[lan]
            (corpus_lan, dictionary_lan) = self.generate_corpus_dictionary(
                input_documents=documents_lan, isFilter=isFilter
            )
            self.saveCorpus(f_out, documents_lan, corpus_lan, dictionary_lan)

        # All languages
        if len(languages) > 1:
            f_out = os.path.join(doc_dir, '%s.pkl' % '_'.join(languages))
            self.saveCorpus(f_out, documents, corpus, dictionary)


    def saveCorpus(self, outputPath, documents, corpus, dictionary):
        """
        Save corpus in a local picked file
        Parameters
        ----------
        outputPath
        documents
        corpus
        dictionary

        Returns
        -------

        """
        if not os.path.isdir(os.path.dirname(outputPath)):
            os.makedirs(os.path.dirname(outputPath))
        with open(outputPath, 'wb') as handle:
            pickle.dump((documents, corpus, dictionary), handle)
        print("Done with collecting documents: %s" % os.path.basename(outputPath))
