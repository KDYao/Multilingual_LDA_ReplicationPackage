"""
Specific to BB's dataset, we extract terms from different languages
"""
import json
import dill
import os
from src.resources.utils.utils import parse_args_term_extraction
from collections import defaultdict
from src.dataprocessing.WordProcess import get_stopwords, word_breaker
from nltk import word_tokenize
import numpy
from nltk import PorterStemmer


class TermExtraction:
    def __init__(self, f_input, f_output, processing, languages):
        if not os.path.isfile(f_input):
            raise FileNotFoundError('Cannot find input %s' % f_input)
        if f_output:
            if not os.path.isfile(f_output):
                raise FileNotFoundError('Cannot find specified output %s' % f_output)
        self.input_data = self._load_input(f_input)
        if f_output:
            self.f_output = f_output
        else:
            self.f_output = f_input.rstrip('.json') + '.pkl'
        if not os.path.isdir(os.path.dirname(self.f_output)):
            os.makedirs(os.path.dirname(self.f_output))
        self.languages = self._check_languages(languages)
        self.is_processing = processing

    def _check_languages(self, languages):
        """

        Parameters
        ----------
        language: languages to be processed

        Returns
        -------

        """
        all_languages = []
        for post_dict in self.input_data:
            for post in post_dict['posts']:
                 all_languages += [x['language'] for x in post['paragraphs']]
        available_languages = list(set(all_languages))
        del all_languages
        if languages:
            chosen_languages = [x.strip() for x in languages.split(',')]
            unexpected_languages = [x for x in chosen_languages if x not in available_languages]
            if len(unexpected_languages) != 0:
                raise ValueError('Unexpected languages found: %s' % ','.join(unexpected_languages))
            return chosen_languages
        else:
            return available_languages

    def _load_input(self, f):
        """
        Load input json
        Parameters
        ----------
        f

        Returns
        -------

        """
        with open(f, 'r') as fr:
            input_data = json.load(fr)
        return input_data

    def extract(self):
        """
        Extract terms from different languages
        Returns
        -------
        """
        # Iterate all posts
        dict_terms = defaultdict(lambda: defaultdict(list))
        if self.is_processing:
            stopwords = get_stopwords()
        else:
            stopwords = None
        for post_dict in self.input_data:
            pid = post_dict['id']
            for post in post_dict['posts']:
                for paragraph in post['paragraphs']:
                    p_language = paragraph['language']
                    if p_language in self.languages:
                        p_text = self.process_text(paragraph['text'], stopwords)
                        dict_terms[p_language][pid] = p_text
        with open(self.f_output, 'wb') as fw:
            dill.dump(dict_terms, fw)



    def process_text(self, tx, stopwords):
        """
        Process text by performing stemming and removing stopwords

        There are certain rules for processing different types of language
        For example, in Mohammed's defined rules (dataprocessing/RawHTMLProcessing.py):
        Log terms:
            - A term should not be purely numeric
            - A term needs to be alpha-numeric
            - Special characters are removed from a term following certain rules
            - A term should contain more than 2 characters
        Text terms:
            - A term should not occur in stopwords list
            - A term should not be purely numeric
            - A term should contain more than 2 characters

        Different processing rules should be applied to different language terms
        Currently we are testing on a pseudo-dataset,
        so we perform a single processing rule following Mohammed's contribution for all languages
        Parameters
        ----------
        tx
        Returns
        -------
        """
        terms = word_tokenize(tx)
        numpy.seterr('raise')
        ps = PorterStemmer()
        if self.is_processing:
            # Skip further processing if processing is disabled
            return terms
        # Apply all rules;
        # TODO: consider customize rules for different languages in the future
        if stopwords:
            terms = [
                x for x in word_breaker(terms) if
                x.lower() not in stopwords and len(x) > 2 and
                x.isnumeric() == False and x.isalnum() == True
            ]
        else:
            terms = [
                x for x in word_breaker(terms) if
                len(x) > 2 and
                x.isnumeric() == False and x.isalnum() == True
            ]
        return [ps.stem(term) for term in terms]


if __name__ == '__main__':
    args, _ = parse_args_term_extraction()
    term_ex = TermExtraction(
        f_input=args.input,
        f_output=args.output,
        #refined=args.refined,
        processing=args.processing,
        languages=args.languages
    )
    term_ex.extract()