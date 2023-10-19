import os
import argparse
import platform
import socket
import sqlite3
import pandas as pd
from functools import wraps
from threading import Thread
from multiprocessing import Process, Pool
from enum import Enum
import pickle
import statistics


class EvaluationMetrics(Enum):
    """
    List the metrics for hyperparameter optimization
    """
    COHERENCE = 1
    PERPLEXITY = 2
    LOGALIGNMENT = 3
    INTERCORPUSRATIO = 4
    AGGREGATED = 5

def check_existance(f, type=''):
    """
    Check if file or folder exists
    Parameters
    ----------
    f: file/folder path
    type: "d" for folder, "f" for file

    Returns
    f: path if exists
    -------

    """
    if type == "d":
        if os.path.isdir(f):
            return f
        else:
            raise NotADirectoryError('Folder not exist: %s' % f)
    elif type == "f":
        if os.path.isfile(f):
            return f
        else:
            raise FileNotFoundError('File not exist: %s' % f)
    else:
        # Check both
        if os.path.isdir(f) or os.path.isfile(f):
            return f
        else:
            raise AttributeError('Path not exist: %s' % f)


def parse_args_train_lda(*args, **kwargs):
    """
    Parse input params for lda modeling
    Returns
    -------
    args: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Input args for LDA running', *args, **kwargs)
    parser.add_argument('--LDAType',
                        type=str,
                        default=None,
                        help="""
                        Please input the number of selected LDA model
                        The types of LDA models: 
                            - LDA1: Single LDA Topic Model with word prefix preprocessing
                            - LDA2TEXT: LDA Topic Model for text terms
                            - LDA2LOG: LDA Topic Model for text terms 
                            - LDA2: Two Level LDA Topic Model 
                            - LDA2_SEP: Two Level LDA Topic Model that reads best models from LDA2TEXT & LDA2LOG
                            - LDA3: Bilingual LDA
                        """,
                        required=True)

    parser.add_argument('-l',
                        '--languages',
                        type=str,
                        default=None,
                        required=True,
                        help='Languages to be processed. Split languages by command (e.g., LOG,TEXT)\n'
                             'If not specified, all languages occurred in the dictionary will be considered')

    # This settings are for LDA1/3: Only 1 LDA model will be running
    # Or for combined model of LDA2_SEP
    parser.add_argument('--numTopics',
                        type=int,
                        default=10,
                        nargs='?',
                        required=False,
                        help='The number of topics')
    parser.add_argument('--alpha',
                        type=float,
                        default=5.0,
                        nargs='?',
                        required=False,
                        help='Dirichlet hyperparameter alpha: Document-Topic Density')
    parser.add_argument('--beta',
                        type=float,
                        default=0.01,
                        nargs='?',
                        required=False,
                        help='Dirichlet hyperparameter beta: Word-Topic Density')

    parser.add_argument('--saveLDAVis',
                        default=False,
                        required=False,
                        action='store_true',
                        help='Save LDAVis of the result. Use this for demo purposes once best params are confirmed')
    return parser.parse_known_args()


def parse_args_baseline_lda(*args, **kwargs):
    """
    Parse input params for lda baseline
    Returns
    -------
    args: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Input args for LDA running', *args, **kwargs)
    parser.add_argument('--LDAType',
                        type=str,
                        default=None,
                        help="""
                        Please input the number of selected LDA model
                        The types of LDA models: 
                            - LDA1: Single LDA Topic Model with word prefix preprocessing
                            - LDA2TEXT: LDA Topic Model for text terms
                            - LDA2LOG: LDA Topic Model for text terms 
                            - LDA2: Two Level LDA Topic Model 
                            - LDA2_SEP: Two Level LDA Topic Model that reads best models from LDA2TEXT & LDA2LOG
                            - LDA3: Bilingual LDA
                        """,
                        required=True)

    parser.add_argument('--numRepeats',
                        type=int,
                        default=10,
                        nargs='?',
                        required=False,
                        help='How many times to run each LDA model'
                        )
    return parser.parse_known_args()


def parse_args_tune_lda(*args, **kwargs):
    """
    Parse input params for tuning LDA with OpenTuner
    Returns
    -------
    args: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Input args for tuning LDA', *args, **kwargs)
    parser.add_argument('--LDAType',
                        type=str,
                        default=None,
                        help="""
                        Please input the number of selected LDA model
                        The types of LDA models: 
                            - LDA1: Single LDA Topic Model with word prefix preprocessing
                            - LDA2TEXT: LDA Topic Model for text terms
                            - LDA2LOG: LDA Topic Model for text terms 
                            - LDA2: Two Level LDA Topic Model 
                            - LDA2_SEP: Two Level LDA Topic Model that reads best models from LDA2TEXT & LDA2LOG
                            - LDA3: Bilingual LDA
                        """,
                        required=True)

    parser.add_argument('-l',
                        '--languages',
                        type=str,
                        default=None,
                        required=True,
                        help='Languages to be processed. Split languages by command (e.g., LOG,TEXT)\n'
                             'If not specified, all languages occurred in the dictionary will be considered')
    # The following code is reserved for OpenTuner
    parser.add_argument('--numTopicsRange',
                        type=str,
                        default='5-10',
                        nargs='?',
                        required=False,
                        help='The range of number of topics')
    parser.add_argument('--alphaRange',
                        type=str,
                        default='1.0-30.0',
                        nargs='?',
                        required=False,
                        help='Dirichlet hyperparameter alpha: Document-Topic Density')
    parser.add_argument('--betaRange',
                        type=str,
                        default='0.01-5.00',
                        nargs='?',
                        required=False,
                        help='Dirichlet hyperparameter beta: Word-Topic Density')

    # Add evaluation metrics selection
    parser.add_argument('-m',
                        '--metric',
                        type=int,
                        default=2,
                        nargs='?',
                        required=False,
                        help='Select the metric that is used of hyperparameter optimization by its number:\n{}'
                        .format('\n'.join(['{}={}'.format(x.name, x.value) for x in EvaluationMetrics])))

    parser.add_argument('--paraeval',
                        action='store_true',
                        required=False,
                        help="Run 10 parallel models and use the median value for training\n"
                             "This is to minimize the randomness from LDA modeling with sampling techniques\n"
                             "Noted that the number of cores for mallet will be divided in 10 parts"
                        )

    return parser.parse_known_args()


def parse_args_data_processing(*args, **kwargs):
    """
    Process data from raw HTMLs to processed data dict
    Parameters
    ----------
    args
    kwargs

    Returns
    -------

    """
    parser = argparse.ArgumentParser(description='Input args for processing raw data', *args, **kwargs)

    parser.add_argument('--sections',
                        type=str,
                        default='',
                        nargs='?',
                        required=False,
                        help='(Optional) List the sections to be processed, comma separated. Will process all sections if not specified.')
    parser.add_argument('-s',
                        '--serialization',
                        default=False,
                        required=False,
                        action='store_true',
                        help='Parse HTMLs to serialized data')

    parser.add_argument('-t',
                        '--termExtraction',
                        default=False,
                        required=False,
                        action='store_true',
                        help='Process serialized data and extract terms'
                        )
    return parser.parse_known_args()


def parse_args_term_extraction(*args, **kwargs):
    """
    Extract terms from JSON-formatted data
    Parameters
    ----------
    args
    kwargs

    Returns
    -------

    """
    parser = argparse.ArgumentParser(description='Input args for term extraction from json', *args, **kwargs)

    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='The path of json file')

    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default=None,
                        required=False,
                        help='The output of terms of a json file in pickled format')

    # parser.add_argument('-r',
    #                     '--refined',
    #                     default=False,
    #                     required=False,
    #                     action='store_true',
    #                     help='Specify if the json file is refined'
    #                     )

    parser.add_argument('-p',
                        '--processing',
                        default=True,
                        required=False,
                        action='store_true',
                        help='Processing terms. Including remove English&Github stopwords and stemming.')

    # TODO: Extend functionality for multiple languages; now we support two
    parser.add_argument('-l',
                        '--languages',
                        type=str,
                        default=None,
                        required=False,
                        help='Languages to be processed. Split languages by command (e.g., LOG,TEXT)\n'
                             'If not specified, all languages occurred in the dictionary will be considered')
    return parser.parse_known_args()

def getPath(param_str: str, check_existence=False):
    """
    Get path according to current OS platform/System Name
    -------
    Parameters
    -------
    param_str: The string of parameters
    ------
    Returns
    -------
    path
    """
    pathes = {
        'MALLET': {
            # OS
            'DARWIN': 'tools/Mallet/bin/mallet',
            'WINDOWS': 'tools/Mallet/bin/mallet',
            'LINUX': {
                'SERVER1': 'tools/Mallet/bin/mallet',
                'SERVER2': 'tools/Mallet/bin/mallet',
            }
        },
        'ROOT': {
            # OS
            'DARWIN': os.path.join(get_project_root(), 'data'),
            'WINDOWS': os.path.join(get_project_root(), 'data'),
            'LINUX': {
                'SERVER1': os.path.join(get_project_root(), 'data'),
                'SERVER2': os.path.join(get_project_root(), 'data'),
            }
        },
        'CORPUS_NAME':{
            # OS
            'DARWIN': 'foobar_reformatted.pkl',
            'WINDOWS': 'foobar_reformatted.pkl',
            'LINUX': {
                'SERVER1': 'foobar_reformatted.pkl',
                'SERVER2': 'foobar_reformatted.pkl',
            }
        }
    }

    try:
        # Get path: By key --> OS
        p = pathes[param_str.upper()][platform.system().upper()]
        if isinstance(p, dict):
            # Choose by server name
            serverName = socket.gethostname().upper()
            if serverName in p.keys():
                p = p[serverName]
            else:
                # If unknown server name, we assume that's from a default server
                p = p['SERVER1']
    except Exception as e:
        raise RuntimeError('Error getting path for %s: %s' % (param_str, e))
    if check_existence:
        return check_existance(p)
    else:
        return p


def getWorkers(cpus=None, limit_usage_machines=[], limit_usage_perc=0.5):
    """
    Return the number of CPUs we can use, based on machine type;
    We basically use a given portion of CPU
    For other machines (e.g. Compute Canada Nodes) we use all available CPUs
    Returns
    -------

    """
    cpu_count = os.cpu_count()
    if cpus:
        if isinstance(cpus, float):
            return round(cpus * cpu_count)
        elif isinstance(cpus, int):
            return cpus
        else:
            raise TypeError('Unrecogzied input type %s' % str(cpus.__name__))
    if socket.gethostname().lower() in limit_usage_machines:
        return int(cpu_count * 0.5)
    else:
        # Use all cpus if not on SAIL servers
        return cpu_count

def run_async(func):
    """
    Run function in parallel
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # queue = Queue()
        # thread = Thread(target=func, args=(queue,) + args, kwargs=kwargs)
        # thread.start()
        # return queue
        thread = Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        #thread.join()
        return thread
    return wrapper


def run_async_multiprocessing(func):
    """
    Run function in parallel
    Be aware that the wrapper may cause error in python 3.8
    https://discuss.python.org/t/is-multiprocessing-broken-on-macos-in-python-3-8/4969
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        proc = Process(target=func, args=args, kwargs=kwargs)
        proc.start()
        return proc
    return wrapper


def list_files_by_time(dir, ext=None, reverse=True):
    """
    List all files, retrieve
    Parameters
    ----------
    dir
    ext
    reverse

    Returns
    -------

    """
    import glob
    if not ext:
        files = list(filter(os.path.isfile, glob.glob(dir)))
    else:
        files = list(filter(os.path.isfile, glob.glob(os.path.join(dir, '*%s' % ext))))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=reverse)
    return files

def unpikle_dat(dat):
    """
    Unpickle compressed parameter set
    It might fail, we are still working on debugging that
    Parameters
    ----------
    dat

    Returns
    -------

    """
    try:
       x = pickle.loads(dat)
    except pickle.UnpicklingError:
       x = None
    return x


def loadtable(dbpath, metric=None):
    """
    Load table from DB. Return the best (available) results and param sets
    Parameters
    ----------
    dbpath
    findTopAvailable

    Returns
    -------

    """
    # Connection to db
    db = sqlite3.connect(dbpath)
    table_conf, table_res = [pd.read_sql_query("SELECT * from %s" % table_name, db)
                             for table_name in ['configuration', 'result']]
    table = pd.merge(table_conf, table_res, on='id')
    table['data'] = table['data'].map(unpikle_dat)

    # Sort by minimizing column "time"
    table = table.sort_values('time', ascending=True)

    if metric:
        if metric != EvaluationMetrics.PERPLEXITY.name:
            table['time'] = table['time'].apply(lambda x: abs(x))
    return table

def findBestParamSet(table, findTopAvailable=True):
    """
    Find top paramsets from table
    Parameters
    ----------
    table
    findTopAvailable

    Returns
    -------

    """
    if findTopAvailable:
        for idx, row in table.iterrows():
            # Find the top val with configuration insides
            res = row['time']
            conf_dat = row['data']
            if res and conf_dat:
                return res, conf_dat
    else:
        top_row = table.iloc[0]
        return top_row['time'], top_row['data']

def retrieve_tuning_res(db_dir, findTopAvailable=True):
    """
    Retrieve the tuning result from OpenTuner database
    and find the best (available) performing tuning result and parameter sets

    Parameters
    ----------
    db_dir: The directory of dataset. There should be only one db file. If more than one file exists, find the latest db
    findTopAvailable: The parameter sets are stored in pickled files.
            Sometimes when unpickling fails, find the first available parameter sets that receives nearly best result

    Returns
    -------

    """
    db_files = list_files_by_time(dir=db_dir, ext='.db')
    # We only load the latest file
    db_f = db_files[0]
    return findTopAvailable(loadtable(dbpath=db_f))


def get_project_root():
    """
    Get the root directory of project (main folder as root)
    Returns
    -------
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_median_vals(f, col):
    """
    Get the median value of a column
    The input can be either a file path or a dataframe
    Parameters
    ----------
    f
    col

    Returns
    -------

    """
    if isinstance(f, str):
        try:
            df = pd.read_csv(f)
        except Exception:
            raise ValueError('Unable to load file as dataframe: %s' % f)
    elif isinstance(f, pd.DataFrame):
        df = f
    else:
        raise TypeError('Unable to recognize object type, should be a filepath as string or a dataframe')

    try:
        med = df[col].median()
        try:
            # If med in list
            idx = list(df[col]).index(med)
        except ValueError:
            # Find closest to med value
            # This happens when len(repeats) is even, and the median is the mean of two vals
            closest_val = min(list(df[col]), key=lambda x: abs(x - med))
            idx = list(df[col]).index(closest_val)
        return med, idx
    except Exception:
        print('Unable to get median value for column: %s' % col)
        return None, None