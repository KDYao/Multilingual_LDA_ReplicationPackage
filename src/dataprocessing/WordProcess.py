#!/usr/bin/env python

"""
THIS PART OF CODE IS AN COMBINATION OF STOPWORDS PROCESSING AND DOCUMENTS PROCESSING
"""

import re
import ssl
import string
import os

import nltk
from nltk import RegexpTokenizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
from spellchecker import SpellChecker

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
checker = SpellChecker()
checker.word_frequency.load_text_file(os.path.join(__location__, 'bc_toollist.txt'))

def read_stopwords_from_file (file_path):
    #Open a file in the same directory as the containing module
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    stopword_list = []
    with open(os.path.join(__location__,file_path), 'r') as reader:
        line = reader.readline()
        line = re.sub('\"|,', ' ' ,line)
        words = [word for word in re.split('\s+',line) if len(word)>1]
        stopword_list.extend(words)
        while line:
            line = reader.readline()
            line = re.sub('\"|,', ' ',line)
            words = [word for word in re.split('\s+',line) if len(word)>1]
            stopword_list.extend(words)
    return stopword_list


# Now merge the stopword list with that of nltk and include the punctuations also
def get_stopwords(isLog=False):
    for pkg_path, pkg in [('tokenizers/punkt', 'punkt'), ('corpora/stopwords', 'stopwords')]:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        try:
            nltk.data.find(pkg_path)
        except LookupError:
            nltk.download(pkg)
    stopwords_list = []
    nltk_stopwords = stopwords.words("english")
    if isLog:
        github_stopwords = read_stopwords_from_file('github_stopwords.txt')
        stopwords_list.extend(github_stopwords)
    stopwords_list.extend(nltk_stopwords)

    punctuations = string.punctuation
    for p in punctuations:
        stopwords_list.append(p)
    return set(stopwords_list)


def process_terms(text, stopwords=None, minlen=2, wordbreak=False):
    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    # Lowercase
    words = tokenizer.tokenize(text.lower())
    # Remove punctuations
    # Remove stopwords
    if stopwords:
        words = [word.lower() for word in words if word not in stopwords and word.isnumeric() == False and len(word) >= minlen]
    else:
        words = [word.lower() for word in words if word.isnumeric() == False and len(word) >= minlen]
    if wordbreak:
        return word_breaker(words)
    return words


'''
This function breaks long words into small pieces based on the presence of characters"
Example-1: ... or ........
Example-2: ==== or ===================
#Example-3: C:\programfiles\System32\point3.dll
#Example-4: \Subject
#Example-5: text.If or close.Then, => This occurs because the nltk word tokenizer is not always able to identify sentence boundary
'''
def word_breaker(word_list):  # we use domain specific knowledge to break the word
    output = []
    for word in word_list:
        if '==' in word:
            output.extend(re.split('=+', word))
        elif '__' in word:
            output.extend(re.split('_+', word))
        elif ':' in word:
            output.extend(re.split(':+', word))
        elif '\\' in word:
            output.extend(re.split('\\\\', word))
        elif '..' in word:
            output.extend(re.split('\.+|\\s+', word))
        elif '/' in word:
            output.extend(re.split('/', word))
        else:
            if '.' in word:
                index = word.index('.')
                if (index < len(word) - 2 and word[index + 1].isupper()):
                    output.extend(re.split('', word))
                else:
                    output.append(word)
            else:
                output.append(word)
    return output


def get_text(post):
    """
    Extract the natural language parts of a post if this post does not contain MiniToolbox logs
    Parameters
    ----------
    post

    Returns
    -------

    """
    # Template:
    # Tool: {Pattern: IsRegex}
    tools_check = {
        # Tool: HijackThis
        # Ref: https://www.bleepingcomputer.com/download/hijackthis/
        'HijackThis': {
            # Example: tid-link-398646_1.html
            r'Logfile of Trend Micro HijackThis': False,
            # Example: tid-link-389175_1.html
            r'Logfile of HijackThis v': False
        },
        # Tool: Malwarebytes Anti-Malware
        # Ref: https://www.bleepingcomputer.com/download/malwarebytes-anti-malware/
        # Example:
        #   - tid-link-396186_1.html
        #   - tid-link-616292_1.html
        # Pattern: r'Malwarebytes.*?\nwww.malwarebytes.'
        # Example2: https://www.bleepingcomputer.com/forums/t/251641/perhaps-an-infected-laptop-computer/
        # Pattern2: r'Malwarebytes' Anti-Malware \d+.\d+\nDatabase'
        'Malwarebytes': {
            r'Malwarebytes.*?www.malwarebytes.': True,
            r"Malwarebytes'? Anti-Malware.*Database version:": True,
        },
        # Early version of MBAM log
        # https://www.bleepingcomputer.com/forums/t/247242/backdoortidserv-infected-computer/#entry1378124
        # Keep it as it is
        'MBAM': {
            r'AMmbam-log-': False
        },
        # Tool: ESET
        # Ref: https://www.eset.com/int/home/online-scanner/
        # Example: https://www.bleepingcomputer.com/forums/t/356063/infected-with-trojanadh/page-2
        # Pattern: ESETSmartInstaller@High as CAB hook log
        'ESET': {
            r'ESET.*?CAB hook log': True
        },
        # Tool: OTL Download
        # Ref: https://www.bleepingcomputer.com/download/otl/
        # Example: https://www.bleepingcomputer.com/forums/t/533856/otl-logs/
        # Pattern: OTL logfile created on
        'OTL': {
            r'OTL logfile created on': False
        },
        # Tool: ComboFix
        # Ref: https://www.bleepingcomputer.com/download/combofix/
        # Pattern: r'ComboFix\s+\d+-\d+-\d+'
        'ComboFix': {
            r'ComboFix\s+\d+-\d+-\d+': True
        },
        # Tool: SUPERAntiSpyware
        # Ref: https://www.bleepingcomputer.com/download/superantispyware/
        # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-417496_1.html
        # Pattern: SUPERAntiSpyware Scan Log\nhttp://www.superantispyware.com
        'SUPERAntiSpyware': {
            # Need to use Ignore case of re.search()
            r'SUPERAntiSpyware Scan Log http://www.superantispyware.com': True
        },
        # Tool: TDSSKiller
        # Ref: https://www.bleepingcomputer.com/download/tdsskiller/
        # Example:
        #   - http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-553398_1.html
        #   - http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-537204_1.html
        # Pattern: <datetime> <addr> TDSS rootkit removing tool
        'TDSS':{
            r'TDSS Qlook Version': False,
            r'TDSS rootkit removing tool \d+.\d+.': True,
            r'TDSSKiller logs in order': False
        },
        # Tool: RKUnhookerLE.exxe
        # Example:
        # - https://www.bleepingcomputer.com/forums/t/406706/redirect-websites/
        # - http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-408580_2.html
        # Pattern: RkU Version:
        'RkU': {
            r'RkU Version:': False
        },
        # Tool: Avira AntiVir
        # Ref: https://www.avira.com/
        'Avira':{
            r'Avira AntiVir Personal Report file': False
        },
        # Tool: GMER
        # Ref: http://www.gmer.net
        # Pattern:
        'GMER': {
            r'GMER.*?http://www.gmer.net': True
        },
        # Tool: rootkit removing tool
        # Example:
        'rootkit removing tool': {
            r'rootkit removing tool \d+.\d+.': True
        },

        # Tool: Windows Registry Editor
        # Example: 506826
        'Windows Registry Editor': {
            r'Windows Registry Editor Version': False
        },

        # Tool: Junkware removal tool
        'JRT': {
            r'Junkware Removal Tool (JRT) by': False
        },

        # Tool: Windows Repair
        # Example: 497182
        'Windows Repair': {
            r'Windows Repair v': False
        },

        # Tool: Farbar Recovery Scan Tool
        # Ref: https://www.bleepingcomputer.com/download/farbar-recovery-scan-tool/
        # Example: https://www.bleepingcomputer.com/forums/t/669350/result-of-farbar-recovery-scan-tool/
        # Pattern: Scan result of Farbar Recovery Scan Tool(FRST)
        'Farbar Recovery Scan Tool': {
            r'Scan result of Farbar Recovery Scan Tool(FRST)': False
        },

        # Tool: MiniToolBox
        # Ref: https://www.bleepingcomputer.com/download/minitoolbox/
        # Example: https://www.bleepingcomputer.com/forums/t/613807/thrown-a-lot-at-googleru-and-it-will-not-just-jog-on/
        # Pattern: MiniToolBox by Farbar
        'MiniToolBox': {
            r'MiniToolBox by Farbar': False
        },

        # Tool: AdwCleaner
        # Ref: https://www.bleepingcomputer.com/download/adwcleaner/
        # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-674668_1.html
        # Pattern: # AdwCleaner <version>
        'AdwCleaner':{
            r'# AdwCleaner ': False
        },

        # Tool: aswMBR
        # Ref: https://www.bleepingcomputer.com/download/aswmbr/
        # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-457013_1.html
        'aswMBR': {
            r'aswMBR version': False
        },


        # Tool: SecurityCheck
        # Ref: https://www.bleepingcomputer.com/download/securitycheck/
        # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-674668_1.html
        'SecurityCheck:': {
            r'SecurityCheck by': False,
            r'Security Check version': False
        },


        # Tool: Farbar Sertice Scanner
        # Ref: https://www.bleepingcomputer.com/download/farbar-service-scanner/
        # Example:http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-677201_1.html
        'Farbar Service Scanner': {
            r'Farbar Service Scanner Version': False
        },

        # Tool: McAfee Security Scan
        'McAfee Security Scan': {
            r'McAfee Security Scan Plus (Version': False
        },

        # Tool: System Information Viewer
        'System Information Viewer': {
            r'System Information Viewer V': False,
        },

        # Tool: Rkill
        # Ref:
        # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-674223_1.html
        'Rkill': {
            r'Rkill.*?by': True
        },
        # Tool: SystemLook
        # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-460522_1.html
        'SystemLook': {
            r'SystemLook.*?by': True
        },
        # Tool: Hitman Pro
        # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-536090_2.html
        'Hitman Pro': {
            r'HitmanPro.*?www.hitmanpro.com': True
        },

        # Tool: MBRCheck
        # Example: https://www.bleepingcomputer.com/forums/t/427851/fake-mbr-invisible-webpages-opened/
        'MBRCheck': {
            r'MBRCheck, version': False
        },

        # Some tools which we do not know where Parvez found
        'Unknown':{
            # Not found, keep it as it is
            r' DDS (Ver_': False,
            # Pattern: Dump\sfile\s+:
            # Example: http://localhost:9021/files/BACKUP/project/BB/data/raw-data/BleepingComputerHtmlFiles/am-i-infected-what-do-i-do/tid-link-525755_3.html
            r'Dump File\s+: ': True,
            'LOG FILE:': False
        }
    }

    posttext = post.text
    positions = {}
    matched_tools = [tool for tool in tools_check.keys() if tool.lower() in posttext.lower()] + ['Unknown']
    for tool in matched_tools:
        patternSet = tools_check[tool]
        for pattern, isRegex in patternSet.items():
            if isRegex:
                pattern_search = re.search(pattern, posttext, re.IGNORECASE)
                if pattern_search:
                    position = pattern_search.start()
                else:
                    position = None
            else:
                try:
                    position = posttext.index(pattern)
                except ValueError:
                    position = None
            if position:
                positions[position] = {
                    'Tool': tool,
                    'MatchedPattern': pattern
                }
    text = None
    if positions:
        # If positions is not empty
        start_index_log = min(positions.keys())
        text = posttext[0:start_index_log]
    else:
        # Check if this is log
        if not is_log(post):
            text = post.text

    return text


def is_log(post):
    #the goal is to guess whether this is a log or not
    text = post.text.lower()
    if text.count("127.0.0.1") > 15:
        return True
    elif text.count("c:\\users\\") > 15:
        return True
    elif text.count("\\program files\\") > 15:
        return True
    elif text.count("\\windows\\system32") > 15:
        return True
    elif text.count("hklm\\") > 15:
        return True
    else:
        return is_log_finer(text)


def is_log_finer(text, thresh=0.685):
    """
    Determine if the input string is log with a spellchecker
    The threshold is determined through the experiment with over 500 samples of data points mixed with log and NL (manually labeled)
    Parameters
    ----------
    text
    thresh

    Returns
    -------

    """
    words = process_terms(text)
    if words:
        known_words = checker.known(words)
        unknown_words = checker.unknown(words)
        known_words_ratio = len(known_words)/(len(known_words) + len(unknown_words))
        if known_words_ratio < thresh:
            return True
    return False
