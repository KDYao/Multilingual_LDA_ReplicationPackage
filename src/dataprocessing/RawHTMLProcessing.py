import datetime
import json
import pickle
import subprocess
import time
import logging
from bs4 import BeautifulSoup
from multiprocessing import Process
from nltk import word_tokenize

from src.dataprocessing import Post
from src.dataprocessing.WordProcess import *
from src.resources.utils.utils import getPath, check_existance, parse_args_data_processing

logging.basicConfig(
    level=logging.INFO)


class Serialization:
    def __init__(self):
        self.postIdToFilesDic = {}
        self.postIdToPostsDic = {}

    def parse(self, post_id, page_num, html_content):
        post_list = []
        soup = BeautifulSoup(html_content, 'html.parser')
        title_element = soup.find("title")
        keyword_element = soup.find("meta", attrs={'name': "keywords"})

        meta_elements = soup.find_all("meta")
        element_counter = -1
        for elem in meta_elements:
            if elem.has_attr("name") and elem["name"] == "description " and elem.has_attr("content"):
                post_content = elem["content"]

        div_elements = soup.find_all("div", {'class': 'post_body'})
        for elem in div_elements:

            # collect the post date
            commentTimeElement = elem.find("abbr",
                                           attrs={'itemprop': 'commentTime', 'class': 'published', 'title': True})
            inner_elem = elem.find("div", attrs={'itemprop': 'commentText', 'class': 'post entry-content'})
            if inner_elem is not None:
                element_counter = element_counter + 1
                post_content = inner_elem.text.replace("\n", " ")
                internal_links = []
                external_links = []

                # collect links
                links = inner_elem.findAll("a", attrs={"href": True})
                for link in links:
                    if link["href"].startswith("http://www.bleepingcomputer.com"):
                        internal_links.append(link["href"])
                    else:
                        external_links.append(link["href"])

                post = Post()
                post.id = post_id
                if commentTimeElement is not None:
                    date_time_obj = str(commentTimeElement["title"])
                    post.published = str(datetime.datetime.strptime(date_time_obj[0:date_time_obj.rindex("-")], '%Y-%m-%dT%H:%M:%S'))

                # print("Page Number: "+str(page_num)+" Element Counter: "+str(element_counter)+" Title: "+title_element.text+"  ")
                if page_num == 1 and element_counter == 0:
                    post.title = title_element.text
                    post.post_type = 1
                    if keyword_element is not None:
                        post.tags = keyword_element["content"]
                else:
                    post.post_type = 2
                post.text = post_content
                post.internal_links = internal_links
                post.external_links = external_links
                post_list.append(post)
        return post_list

    def initialize(self, data_path):

        start_time = time.time()
        # Step-1: collect file names
        print("start scanning ... {}".format(data_path))
        list_files = [(f.name) for f in os.scandir(data_path) if f.is_file() and f.name.endswith(".html") is True]
        file_counter = 0
        print("Total collected files for data path: {} is: {}".format(data_path,len(list_files)))

        # Step-2: group files based on the question id
        for (file_name) in list_files:
            file_counter = file_counter + 1
            underscore_pos = file_name.index("_")
            post_id = int(file_name[len("tid-link-"):file_name.index("_")])
            page_num = int(file_name[(underscore_pos + 1):len(file_name) - len(".html")])
            if post_id in self.postIdToFilesDic:
                self.postIdToFilesDic[post_id].append(file_name)
            else:
                self.postIdToFilesDic[post_id] = [file_name]

        # Step-3: now group all posts by their associated question id
        progress = 0
        file_with_error = 0
        for postId in self.postIdToFilesDic.keys():
            progress = progress + 1
            if progress % 1000 == 0:
                print("Current progress: {}/{}".format(progress, len(self.postIdToFilesDic)))

            post_list = []
            for file_name in self.postIdToFilesDic[postId]:
                #print("File: {}".format(file_name))
                # print("FilePath: "+str(file_path))
                underscore_pos = file_name.index("_")
                page_num = int(file_name[(underscore_pos + 1):len(file_name) - len(".html")])

                with open(os.path.join(data_path,file_name), encoding="utf8") as f:
                    file_content = f.read()
                    for l in self.parse(postId, page_num, file_content):
                        post_list.append(l)

            # print("Post Id: {} Page Number:{} ".format(postId, page_num))
            # each post list should have a question

            # collect the question post
            question_post = None
            for post in post_list:
                if post.post_type==1:
                    question_post = post
                    break

            if question_post is None:
                print("Error in post id:{}  {}".format(postId, data_path))
                print("Total Post List: {}".format(len(post_list)))
                for file_name in self.postIdToFilesDic[postId]:
                    print("File name: {}".format(file_name))
                for post in post_list:
                    post.print()
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("-------------------------------------------------------")
                #raise Exception("Error in printing post "+str(postId))
                file_with_error = file_with_error + 1
            else:
                self.postIdToPostsDic[postId] = (question_post.export_post(), [x.export_post() for x in post_list])

        print("File With Error: {}".format(file_with_error))
        print("--- time to run initialization in %s seconds ---" % (time.time() - start_time))
        print("Total Posts: {}".format(len(self.postIdToPostsDic)))

    def serialize(self, serialize_file_path):
        start_time = time.time()
        with open(serialize_file_path, 'wb') as serialize_json:
            serialize_json.write(json.dumps(self.__dict__).encode())
        print("---serialization time in %s seconds ---" % (time.time() - start_time))


#@run_async_multiprocessing
def serialize(sec, dataroot):
    """
    Serializing script that used for parallelization
    It also examines whether a folder is compressed
    Parameters
    ----------
    sec

    Returns
    -------

    """
    rawDataDir = os.path.join(*[dataroot, 'raw-data', 'BleepingComputerHtmlFiles'])
    sec_dir = os.path.join(rawDataDir, sec)
    sec_zip_f = os.path.join(rawDataDir, sec + '.7z')
    if not os.path.isdir(sec_dir):
        if os.path.isfile(sec_zip_f):
            # Unzip compressed file
            p = subprocess.Popen('7z x {} -o{}'.format(sec_zip_f, rawDataDir))
            p.wait()
        else:
            raise FileNotFoundError('No related files to section {} found in {}'.format(sec, rawDataDir))
            #return
    section = Serialization()
    section.initialize(sec_dir)
    section.serialize(os.path.join(*[dataroot, 'serialized-raw-data', sec + '.ser']))

    # remove folder if zip is found to save some space
    if os.path.isdir(sec_dir) and os.path.isfile(sec_zip_f):
        os.rmdir(sec_dir)


class TermExtraction:
    """
    This is used to be the DataCollection.py
    It reads serialized data and convert them to lists of terms
    """

    def createDataDictFile(self, section_name, dataroot, isExport=True, q=None):
        # load the serialized files one after another
        stop_words = get_stopwords()
        section_dict = {}  # key is the section name, value is a list of tuples (questionPost, postList)

        serDataDir = os.path.join(*[dataroot, 'serialized-raw-data'])
        sec_f = os.path.join(serDataDir, section_name + '.ser')
        check_existance(sec_f, 'f')

        # Load serialized file
        with open(sec_f, "rb") as serializeFile:
            section = json.load(serializeFile)
            print("Path: {} {}".format(sec_f, len(section['postIdToFilesDic'])))
            tuple_list = []
            print("--------------------------------------------------------------------------------------------------")
            for postId in section['postIdToPostsDic']:
                (questionPost, postList) = section['postIdToPostsDic'][postId]
                questionPost = Post(questionPost)
                postList = [Post(x) for x in postList]
                for post in postList:
                    if " Event log errors: " in post.text:
                        tuple_list.append((questionPost, postList))
                        break
            section_dict[section_name] = tuple_list

        data_dict = {}
        for section_name in section_dict.keys():
            tuple_list = section_dict[section_name]
            postIdToSourceTerms = {}
            postIdToDescriptionTerms = {}
            postIdToTextTerms = {}
            # load the serialized files one after another
            stopwords_nl = get_stopwords(isLog=False)
            stopwords_log = get_stopwords(isLog=True)
            for (questionPost, postList) in tuple_list:
                source_terms = []
                description_terms = []
                text_terms = []
                for post in postList:
                    # The rule here is to first remove punctuations, then find non-stop words which has at least three characters
                    if post.title is not None:
                        text_terms.extend(process_terms(post.title, stopwords=stopwords_nl,
                                                        minlen=2))  # this is a question post because it has title
                        if post.tags is not None:
                            # tokenize_tags = [term for term in re.findall(r"[\w']+", post.tags) if
                            #                  term not in string.punctuation]
                            # text_terms.extend([word for word in word_tokenize(post.title) if
                            #                    word.lower() not in stop_words and len(word) > 2])
                            tokenize_tags = process_terms(post.tags, stopwords=stopwords_nl, minlen=2)
                            text_terms.extend(tokenize_tags)
                    if 'MiniToolBox by Farbar' in post.text:
                        # Text prior to this line may contain normal text but definitely not all text
                        # There are bunches of log before this line, which was previously identified as text
                        # We will try to find the beginning of the
                        index = post.text.index('MiniToolBox by Farbar')
                        text = post.text[0:index]
                        text_terms.extend(process_terms(text, stopwords=stopwords_nl, minlen=3))
                        pattern_source = r'Source: (.*?)\)'
                        pattern_description = r'Description:(.*?)Error:'

                        source_text = " ".join(re.findall(pattern_source, post.text))
                        tokenized_source_text = process_terms(source_text, stopwords=stopwords_log,
                                                              wordbreak=True, minlen=3)

                        description_text = " ".join(re.findall(pattern_description, post.text))
                        tokenized_description_text = process_terms(description_text, stopwords=stopwords_log,
                                                                   wordbreak=True, minlen=3)
                        source_terms.extend(tokenized_source_text)
                        description_terms.extend(tokenized_description_text)
                    else:
                        # Identify text
                        text = get_text(post=post)
                        if text:
                            text_terms.extend(process_terms(text, stopwords=stopwords_nl, minlen=3))
                postIdToSourceTerms[questionPost.id] = source_terms
                postIdToDescriptionTerms[questionPost.id] = description_terms
                postIdToTextTerms[questionPost.id] = text_terms
            data_tuple = (postIdToSourceTerms, postIdToDescriptionTerms, postIdToTextTerms)
            if q:
                q.put(data_tuple)

            data_dict[section_name] = (
            postIdToSourceTerms, postIdToDescriptionTerms, postIdToTextTerms)

        if isExport:
            self.export_data_dict(dataroot=dataroot, data_dict=data_dict, section_name=section_name)
        return data_dict

    def export_data_dict(self, dataroot, data_dict, section_name):
        # save the data_dict
        out_f = os.path.join(*[dataroot, 'post-processed-data', 'data_dict_{}.pkl'.format(section_name)])
        with open(out_f, 'wb') as handle:
            pickle.dump(data_dict, handle)
        print("Complete writing data of section: {} dictionary to the file {}".format(section_name, out_f))

    def print_data_dict(self, data_dict):
        # print the data dictionary
        for entry in data_dict.keys():
            tuple_list = data_dict[entry]
            print("Section: {} Tuple size: {}".format(entry, len(tuple_list)))

def term_extraction(sec, dataroot, isExport=True):
    term_ext = TermExtraction()
    term_ext.createDataDictFile(section_name=sec, dataroot=dataroot, isExport=isExport)

if __name__ == '__main__':
    section_names = [
        'am-i-infected-what-do-i-do',
        'anti-virus-anti-malware-and-privacy-software',
        'dospdaother',
        'external-hardware',
        'general-security',
        'internal-hardware',
        'linux-unix',
        'mac-os',
        'networking',
        'network-streaming-devices',
        'questions-and-advice-for-buying-a-new-computer',
        'service-providers',
        'system-building-and-upgrading',
        'web-browsingemail-and-other-internet-applications',
        'web-site-development',
        'windows-7',
        'windows-8-windows-81',
        'windows-10-support',
        'Windows-95-98-ME',
        'windows-crashes-and-blue-screen-of-death-bsod-help-and-support',
        'windows-vista',
        'virus-trojan-spyware-and-malware-removal-help'
    ]
    dataroot = getPath('ROOT')

    args, _ = parse_args_data_processing()

    if args.sections:
        section_names = [x.strip() for x in args['sections'].split(',')]

    # The multiprocessing wrapper seems to be broken in python 3.8
    # https://discuss.python.org/t/is-multiprocessing-broken-on-macos-in-python-3-8/4969
    if args.serialization:
        proc = [Process(target=serialize, args=(sec, dataroot, )) for sec in section_names]
        [p.start() for p in proc]
        [p.join() for p in proc]

    if args.termExtraction:
        proc = [Process(target=term_extraction, args=(sec, dataroot, )) for sec in section_names]
        [p.start() for p in proc]
        [p.join() for p in proc]