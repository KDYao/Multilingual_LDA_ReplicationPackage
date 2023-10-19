# How is the bleeping computer data organized?
   
A thread consists of multiple posts (questions and answers). Each question has an id, that is also uniquely identify the thread and represented by a post object (`PythonDataCollection/Post.py`).
Each question contains a title, a set of tags and a body (represented by text field of the post object). The answers also have ids but they contain the same id of the associated question.

We have downloaded links to bleeping computer questions for each section. These are stored in `data/raw-data/bleeping_computer_links.zip` files. Each folder in there represents a section. Then we create another scraper to download the content of each thread. The data is stored in the `data/raw-data/BleepingComputerHtmlFiles` folder.

Next, we serialized the data and stored in `data/serialized-raw-data` folder for easy access. You can use pickle library to read the content. Each file in the folder represents data of a section. Each serailized file (i.e., `am-i-infected-what-do-i-do.ser`) represents a Section object (see `PythonDataCollection/Section.py`). The first part represents the section name (i.e., `am-i-infected-what-do-i-do`) in the bleeping computer website. Each section object has two dictionaries: `postIdToFilesDic` and `postIdToPostsDic`. For each post id (i.e., question id), it gives the associated files or a tuple (i.e., (`questionPost`, `postList`)). The postList consists of the question and associated answers of a thread. You can check the `PostTypeId` files of a post to determine whether a post is a question or an answer. The `questionPost` explicitly helps you to find the question. However, `questionPost` also exists in the `postList`.

To learn how to read the serialized file and read the content, go to this file: `PythodDataCollection/DataCollection.py` and check the first loop. What the file does is that it first read the threads of each section and determine which thread contain the log type we are interested in (i.e., containing text " Event log errors: ") and then store them in sectionDict. The key is the section name and the value is list if tuples (i.e., (questionPost, postList)) 

# How do we extract log and text terms from the posts?

Now, we need to extract the log term and text terms from each thread. That is what the second loop does in the `PythodDataCollection/DataCollection.py`. The log part is divided into two areas (one is the source and another is the description). Thus, we keep the content into `postIdToSourceTerms`, `postIdToDescriptionTerms`, `postIdToTextTerms`. As you can understand, the first two dictionaries refer to log terms.
Finally, we use the following code to store them:

```python
data_dict[section_name] = (postIdToSourceTerms,postIdToDescriptionTerms,postIdToTextTerms,postIdToThreads)
```

The result is stored in the `data.dict` file. Thus, from the `data.dict` file, we can create the documents that will act as an input for the topic modelling algorithm.

# How do we remove other log types from the posts?

Pls. check the `Preprocessor/DocumentPreprocessing.py` file. There is a method that we use to determine whether the post contains any log types we are not interested in (see check_log_position method). It uses a rule based approach to find the start position of the log. We keep everything from the start position of the post to the start position of the log and remove the rest. Finding the end position is not an easy task due to the unstructured natures of the content. It might be the case that some text left at the end but what I found that mmajority of the cases it just the log.

The `stopwords.py` file gives us the list of stop words. We use python nltk library word_tokenize method to tokenize texts. But it has its own limitations. Thus, we have another method (called word_breaker) to break the long texts that nltk library could not break into small pieces.

As an example, the posts occassionally contain `..................` or `====================` or `close.If` (i.e., failed to identify that close and If should be two words and . is the end of a sentence). The `word_breaker` method helps us to tackle those cases so that we can purform another filtering to remove unwated words.

# How to create documents as input for a topic modelling algorithm?

As an example, I will explain creating documents for the option-1 topic modelling in our paper. The idea is we load the data.dict. This will give us the souce terms, description terms (i.e., terms that are associated with the log parts) and text terms (i.e., non log terms). All we did is to appeand TEXT or LOG prefix to determine where the word is coming for and create a document. pls. see `PythonDataCollection/Option1Corpus.py` file. We serialize the output using pickle. The list of documents will be used as an input to the topic modelling algorithm.
