# Multilingual_LDA_ReplicationPackage
Replication package of our paper "Finding Associations between Natural and Computer Languages: A Case-Study of Bilingual Lda Applied to the Bleeping Computer Forum Posts"


# Documentation

# Setups

This part introduces the setup of our code and external tools.

## Requirements

We require the following tools to be installed before setups:

- Apache Ant ([https://ant.apache.org/](https://ant.apache.org/))
- Mallet
- Python 3

For mallet setup, you can use the following command:

```bash
# Setup Mallet
mkdir -p ../resources/Mallet
git clone https://github.com/mimno/Mallet.git ../resources/Mallet
#module load ant
# JRE mem allocation, in case ant fails due to heap error
set ANT_OPTS=-Xmx2G -XX:MaxPermSize=512m
(cd ../resources/Mallet && ant)
```

Make sure you find Mallet is successfully installed. Or you can also download executable Mallet from its homepage.

## Code setups

Use the following command to automatically setup required packages:

```python
# Remove --user argument if you are using a virtual environment
pip3 install -e --user main/
```

## Path setups

In file `$PROJHOME/src/resources/utils/utils.py`, you will need to define the path parameters to be used in your environment. 

There are three path parameters, which are:

- MALLET (the home folder of Mallet)
- ROOT(Root directory of the project data)
- CORPUS_NAME (the pickled filename of the project).

The logic is to choose path according to the current running environment by OS name and machine name (since there are not too many pathes to setup, we just configure pathes in the code instead of in a separate properties file). 

<aside>
💡 You may need to add the OS name (and maybe machine name), just like other examples. Or you can simplify this function as well, if you are not running the code in multiple environments.

</aside>

# Processing Data

## Input data

### Intro

An example of the input data should be like this:

```python
{
    "id": "mixed.1",
    "posts": [
      {
        "id": "foo.1.0",
        "paragraphs": [
          {
            "id": "foo.1.0.0",
            "language": "FOO",
            "text": "footerm1, footerm2, footerm3"
          }
        ]
      },
      {
        "id": "bar.1.0",
        "paragraphs": [
          {
            "id": "bar.1.0.0",
            "language": "BAR",
            "text": "barterm1, barterm2, barterm3"
          }
        ]
      }
    ]
}
```

This requires some code changes from Dr. Malton's generator. 

The purpose of such reformatting is to make sure that at least 2 different languages are included in each post.

The tokens in selected JSON files will be extracted and aggregated by post, then by language. The post-processed data will be stored in a pickle file, which will be used to create different corpuses for different LDA models.

### Command

**Execute from file:** `$PROJHOME/src/dataprocessing/TermExtraction.py`

**Args:**

- —input/-i (requied): The path of the JSON file.
- —output/-o: The output of the post-processed pickle file. Will use the default folder (`$ROOT/post-processed-data`) if not specified.
- —processing/-p: Process terms following defined rules. Currently, we disable processing by default. The processing rule should be double checked when applied on real data.
- —languages/-l: The languages to be considered. Will consider all languages if not specified.

```bash
# Example
python3 TermExtraction.py -i ../testdata/foobar_reformatted.json -o ../data/post-processed-data/foobar_reformatted.pkl
```

<aside>
💡 After discussion with Dr. Malton, we realize that maybe other semi-format material such as log, scripts, commands will be considered as different languages in the future work. Therefore, we make some changes to the code and now it should support multiple languages instead of just two (i.e., Log/Text).

</aside>

## Generate Corpus

Next, based on the post-processed data, we generate corpus for each LDA model. 

Specifically, we consider a separate corpus for each language that only include terms of that language.

We also aggregate terms from different languages to build another corpus for LDA1. 

For example, if we have two languages: FOO and BAR, we will get the following corpus files by the end of this step:

- FOO.pkl
- BAR.pkl
- FOO_BAR.pkl

Since these files should not be mutable during running, we just generate them once without overwriting them at runtime.

The generated corpus will be preserved under `$ROOT/documents` folder.

Check `$PROJHOME/src/dataprocessing/BuildCorpus.py` for detailed implementations. 

<aside>
💡 You don't need to run this step separately. The corpus will be checked (and generated if needed) before running an LDA model.

</aside>

# Running experiments

We will cover the experiments of running a single LDA model and tuning a single LDA model.

<aside>
💡 Before performing the following steps. Make sure to finish the **Path Setup** Step.

</aside>

## Running Single LDA model

In file `$PROJHOME/src/LDA/RunLDAOpts.py`, use the following arguments to run each LDA model.

- —LDAType (required): The type of LDA model. It should be either LDA1, LDA2, or LDA3.
- —languages (required): The languages to be considered when analyzing. Separate languages by comma.
- —numTopics (default to 10): The number of topics
- —alpha (default to 5.0)
- —beta (default to 0.01)
- —saveLDAVis (disabled by default): Preserve the results with a LDAVis graph

An example below shows how to run single LDA model:

```python
python3 RunLDAOpts.py --LDAType LDA1 --languages=FOO,BAR --numTopics 10 --alpha 5.0 --beta 0.01 --saveLDAVis
```

## Tuning single LDA model

In file `$PROJHOME/src/HyperTuning/HyperParamOptimization.py`, use the following arguments to tune each LDA model. 

- —LDAType (required): The type of LDA model. It should be either LDA1, LDA2, or LDA3.
- —languages (required): The languages to be considered when analyzing. Separate languages by comma.
- —numTopicsRange (default 5-10): The range of the selection of number of topics.
- —alphaRange (default 1.0-30.0): The range of the selection of alpha.
- —betaRange(default 0.01-5.00): The range of the selection of beta.
- —metric (default 2): The evaluation metric index

Since we integrate the parameter setting with OpenTuner, so the following parameters setting from OpenTuner is also support. For more details on these arguments, please check this [link](https://github.com/jansel/opentuner/blob/master/examples/tutorials/gettingstarted.md).

- —stop-after (disabled by default): Total seconds allocated for tuning. No more tuning iterations will be performed after the given time.
- —no-dups (disabled by default): The --no-dups flag hides warnings about duplicate results.

The evaluation metrics indexes are:

```
COHERENCE = 1
PERPLEXITY = 2
LOGALIGNMENT = 3
INTERCORPUSRATIO = 4
AGGREGATED = 5
```

The tuning process will minimize perplexity or maximize the other evaluation metrics.

An example on tuning LDA is:

```python
python3 HyperParamOptimization.py --LDAType LDA2 --languages FOO,BAR --numTopicsRange 10-800 --alphaRange 0.001-50.0 --betaRange 0.001-50.0 --no-dups --metric 5 --stop-after 86400
```

# Analysis

An analysis script can be found at `$PROJHOME/src/analytics/graphics/MonitoringMetrics.py`.

You will need to manually setup the LDAs and metrics to be evaluated at the designated directory.

The will update the best parameter results at `$PROJHOME/result/bestparam/BestParam.json`

A graph of the best performing results based on iterations will be generated under the `Graph` folder under the same designated directory.