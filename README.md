# LLM-based metadata cleanup (ROR)

This is just an example on how to do metadata cleanup using an LLM
(embedding) and a vector store, using the [ROR](https://ror.org/)
dataset as an example.

There's an `index.py` script which takes the path to the ROR CSV as
a parameter and generates a `vectorstore.json` which contains vector
representations of all ROR institutions' names
according to the model.

```sh
$ python index.py ror.csv
```

Then, you can provide the `search.py` script with the path to a text
file containing the raw institution names, one per line.

```sh
$ python search.py queries.txt
```

The script will return the most likely match on the ROR DB for each of
the entries.

The script requires a modern (as of 2025) version of Python as well as
[`langchain`](https://www.langchain.com/). Don't forget to install the
requirements first:

```py
$ pip install -r requirements.txt
```

You need to have [`ollama`](https://ollama.com) installed and running and to
pull the corresponding model beforehand:

```sh
$ ollama pull sam860/granite-embedding-multilingual:107m-F16
```

I tested this with several models, but the one which seems to work the best
across languages is [`sam860/granite-embedding-multilingual:107m-F16`](https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb).
