import os
import os.path
import numpy as np
import string
import unicodedata
from itertools import repeat
from itertools import chain
from functools import partial
from itertools import compress
from collections import Counter
from multiprocessing.pool import Pool

import constants as c
from experiment_clef import _count_words
from text2vec import clean

def tokenize(doc):
    return doc.split()

def compute_idf_weights(documents):
    """
    Returns a mapping { term: IDF_term }
    :param text: list of documents in corpus
    :param processes: paralellization parameter
    :return:
    """

    collection_size = len(documents)
    flat_words = list(chain(*documents))

    doc_frequencies = dict(Counter(flat_words))
    idf_mapping = {term: np.log(float(collection_size) / doc_frequency) for term, doc_frequency in doc_frequencies.items()}
    return idf_mapping

def compute_tf_weights(documents, processs = 1):
    pool = Pool(processes=processs)
    document_distributions = pool.map(_count_words, documents)
    pool.close()
    pool.join()
    collection_size = sum([sum(document.values()) for document in document_distributions])
    collection_distribution = Counter()
    for document in document_distributions:
        collection_distribution.update(document)  # { token: frequency }
    collection_distribution = dict(collection_distribution)
    return collection_distribution

def sampleSent(doc, tfidf, k = 5):
    k = 5
    doc = doc.split(".")
    scores = []
    for i, s in enumerate(doc):
        total = sum([tfidf[w.lower().strip(string.punctuation)] for w in s.split() if w in tfidf])
        scores.append([total, i])
    
    scores.sort(key = lambda x : x[0], reverse = True)
    return ". ".join(doc[i].strip() for _, i in scores[:k])

def getTfidf(documents, processs = 4):
    pool = Pool(processes=processs)

    # print("Start preprocessing data %s" % timer.pprint_lap())
    clean_to_lower = partial(clean, to_lower=True)
    # tokenize_doc_language = partial(tokenize, language=dlang_long, exclude_digits=True)
    documents = pool.map(clean_to_lower, documents)
    documents = pool.map(tokenize, documents)
    pool.close()
    pool.join()

    tf_weights = compute_tf_weights(documents)
    idf_weights = compute_idf_weights(documents)
    tfidf = {k : tf_weights[k]*idf_weights[k] for k in tf_weights}
    return tfidf

def getDocs(qid, query2docs, allDocs, relDocsCount, nonRelDocsCount, sample = True):
    if sample :
        relDocs = np.random.choice(query2docs[qid], min(relDocsCount, len(query2docs[qid])), replace = False)
    else:
        relDocs = query2docs[qid]
    
    leftDocs = allDocs - set(relDocs)
    nonRelDocs = np.random.choice(list(leftDocs), nonRelDocsCount, replace = False)
    return relDocs, nonRelDocs

if __name__ == '__main__':
    
    np.random.seed(123)
    processs, k = 4, 10
    documents = [] #list of documents
    docid2doc = {} #list of docids
    queries = []
    qidtoq = {}
    allDocs = set(docid2doc.keys())
    tfidf = getTfidf(documents, processs)
    pool = Pool(processes=processs)
    subsetDocs = pool.starmap(sampleSent, zip(documents, repeat(tfidf), repeat(k)))
    pool.close()
    pool.join()
    assert len(subsetDocs) == len(documents)
    # saveSubsetDocs()
    # loadSubSetDocs()
    relDocsCount, nonRelDocsCount = 2, 2

    query2docs = defaultdict(list)
    with open("data.tsv", "w") as f:
        for qid, i in qidtoq.items():
            #add relevant documents
            relDocs, nonRelDocs = getDocs(qid, query2docs, allDocs, relDocsCount, nonRelDocsCount)
            for docid in relDocs:
                f.write("\t".join([str(1), qid, docid, queries[qidtoq[qid]], subsetDocs[docid2doc[docid]]] ))

            #add nonrelevant documents
            for d in nonRelDocs:
                f.write("\t".join([str(0), qid, docid, queries[qidtoq[qid]], subsetDocs[docid2doc[docid]]] ))
    