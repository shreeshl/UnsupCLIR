import os
import os.path
import numpy as np
import string
import unicodedata
from itertools import repeat
from itertools import chain
from functools import partial
from itertools import compress
from collections import Counter, defaultdict
from multiprocessing.pool import Pool
from collection_extractors import extract_italian_lastampa
import constants as c
from experiment_clef import _count_words
from text2vec import clean
from load_data import load_clef_documents, load_queries

def tokenize(doc):
    return doc.split()

def prepare_experiment(doc_dirs, limit_documents, query_file, limit_queries, query_language = 'en'):
    """
    Loads documents, evaluation data and queries needed to run different experiments on CLEF data.
    :param doc_dirs: directories containing the corpora for a specific CLEF campaign
    :param limit_documents: for debugging purposes -> limit number of docs loaded
    :param query_file: CLEF Topics (i.e., query) file
    :param limit_queries: for debugging purposes -> limit number of queries loaded
    :param query_language: language of queries
    :param relevance_assessment_file: relevance assesment file
    :return:
    """
    if limit_documents is not None:
        limit_documents -= 1
    documents = []
    doc_ids = []
    limit_reached = False
    for doc_dir, extractor in doc_dirs:
        if not limit_reached:
            for root, dirs, files in os.walk(doc_dir):
                for file in files:
                    if '.DS' in file : continue
                    tmp_doc_ids, tmp_documents = load_clef_documents(os.path.join(root, file), extractor, limit_documents)
                    documents.extend(tmp_documents)
                    doc_ids.extend(tmp_doc_ids)
                    if len(documents) == limit_documents:
                        limit_reached = True
                        break

    query_ids, queries = load_queries(query_file, language_tag=query_language, limit=limit_queries)
    # print("Documents loaded %s" % (timer.pprint_lap()))
    return doc_ids, documents, query_ids, queries


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
    

    # Prepare italian CLEF data
    limit_documents = 1000
    limit_queries = 10
    year = "2001"
    it_lastampa = (c.PATH_BASE_DOCUMENTS + "italian/la_stampa/", extract_italian_lastampa)
    italian = {"2001": [it_lastampa]}

    _all = {"italian": italian}

    doc_dirs = _all["italian"][year]
    current_path_queries = c.PATH_BASE_QUERIES + year + "/Top-en" + year[-2:] + ".txt"
    np.random.seed(123)
    processs, k = 4, 10
    docids, documents, qids, queries = prepare_experiment(doc_dirs, limit_documents, current_path_queries, limit_queries)
    docid2doc = {d:i for i, d in enumerate(docids)}
    qidtoq    = {str(q):i for i, q in enumerate(qids)}

    allDocs = set(docid2doc.keys())
    tfidf = getTfidf(documents, processs)
    pool = Pool(processes=processs)
    subsetDocs = pool.starmap(sampleSent, zip(documents, repeat(tfidf), repeat(k)))
    pool.close()
    pool.join()
    assert len(subsetDocs) == len(documents)
    # saveSubsetDocs()
    # loadSubSetDocs()
    relDocsCount, nonRelDocsCount = 3, 10

    query2docs = defaultdict(list)
    with open("out", "r") as f:
        for line in f:
            line = line.split()
            query2docs[line[0][-2:]].append(line[2])


    with open("data.tsv", "w") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for qid, i in qidtoq.items():
            if qid not in query2docs :
                print("No relevance judgements available for query %s"%(qid))
                continue
            #add relevant documents
            relDocs, nonRelDocs = getDocs(qid, query2docs, allDocs, relDocsCount, nonRelDocsCount)
            for did in relDocs:
                tsv_writer.writerow([str(1), qid, did, queries[qidtoq[qid]], subsetDocs[docid2doc[did]]])

            #add nonrelevant documents
            for did in nonRelDocs:
                tsv_writer.writerow([str(0), qid, did, queries[qidtoq[qid]], subsetDocs[docid2doc[did]]])
    