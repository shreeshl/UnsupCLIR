import os
import os.path
import numpy as np
import string
import unicodedata
import csv
from itertools import repeat
from itertools import chain
from functools import partial
from itertools import compress
from collections import Counter, defaultdict
from multiprocessing.pool import Pool
from collection_extractors import extract_italian_lastampa, extract_italian_sda9495, extract_finish_aamuleth9495
import constants as c
from experiment_clef import _count_words
from text2vec import clean
from load_data import load_clef_documents, load_queries, load_relevance_assessments
import pickle as p

def tokenize(doc):
    """
    naive tokenization based on whitespace
    """
    return doc.split()

def prepare_experiment(doc_dirs, limit_documents, query_file, limit_queries, relevance_assessment_file = None, query_language = 'en'):
    """
    Loads documents, evaluation data and queries needed to run different experiments on CLEF data.
    :param doc_dirs: directories containing the corpora for a specific CLEF campaign
    :param limit_documents: for debugging purposes -> limit number of docs loaded
    :param query_file: CLEF Topics (i.e., query) file
    :param limit_queries: for debugging purposes -> limit number of queries loaded
    :param query_language: language of queries
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
                    if 'sda_italian_94' in root : print(file)
                    if '.DS' in file : continue
                    tmp_doc_ids, tmp_documents = load_clef_documents(os.path.join(root, file), extractor, limit_documents)
                    documents.extend(tmp_documents)
                    doc_ids.extend(tmp_doc_ids)
                    if limit_documents is not None and len(documents) == limit_documents:
                        limit_reached = True
                        break

    if relevance_assessment_file:
        relass = load_relevance_assessments(relevance_assessment_file)
    query_ids, queries = load_queries(query_file, language_tag=query_language, limit=limit_queries)
    # print("Documents loaded %s" % (timer.pprint_lap()))
    return doc_ids, documents, query_ids, queries, relass if relevance_assessment_file else None

def unique(document):

    return list(set(document))

def compute_idf_weights(documents, processes = 4):
    """
    Returns a mapping { term: IDF_term }
    :param documents: list of documents in corpus
    """

    collection_size = len(documents)
    pool = Pool(processes=processes)
    documents = pool.map(unique, documents)
    pool.close()
    pool.join()
    flat_words = list(chain(*documents))

    doc_frequencies = dict(Counter(flat_words))
    idf_mapping = {term: np.log(float(collection_size) / doc_frequency) for term, doc_frequency in doc_frequencies.items()}
    return idf_mapping

def compute_tf_weights(documents, processs = 4):
    """
    Returns a mapping { term: tf_term }
    :param documents: list of documents in corpus
    :param processes: parallelization parameter
    """

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

def getTfidf(documents, processs = 4):
    """
    Returns tfidf score dictionary for each word
    :param documents: list of documents in corpus
    :param processes: parallelization parameter
    """

    pool = Pool(processes=processs)

    # print("Start preprocessing data %s" % timer.pprint_lap())
    clean_to_lower = partial(clean, to_lower=True)
    # tokenize_doc_language = partial(tokenize, language=dlang_long, exclude_digits=True)
    documents = pool.map(clean_to_lower, documents)
    documents = pool.map(tokenize, documents)
    pool.close()
    pool.join()

    tf_weights = compute_tf_weights(documents, processs)
    idf_weights = compute_idf_weights(documents, processs)
    tfidf = {k : tf_weights[k]*idf_weights[k] if k in tf_weights else 1e-7*idf_weights[k] for k in idf_weights}
    return tfidf

def sampleSent(doc, tfidf, k = 1000):
    """
    Returns top k sentences based on tf-idf scoring
    :param doc: given (docid, document)
    :tfidf : dict with tfidf score for each word
    :k : maximum total words in the returned sentences
    """

    _id, doc = doc
    doc = doc.strip(string.punctuation).split(".")
    scores = []
    for i, s in enumerate(doc):
        total = sum([tfidf[w.lower().strip(" " + string.punctuation)] for w in s.split() if w in tfidf])
        scores.append([total, i])
    
    scores.sort(key = lambda x : x[0], reverse = True)
    out = []
    total = 0
    for _, i in scores:
        if len(doc[i]) > k: continue
        total += len(doc[i])
        if total <= k or len(out) == 0:
            out.append(doc[i].strip())
        else:
            break
    if len(out) == 0 : out.append(doc[scores[0][1]].strip()[:1000])
    return _id, ". ".join(out)

def getDocs(qid, query2docs, allDocs, relDocsCount, nonRelDocsCount, sample = True):
    """
    Returns relevant and nonrelvant documents for the given query qid
    :param qid : given query id
    :param query2docs : dict between query and it's pseudo relevant documents
    :param allDocs: all document ids in corpus
    :param relDocsCount: no of relevant documents to return
    :param nonRelDocsCount : no of non relevant documents to return
    :param sample: sample relevant documents or return all of them
    """

    if sample :
        relDocs = np.random.choice(query2docs[qid], min(relDocsCount, len(query2docs[qid])), replace = False)
    else:
        relDocs = query2docs[qid]
    
    leftDocs = allDocs - set(relDocs)
    nonRelDocs = np.random.choice(list(leftDocs), nonRelDocsCount, replace = False)
    return relDocs, nonRelDocs

if __name__ == '__main__':
    

    # Prepare italian CLEF data
    # mode = "drmm"
    load = False
    np.random.seed(123)
    limit_documents = None
    limit_queries = None
    relDocsCount, nonRelDocsCount = 50, 100

    it_lastampa = (c.PATH_BASE_DOCUMENTS + "italian/la_stampa/", extract_italian_lastampa)
    it_sda94 = (c.PATH_BASE_DOCUMENTS + "italian/sda_italian/", extract_italian_sda9495)
    it_sda95 = (c.PATH_BASE_DOCUMENTS + "italian/sda_italian_95/", extract_italian_sda9495)
    italian = {"2001": [it_lastampa, it_sda94],
               "2002": [it_lastampa, it_sda94],
               "2003": [it_lastampa, it_sda94, it_sda95]}

    aamu9495 = c.PATH_BASE_DOCUMENTS + "finnish/aamu/"
    fi_ammulethi9495 = (aamu9495, extract_finish_aamuleth9495)
    finnish = {"2001": None, "2002": [fi_ammulethi9495], "2003": [fi_ammulethi9495]}

    processs, k = 4, 1000
    _all = {"finnish": finnish}
    docids, documents, qids, queries = [], [], [], []
    if not load:
        for year in c.YEARs[1:]:
            doc_dirs = _all["finnish"][year]
            current_path_queries = c.PATH_BASE_QUERIES + year + "/Top-en" + year[-2:] + ".txt"
            a1, b1, c1, d1, _ = prepare_experiment(doc_dirs, limit_documents, current_path_queries, limit_queries)
            docids.extend(a1)
            documents.extend(b1)
            qids.extend(c1)
            queries.extend(d1)

        print("===> Data Loaded")
        print(len(documents), len(queries))
        pool = Pool(processes=processs)
        clean_to_lower = partial(clean, to_lower=True)
        documents = pool.map(clean_to_lower, documents)
        pool.close()
        pool.join()
        queries = list(map(clean_to_lower, queries))
        
        tfidf = getTfidf(documents, processs)

        docDict = defaultdict(str)
        for i, doc in enumerate(documents):
            docDict[docids[i]] += ". " + doc
        
        subsetDocs = list(docDict.items())
        pool = Pool(processes=processs)
        subsetDocs = pool.starmap(sampleSent, zip(subsetDocs, repeat(tfidf), repeat(k)))
        pool.close()
        pool.join()
        subsetDocs = {did : doc for did, doc in subsetDocs if len(doc.strip())}
        allDocs = set(subsetDocs.keys())
        
        data = {"docids" : docids, "documents" : documents, "qids" : qids, "queries" : queries, "subsetDocs" : subsetDocs}
        p.dump(data, open("data.p", "wb"))
        p.dump(tfidf, open("tfidf.p", "wb"))
        print("===> Data Processed")
    else:
        data = p.load(open("data.p", "rb"))
        docids, documents, qids, queries, subsetDocs = data["docids"], data["documents"], data["qids"], data["queries"], data["subsetDocs"]

    query2docs = defaultdict(list)
    with open("../out", "r") as f:
        for line in f:
            line = line.split()
            if line[2] in allDocs:
                query2docs[int(line[0])].append(line[2])
    print("===> Saving Files")
    with open("train.tsv", "w") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for i, qid in enumerate(qids):
            if qid not in query2docs :
                print("No relevance judgements available for query %s"%(qid))
                continue
            #add relevant documents
            relDocs, nonRelDocs = getDocs(qid, query2docs, allDocs, relDocsCount, nonRelDocsCount)
            for did in relDocs:
                tsv_writer.writerow([str(1), qid, did, queries[i], subsetDocs[did]])
                
            #add nonrelevant documents
            for did in nonRelDocs:
                tsv_writer.writerow([str(0), qid, did, queries[i], subsetDocs[did]])
    
    with open("train_drmm.tsv", "w") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for i, qid in enumerate(qids):
            if qid not in query2docs :
                print("No relevance judgements available for query %s"%(qid))
                continue
            #add relevant documents
            relDocs, nonRelDocs = getDocs(qid, query2docs, allDocs, relDocsCount, nonRelDocsCount)
            for did in relDocs:
                tsv_writer.writerow([str(1), queries[i], subsetDocs[did]])

            #add nonrelevant documents
            for did in nonRelDocs:
                tsv_writer.writerow([str(0), queries[i], subsetDocs[did]])