import csv
from itertools import repeat
from functools import partial
from collections import defaultdict
from multiprocessing.pool import Pool
from collection_extractors import extract_italian_lastampa, extract_italian_sda9495
import constants as c
from text2vec import clean
import pickle as p
from createData import *

if __name__ == '__main__':
    

    # Prepare italian CLEF data
    limit_documents = None
    limit_queries = None
    target_language = "IT"
    it_lastampa = (c.PATH_BASE_DOCUMENTS + "italian/la_stampa/", extract_italian_lastampa)
    it_sda94 = (c.PATH_BASE_DOCUMENTS + "italian/sda_italian/", extract_italian_sda9495)
    it_sda95 = (c.PATH_BASE_DOCUMENTS + "italian/sda_italian_95/", extract_italian_sda9495)
    italian = {"2001": [it_lastampa, it_sda94],
               "2002": [it_lastampa, it_sda94],
               "2003": [it_lastampa, it_sda94, it_sda95]}

    processs, k = 4, 1000
    _all = {"italian": italian}
    tfidf = p.load(open("tfidf.p", "rb"))

    for year in c.YEARs:
        doc_dirs = _all["italian"][year]
        current_path_queries = c.PATH_BASE_QUERIES + year + "/Top-en" + year[-2:] + ".txt"
        current_assessment_file = c.PATH_BASE_EVAL + year + "/qrels_" + target_language
        docids, documents, qids, queries, relass = prepare_experiment(doc_dirs, limit_documents, current_path_queries, \
                                                    limit_queries, relevance_assessment_file=current_assessment_file)
        print("===> Data Loaded")
        pool = Pool(processes=processs)
        clean_to_lower = partial(clean, to_lower=True)
        documents = pool.map(clean_to_lower, documents)
        pool.close()
        pool.join()
        queries = list(map(clean_to_lower, queries))

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
        print("===> Data Processed")
        with open("test_%s.tsv"%(year), "w") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for i, qid in enumerate(qids):
                for did in allDocs:
                    label = "1" if qid in relass and did in relass[qid] else "0"
                    tsv_writer.writerow([label, qid, did, queries[i], subsetDocs[did]])
        
        with open("test_drmm_%s.tsv"%(year), "w") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for i, qid in enumerate(qids):
                for did in allDocs:
                    label = "1" if qid in relass and did in relass[qid] else "0"
                    tsv_writer.writerow([label, queries[i], subsetDocs[did]])
        print(year + " Finished")