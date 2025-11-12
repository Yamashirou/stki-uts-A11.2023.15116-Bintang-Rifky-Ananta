
import math
from collections import Counter

def compute_df(docs):
    df={}
    for txt in docs.values():
        for t in set(txt.split()):
            df[t]=df.get(t,0)+1
    return df

def compute_idf(df,N,smooth=True):
    idf={}
    for t,v in df.items():
        if smooth:
            idf[t]=math.log((N+1)/(v+1))+1
        else:
            idf[t]=math.log(N/v) if v!=0 else 0.0
    return idf

def build_tfidf_vectors(docs):
    N = len(docs)
    vocab = sorted({t for txt in docs.values() for t in txt.split()})
    df = compute_df(docs)
    idf = compute_idf(df,N)
    doc_vectors = {}
    for doc_id, txt in docs.items():
        tokens = txt.split()
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        vec = [ (tf.get(t,0)/total) * idf.get(t,0) for t in vocab ]
        doc_vectors[doc_id] = vec
    return vocab, idf, doc_vectors

def cosine_sim(a,b):
    num = sum(x*y for x,y in zip(a,b))
    den_a = math.sqrt(sum(x*x for x in a))
    den_b = math.sqrt(sum(y*y for y in b))
    if den_a==0 or den_b==0: return 0.0
    return num/(den_a*den_b)

def rank_query_tfidf(query, docs, k=5):
    vocab, idf, doc_vectors = build_tfidf_vectors(docs)
    from collections import Counter
    qc = Counter(query.split())
    qtotal = sum(qc.values()) or 1
    qvec = [ (qc.get(t,0)/qtotal) * idf.get(t,0) for t in vocab ]
    scores = [(doc_id, cosine_sim(qvec, vec)) for doc_id, vec in doc_vectors.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def build_bm25(docs,k1=1.5,b=0.75):
    N = len(docs)
    df = compute_df(docs)
    avgdl = sum(len(d.split()) for d in docs.values())/N
    idf = {}
    for t,v in df.items():
        idf[t]=math.log(1 + (N - v + 0.5) / (v + 0.5))
    index = {}
    for doc_id, txt in docs.items():
        tf = Counter(txt.split())
        index[doc_id] = {'tf': tf, 'len': len(txt.split())}
    return {'N':N,'df':df,'avgdl':avgdl,'idf':idf,'index':index,'k1':k1,'b':b}

def score_bm25(query, bm25_index, topk=5):
    q_tokens = query.split()
    scores=[]
    N = bm25_index['N']; avgdl = bm25_index['avgdl']; k1=bm25_index['k1']; b=bm25_index['b']; idf=bm25_index['idf']
    for doc_id, meta in bm25_index['index'].items():
        score=0.0
        for q in q_tokens:
            if q not in meta['tf']: continue
            tf = meta['tf'][q]
            denom = tf + k1 * (1 - b + b * (meta['len']/avgdl))
            score += idf.get(q,0) * ((tf * (k1+1)) / denom)
        scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]
