import math

def precision(retrieved,relevant):
    if not retrieved: return 0.0
    return len(set(retrieved)&set(relevant))/len(retrieved)

def recall(retrieved,relevant):
    if not relevant: return 0.0
    return len(set(retrieved)&set(relevant))/len(relevant)

def f1_score(retrieved,relevant):
    p=precision(retrieved,relevant); r=recall(retrieved,relevant)
    if p+r==0: return 0.0
    return 2*p*r/(p+r)

def precision_at_k(retrieved,relevant,k):
    return precision(retrieved[:k], relevant)

def average_precision_at_k(retrieved,relevant,k):
    num_rel=0; score=0.0
    for i in range(min(k,len(retrieved))):
        if retrieved[i] in relevant:
            num_rel+=1; score += num_rel/(i+1)
    if num_rel==0: return 0.0
    return score/len(relevant)

def ndcg_at_k(retrieved,relevant,k):
    def dcg_at_k(retr,k):
        dcg=0.0
        for i in range(min(k,len(retr))):
            rel = 1.0 if retr[i] in relevant else 0.0
            dcg += rel / math.log2(i+2)
        return dcg
    def idcg_at_k(rel,k):
        rels = min(len(rel),k)
        return sum(1.0/math.log2(i+2) for i in range(rels))
    idcg = idcg_at_k(relevant,k)
    if idcg==0: return 0.0
    return dcg_at_k(retrieved,k)/idcg
