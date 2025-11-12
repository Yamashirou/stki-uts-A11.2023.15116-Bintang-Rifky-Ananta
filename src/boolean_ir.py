
from typing import Dict, List
def build_inverted_index(docs: Dict[str,str]):
    inv = {}
    for doc_id, txt in docs.items():
        for t in set(txt.split()):
            inv.setdefault(t, []).append(doc_id)
    for t in inv: inv[t] = sorted(inv[t])
    return inv

def boolean_query(query: str, inv_index: Dict[str,List[str]], all_docs: List[str]):
    tokens = query.upper().split()
    result = set()
    op = None; i=0
    while i < len(tokens):
        tk = tokens[i]
        if tk in ('AND','OR'):
            op = tk; i+=1; continue
        if tk == 'NOT':
            i+=1; term = tokens[i].lower(); postings = set(all_docs) - set(inv_index.get(term,[]))
        else:
            term = tk.lower(); postings = set(inv_index.get(term,[]))
        if not result:
            result = postings
        else:
            if op=='AND': result = result & postings
            elif op=='OR': result = result | postings
            else: result = result & postings
        i+=1
    return sorted(result)
