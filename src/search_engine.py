
import argparse, os
from src import preprocess, boolean_ir, vsm_ir
def load_docs(folder):
    docs={}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            with open(os.path.join(folder,fname),'r',encoding='utf-8') as f:
                docs[fname]=f.read()
    return docs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', choices=['boolean','vsm','bm25'], default='vsm')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--query', type=str, required=True)
    args=parser.parse_args()
    raw='data/raw'; processed='data/processed'
    if not os.path.exists(processed) or not os.listdir(processed):
        print('Processing...'); preprocess.process_folder(raw, processed)
    docs = load_docs(processed)
    if args.model=='boolean':
        inv = boolean_ir.build_inverted_index(docs)
        res = boolean_ir.boolean_query(args.query, inv, sorted(docs.keys()))
        for r in res: print(r)
    elif args.model=='vsm':
        res = vsm_ir.rank_query_tfidf(args.query, docs, k=args.k)
        for doc,score in res: print(doc, score)
    else:
        idx = vsm_ir.build_bm25(docs)
        res = vsm_ir.score_bm25(args.query, idx, topk=args.k)
        for doc,score in res: print(doc, score)

if __name__=='__main__': main()
