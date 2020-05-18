import numpy as np

def relevance_feedback(vec_docs, vec_queries, sim,gt, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    a=0.5
    b=1
    n_doc=sim.shape[0]
    n_que=sim.shape[1]
    dR=[]
    ndR=[]
    doc=[]    
    for i in range(n_doc):
        r_d=np.argsort(-sim[:,1])
        doc.append(r_d[:n])
        
    for i in range(len(doc)):
        quer=i
        p=doc[quer]
        rel=[]
        nrel=[]
        for j in range (len(gt)):
            if (gt[j][0]==quer+1):
                nrel.append(gt[j][1])
            else:
                rel.append(gt[j][1])
        dR.append(rel)
        ndR.append(nrel)
    
    new_quer=np.zeros([30,11420])
    for i in range(30):
        quer=vec_queries.toarray()[i]
        R=dR[i]
        nR=ndR[i]
        w1=np.zeros([11420,])
        w2=np.zeros([11420,])
        for j in R:
            w1=w1+vec_docs.toarray()[j-1]
        for j in nR:
            w2=w2+vec_docs.toarray()[j-1]
            
        w=a*w1+b*w2
        
        
    rf_sim = w # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model,gt, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    a=0.5
    b=1
    n_doc=sim.shape[0]
    n_que=sim.shape[1]
    dR=[]
    ndR=[]
    doc=[]    
    for i in range(n_doc):
        r_d=np.argsort(-sim[:,1])
        doc.append(r_d[:n])
        
    for i in range(len(doc)):
        quer=i
        p=doc[quer]
        rel=[]
        nrel=[]
        for j in range (len(gt)):
            if (gt[j][0]==quer+1):
                nrel.append(gt[j][1])
            else:
                rel.append(gt[j][1])
        dR.append(rel)
        ndR.append(nrel)
    
    new_quer=np.zeros([30,11420])
    for i in range(30):
        quer=vec_queries.toarray()[i]
        R=dR[i]
        nR=ndR[i]
        w1=np.zeros([11420,])
        w2=np.zeros([11420,])
        for j in R:
            w1=w1+vec_docs.toarray()[j-1]
        for j in nR:
            w2=w2+vec_docs.toarray()[j-1]
            
        w=a*w1+b*w2
        
        
    rf_sim = w # change
    return rf_sim
