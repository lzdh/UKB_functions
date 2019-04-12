# match Y to X by maximizing column wise corr or euclidean distance.
# Used to align rotated principal loadings in CV

import numpy as np
from scipy.stats import pearsonr

def matchit(X,Y,method):

    temp=Y
    result=[]
    
    if method == 'pearson':
        for k in range(X.shape[1]):
            corr = [abs(pearsonr(X[:,k],temp[:,j])[0]) for j in range(temp.shape[1])]   
            result.append(temp[:,np.argmax(corr)])
            temp = np.delete(temp,np.argmax(corr),1)
            
    if method == 'euclidean':
        for k in range(X.shape[1]):
            dist = [np.linalg.norm(X[:,k]-temp[:,j]) for j in range(temp.shape[1])]   
            result.append(temp[:,np.argmin(dist)])
            temp = np.delete(temp,np.argmin(dist),1)
            
    return np.array(result).T
    