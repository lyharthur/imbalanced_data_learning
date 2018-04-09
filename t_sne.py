import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets, manifold

# Scale and visualize the embedding vectors                            
def plot_embedding(X, y, title=None):     
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)                           

    plt.figure()              
    ax = plt.subplot(111)      
    color_map = ['blue', 'red']
    # for i in range(X.shape[0]): 
        # if y[i,0]==0:
        #     color = 'blue'
        # else:
        #     color = 'red'                         
    plt.scatter(X[:, 0], X[:, 1], c=y[:,0], cmap='RdBu')
    plt.legend(loc='upper left')

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig('tsne.png')
    plt.close()
