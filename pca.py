import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='Entrenamiento.csv',
    header=None,
    sep=',')

df.columns=['s1', 's2', 's3', 's4', 's5', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


# split data table into data X and class labels y

# split data table into data X and class labels y

X = df.ix[:, 0:5].values
y = df.ix[:, 5].values.astype(str)
print(X)
print(y)


# plotting histograms

from matplotlib import pyplot as plt
import numpy as np
import math

label_dict = {1: '0',
              2: '1',
              3: '2'}

feature_dict = {0: 's1',
                1: 's2',
                2: 's3',
                3: 's4',
                4: 's5'}

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 10))
    for cnt in range(5):
        plt.subplot(3, 3, cnt+1)
        for lab in ('0','1','2','3','4','5','6','7','8','9','10','11'):
            plt.hist(X[y == lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
        plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

    plt.tight_layout()
    plt.show()

X_std = StandardScaler().fit_transform(X)
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('0', '1', '2', '3','4','5','6','7','8','9','10','11','12'),
                        ('blue', 'green', 'red','cyan','magenta','yellow','black','white','purple','gray','orange','brown')):
        plt.scatter(Y_sklearn[y==lab, 0],
                    Y_sklearn[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()