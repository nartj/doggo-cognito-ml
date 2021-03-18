import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mxnet import nd
from tqdm import tqdm


def visualize():
    data_dir = "data"

    synset = np.unique(pd.read_csv(os.path.join(data_dir, 'labels.csv')).breed).tolist()
    n = len(glob(os.path.join('.', data_dir, 'Images', '*', '*.jpg')))

    yy = pd.value_counts(synset)
    yy[:] = 0
    print('Aggregating labels')
    for i, file_name in tqdm(enumerate(glob(os.path.join('.', data_dir, 'Images', '*', '*.jpg'))), total=n):
        yy[synset.index(file_name.split('/')[3][10:].lower())] += 1
        nd.waitall()

    print("Number of images: %d" % n)
    print("Number of classes: %d" % len(synset))

    yy = yy.sort_values()

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 9)
    sns.set_style("whitegrid")

    ax = sns.barplot(x=yy.index, y=yy, data=pd.DataFrame(synset))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set(xlabel='Dog Breed', ylabel='Count')
    ax.set_title('Distribution of Dog breeds')
    fig.show()

    input("Press Enter to continue...")
