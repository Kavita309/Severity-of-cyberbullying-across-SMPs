import csv
from six.moves import cPickle as pickle
import numpy as np

def csv_to_pkl(path_csv, path_pickle):

    x = []
    with open(path_csv,'r',encoding="utf8") as f:
        read = csv.reader(f)

        for line in read:
            x.append(line)

    with open(path_pickle,'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

csv_to_pkl("C:/Users/kavita/Desktop/BTP Project/DataSets/CSV/TwitterData.csv","C:/Users/kavita/Desktop/BTP Project/DataSets/PKL/TwitterData.pkl")
