
import csv
from six.moves import cPickle as pickle
import numpy as np

def main(path_csv, path_pickle):

    x = []
    with open(path_csv,'r',encoding="utf8") as f:
        read = csv.reader(f)

        for line in read:
            #print (line)
            x.append(line)

    with open(path_pickle,'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

main("C:/Users/kavita/Desktop/BTP_Downloads/sampleTwitter.csv","C:/Users/kavita/Desktop/BTP_Downloads/twitterp.pkl")
