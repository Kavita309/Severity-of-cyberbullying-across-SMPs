import pickle
import base64
import csv

pkl_path = "C:/Users/kavita/Downloads/MarkSheets and Resume/data/data/wiki_data.pkl"
csv_path = "C:/Users/kavita/Desktop/BTP Project/wiki_0.csv"

your_pickle_obj = pickle.loads(open(pkl_path, 'rb').read())
print(len(your_pickle_obj[0]['text']))
print(type(your_pickle_obj[0]))
print(your_pickle_obj[0]['text'])
print(your_pickle_obj[0]['label'])

with open(csv_path,'w', newline='') as csv_file:
    wr = csv.writer(csv_file, delimiter=',')
    for i in range(3000):
        if(your_pickle_obj[i]['label'] == 0):
            print(i)
            if(i!=153 and i!=196 and i!=342 and i!=369 and i!=395 and i!=434 and i!=619 and i!=736 and i!=757 and i!=930 and i!=953 and i!=1049 and i!=1376 and i!=1372 and i!=1486 and i!=1518 and i!=1723 and i!=1752 and i!=1756 and i!=1765 and i!=1889 and i!=1933 and i!=1952 and i!=7926 and i!=2072 and i!=2182 and i!=2219 and i!=2608 and i!=2623 and i!=2636 and i!=2661):
                wr.writerow([your_pickle_obj[i]['text'],your_pickle_obj[i]['label']])
