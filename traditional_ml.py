import pickle, numpy as np
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, make_scorer, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
import warnings


HASH_REMOVE = None
NO_OF_FOLDS = 10
N_CLASS = 4


def load_data(filename):
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = []
    for i in range(len(data)):
        if(HASH_REMOVE):
            x_text.append(p.tokenize((data[i][0]).encode('utf-8')))
        else:
            x_text.append(data[i][0])
        labels.append(data[i][1])
    return x_text,labels


def get_model(m_type):
    if m_type == 'lr':
        logreg = LogisticRegression(class_weight="balanced")
    elif m_type == 'naive':
        logreg =  MultinomialNB()
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif m_type == "svm":
        logreg = LinearSVC(class_weight="balanced")
    else:
        print ("ERROR: Please specify a correct model")
        return None
    return logreg


def train(x_text, labels, model_type, embedding, oversampling):

    dict1 = {'L':0,'M':1,'H':2,'none':3}
    labels = [dict1[b] for b in labels]

    from collections import Counter
    print(Counter(labels))

    if oversampling:
        H = [i for i in range(len(labels)) if labels[i]==2]
        M = [i for i in range(len(labels)) if labels[i]==1]
        L = [i for i in range(len(labels)) if labels[i]==0]
        x_text = x_text + [x_text[x] for x in H]*(2)+ [x_text[x] for x in M]*(2)+ [x_text[x] for x in L]*(2)
        labels = labels + [2 for i in range(len(H))]*(2) + [1 for i in range(len(M))]*(2)+ [0 for i in range(len(L))]*(2)
        print("Counter after oversampling")
        from collections import Counter
        print(Counter(labels))

    if(embedding == "word"):
        print("Using word based features")
        bow_transformer = CountVectorizer(analyzer="word",max_features = 10000,stop_words='english').fit(x_text)
        comments_bow = bow_transformer.transform(x_text)
        tfidf_transformer = TfidfTransformer(norm = 'l2').fit(comments_bow)
        comments_tfidf = tfidf_transformer.transform(comments_bow)
        features = comments_tfidf
    else:
        print("Using char n-grams based features")
        bow_transformer = CountVectorizer(max_features = 10000, ngram_range = (1,2)).fit(x_text)
        comments_bow = bow_transformer.transform(x_text)
        tfidf_transformer = TfidfTransformer(norm = 'l2').fit(comments_bow)
        comments_tfidf = tfidf_transformer.transform(comments_bow)
        features = comments_tfidf

    classification_model(features, labels, model_type)


def classification_model(X, Y, model_type):
    X, Y = shuffle(X, Y, random_state=42)
    print ("Model Type:", model_type)
    kf = KFold(n_splits=NO_OF_FOLDS)
    scores = []
    for train_index, test_index in kf.split(X):
        Y = np.asarray(Y)
        model = get_model(model_type)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        curr_scores = get_scores(y_test, y_pred)
        scores.append(np.hstack(curr_scores))
    print_scores(np.array(scores))


def get_scores(y_true, y_pred):
    # print(":: Confusion Matrix")
    # print(confusion_matrix(y_true, y_pred))
    # print(":: Classification Report")
    # print(classification_report(y_true, y_pred))
    return np.array([
            precision_score(y_true, y_pred, average=None),
            recall_score(y_true, y_pred,  average=None),
            f1_score(y_true, y_pred, average=None)])


def print_scores(scores):
    for i in range(N_CLASS):
            print ("Precision Class %d (avg): %0.3f (+/- %0.3f)" % (i,scores[:, i].mean(), scores[:, i].std() * 2))
            print ("Recall Class %d (avg): %0.3f (+/- %0.3f)" % (i,scores[:,  N_CLASS+i].mean(), scores[:,N_CLASS+i].std() * 2))
            print ("F1_score Class %d (avg): %0.3f (+/- %0.3f)" % (i,scores[:, N_CLASS*2+i].mean(), scores[:,  N_CLASS*2+i].std() * 2))


MODEL_TYPES = ['random_forest','naive','lr','svm']
EMBEDDING = ['word', 'char']
twitter_data_file = "C:/Users/kavita/Desktop/BTP Project/DataSets/PKL/TwitterData.pkl"
formspring_data_file = "C:/Users/kavita/Desktop/BTP Project/DataSets/PKL/FormspringData.pkl"
reddit_data_file = "C:/Users/kavita/Desktop/BTP Project/DataSets/PKL/RedditData.pkl"

warnings.filterwarnings("ignore")
x_text,labels = load_data(reddit_data_file)
train(x_text, labels, MODEL_TYPES[3],EMBEDDING[1],False)
