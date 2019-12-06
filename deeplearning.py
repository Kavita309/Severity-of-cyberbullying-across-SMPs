import pickle
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
import tflearn
from models import get_model
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import warnings


HASH_REMOVE= None
MAX_FEATURES = 2
NUM_CLASSES = 4
LEARN_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 128
TWITTER_DATA_FILE = "C:/Users/kavita/Desktop/BTP Project/DataSets/PKL/TwitterData.pkl"
FORMSPRING_DATA_FILE = "C:/Users/kavita/Desktop/BTP Project/DataSets/PKL/FormspringData.pkl"
REDDIT_DATA_FILE = "C:/Users/kavita/Desktop/BTP Project/DataSets/PKL/RedditData.pkl"


def run_model(data, oversampling_rate, model_type, vector_type, embed_size):
    if data == "twitter":
        x_text, labels = get_data(data, oversampling_rate, TWITTER_DATA_FILE)
    elif data == "formspring":
        x_text, labels = get_data(data, oversampling_rate, FORMSPRING_DATA_FILE)
    else:
        x_text, labels = get_data(data, oversampling_rate, REDDIT_DATA_FILE)
    data_dict = get_train_test(data,  x_text, labels)
    precision, recall, f1_score = train(data_dict, model_type, vector_type, embed_size)


def get_data(data, oversampling_rate, file_path):
    x_text, labels = load_data(file_path)
    NUM_CLASSES = 4
    dict1 = {'L':0,'M':1,'H':2,'none':3}
    labels = [dict1[b] for b in labels]

    H = [i for i in range(len(labels)) if labels[i]==2]
    M = [i for i in range(len(labels)) if labels[i]==1]
    L = [i for i in range(len(labels)) if labels[i]==0]

    x_text = x_text + [x_text[x] for x in H]*(oversampling_rate-1)+ [x_text[x] for x in M]*(oversampling_rate-1)+ [x_text[x] for x in L]*(oversampling_rate-1)
    labels = labels + [2 for i in range(len(H))]*(oversampling_rate-1) + [1 for i in range(len(M))]*(oversampling_rate-1)+ [0 for i in range(len(L))]*(oversampling_rate-1)
    print("Counter after oversampling")
    from collections import Counter
    print(Counter(labels))
    return x_text, labels


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


def get_train_test(data, x_text, labels):
    X_train, X_test, Y_train, Y_test = train_test_split( x_text, labels, random_state=42, test_size=0.10)

    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)

    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))

    trainY = np.asarray(Y_train)
    testY = np.asarray(Y_test)

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

    trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    testY = to_categorical(testY, nb_classes=NUM_CLASSES)

    data_dict = {
        "data": data,
        "trainX" : trainX,
        "trainY" : trainY,
        "testX" : testX,
        "testY" : testY,
        "vocab_processor" : vocab_processor
    }
    return data_dict


def get_embedding_weights(filename, sep):
    embed_dict = {}
    file = open(filename,'r',encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(sep)
        embed_dict[row[0]] = row[1:]
    print('Loaded from file: ' + str(filename))
    file.close()
    return embed_dict


def map_embedding_weights(embed, vocab, embed_size):
    vocab_size = len(vocab)
    embeddingWeights = np.zeros((vocab_size , embed_size))
    n = 0
    words_missed = []
    for k, v in vocab.items():
        try:
            embeddingWeights[v] = embed[k]
        except:
            n += 1
            words_missed.append(k)
            pass
    print("%d embedding missed"%n, " of " , vocab_size)
    return embeddingWeights


def get_embeddings_dict(vector_type, emb_dim):
    if vector_type == 'sswe':
        emb_dim==50
        sep = '\t'
        vector_file = 'DataSets/WordEmbeddings/sswe-u.txt'
    elif vector_type =="glove":
        sep = ' '
        vector_file = 'DataSets/WordEmbeddings/glove.twitter.27B.' + str(emb_dim) + 'd.txt'
    else:
        print ("ERROR: Please specify a correct model or SSWE cannot be loaded with embed size of: " + str(emb_dim))
        return None

    embed = get_embedding_weights(vector_file, sep)
    return embed


def train(data_dict, model_type, vector_type, embed_size, dump_embeddings=False):

    data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    vocab = vocab_processor.vocabulary_._mapping

    print("Running Model: " + model_type + " with word vector initiliazed with " + vector_type + " word vectors.")

    if(model_type!="cnn"):
        model = get_model(model_type, trainX.shape[1], vocab_size, embed_size, 4, LEARN_RATE)
        initial_weights = model.get_weights()
        shuffle_weights(model, initial_weights)
    else:
        model,network = get_model(model_type, trainX.shape[1], vocab_size, embed_size, 4, LEARN_RATE)
        initial_weights = model.get_weights(network.W)
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in initial_weights]
        weights = np.asarray(weights).reshape(initial_weights.shape)
        model.set_weights(network.W, weights)

    if(model_type == 'cnn'):
        if(vector_type!="random"):
            print("Word vectors used: " + vector_type)
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
            model.set_weights(embeddingWeights, map_embedding_weights(get_embeddings_dict(vector_type, embed_size), vocab, embed_size))
            model.fit(trainX, trainY, n_epoch = EPOCHS, shuffle=True, show_metric=True, batch_size=BATCH_SIZE)
        else:
            model.fit(trainX, trainY, n_epoch = EPOCHS, shuffle=True, show_metric=True, batch_size=BATCH_SIZE)
    else:
        if(vector_type!="random"):
            print("Word vectors used: " + vector_type)
            model.layers[0].set_weights([map_embedding_weights(get_embeddings_dict(vector_type, embed_size), vocab, embed_size)])
            model.fit(trainX, trainY, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                  verbose=1)
        else:
            model.fit(trainX, trainY, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                  verbose=1)
    return evaluate_model(model, testX, testY)

def return_data(data_dict):
    return data_dict["data"], data_dict["trainX"], data_dict["trainY"], data_dict["testX"], data_dict["testY"], data_dict["vocab_processor"]

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

def evaluate_model(model, testX, testY):
    temp = model.predict(testX)
    y_pred  = np.argmax(temp, 1)
    y_true = np.argmax(testY, 1)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    print("Precision: " + str(precision) + "\n")
    print("Recall: " + str(recall) + "\n")
    print("f1_score: " + str(f1_score) + "\n")
    print(confusion_matrix(y_true, y_pred))
    print(":: Classification Report")
    print(classification_report(y_true, y_pred))
    return precision, recall, f1_score


def print_scores(precision_scores, recall_scores, f1_scores):
    for i in range(NUM_CLASSES):
        print("\nPrecision Class %d (avg): %0.3f (+/- %0.3f)" % (i, precision_scores[:, i].mean(), precision_scores[:, i].std() * 2))
        print( "\nRecall Class %d (avg): %0.3f (+/- %0.3f)" % (i, recall_scores[:, i].mean(), recall_scores[:, i].std() * 2))
        print( "\nF1 score Class %d (avg): %0.3f (+/- %0.3f)" % (i, f1_scores[:, i].mean(), f1_scores[:, i].std() * 2))


MODEL_TYPES = [ 'cnn', 'lstm', 'blstm', 'blstm_attention']
WORD_VECTOR_TYPES = ["random", "glove" ,"sswe"]
OVERSAMPLING_RATE = 3
DATA = ['twitter', 'formspring', 'reddit']
EMBED_SIZE = [25, 50, 100, 200]

warnings.filterwarnings("ignore")
run_model(DATA[2], OVERSAMPLING_RATE, MODEL_TYPES[0], WORD_VECTOR_TYPES[2], EMBED_SIZE[1])
