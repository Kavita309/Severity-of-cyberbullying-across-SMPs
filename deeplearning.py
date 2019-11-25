import pickle
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
import tflearn
from models import get_model


data = "twitter"
model_type = "cnn"
vector_type = "random"
HASH_REMOVE= None
MAX_FEATURES = 2
NUM_CLASSES = None
LEARN_RATE = 0.01
x = "C:/Users/kavita/Desktop/BTP_Downloads/twitterp.pkl"

def run_model(data, oversampling_rate, model_type, vector_type, embed_size):
    x_text, labels = get_data(data, oversampling_rate)
    data_dict = get_train_test(data,  x_text, labels)
    train(data_dict, model_type, vector_type, embed_size)


def load_data(filename):
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = []
    for i in range(len(data)):
        if(HASH_REMOVE):
            x_text.append(p.tokenize((data[i][1]).encode('utf-8')))
        else:
            x_text.append(data[i][1])
        labels.append(data[i][2])
    return x_text,labels


def get_data(data, oversampling_rate):
    x_text, labels = load_data(x)

    if(data=="twitter"):
        NUM_CLASSES = 3
        dict1 = {'L':0,'M':1,'H':2,'none':3}
        labels = [dict1[b] for b in labels]

        H = [i for i in range(len(labels)) if labels[i]==2]
        M = [i for i in range(len(labels)) if labels[i]==1]
        L = [i for i in range(len(labels)) if labels[i]==0]

        x_text = x_text + [x_text[x] for x in H]*(oversampling_rate-1)+ [x_text[x] for x in M]*(oversampling_rate-1)+ [x_text[x] for x in L]*(oversampling_rate-1)
        labels = labels + [2 for i in range(len(H))]*(oversampling_rate-1) + [1 for i in range(len(M))]*(oversampling_rate-1)+ [0 for i in range(len(L))]*(oversampling_rate-1)
    else:
        NUM_CLASSES = 2
        bully = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in bully]*(oversampling_rate-1)
        labels = list(labels) + [1 for i in range(len(bully))]*(oversampling_rate-1)

    print("Counter after oversampling")
    from collections import Counter
    print(Counter(labels))

    # filter_data = []
    # for text in x_text:
    #     filter_data.append("".join(l for l in text if l not in string.punctuation))

    return x_text, labels

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

def train(data_dict, model_type, vector_type, embed_size, dump_embeddings=False):

    data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)
    # trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    # testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    vocab = vocab_processor.vocabulary_._mapping

    print("Running Model: " + model_type + " with word vector initiliazed with " + vector_type + " word vectors.")
    model = get_model(model_type, trainX.shape[1], vocab_size, embed_size, NUM_CLASSES, LEARN_RATE)

    initial_weights = model.get_weights()
    shuffle_weights(model, initial_weights)
    print(initial_weights)

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

run_model(data, 3, model_type, vector_type, 50)
