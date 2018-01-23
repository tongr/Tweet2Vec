import lasagne
import theano
import theano.tensor as T
import cPickle as pkl
import io

from .t2v import tweet2vec, load_params


def invert(d):
    out = {}
    for k,v in d.iteritems():
        out[v] = k
    return out


def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer,
                                        n_classes,
                                        W=params['W_cl'],
                                        b=params['b_cl'],
                                        nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), lasagne.layers.get_output(emb_layer)


def prepare_data(input_file, max_len, preprocessor=lambda txt: txt):
    print("Preparing Data...")
    Xt = []
    with io.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            Xc = preprocessor(line.rstrip('\n'))
            Xt.append(Xc[:max_len])
    return Xt, None


def build_network(path, m_num=None, max_classes=6000):
    print("Loading model params...")
    if m_num is not None:
        params = load_params('%s/model_%d.npz' % (path,m_num))
    else:
        params = load_params('%s/best_model.npz' % path)

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, max_classes)
    inverse_labeldict = invert(labeldict)

    print("Building network...")
    # Tweet variables
    tweet = T.itensor3()
    t_mask = T.fmatrix()

    # network for prediction
    predictions, embeddings = classify(tweet, t_mask, params, n_classes, n_char)

    # Theano function
    print("Compiling theano functions...")
    predict = theano.function([tweet, t_mask], predictions)
    encode = theano.function([tweet, t_mask], embeddings)
    return predict, encode, chardict, n_char, inverse_labeldict

