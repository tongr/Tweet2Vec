import numpy as np
import sys
import codecs
import batch_char as batch

from utils.t2v.settings_char import N_BATCH, MAX_LENGTH, MAX_CLASSES
from utils.embedding import prepare_data, build_network


def main(args):
    data_path = args[0]
    model_path = args[1]
    save_path = args[2]

    # Test data
    Xt, _ = prepare_data(input_file=data_path, max_len=MAX_LENGTH)

    # Model
    predict, encode, chardict, n_char, inverse_labeldict = build_network(path=model_path,
                                                                         m_num=int(args[3]) if len(args) > 3 else None,
                                                                         max_classes=MAX_CLASSES)

    # Test
    print("Encoding and saving...")
    out_pred = []
    out_emb = []
    numbatches = len(Xt)/N_BATCH + 1
    for i in range(numbatches):
        xr = Xt[N_BATCH*i:N_BATCH*(i+1)]
        x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
        p = predict(x,x_m)
        e = encode(x,x_m)
        ranks = np.argsort(p)[:,::-1]

        for idx, item in enumerate(xr):
            out_pred.append(' '.join([inverse_labeldict[r] if r in inverse_labeldict else 'UNK' for r in ranks[idx,:5]]))
            out_emb.append(e[idx, :])

    # Save
    print("Saving...")
    with codecs.open('%s/predicted_tags.txt' % save_path, 'w', 'utf8') as f:
        for item in out_pred:
            f.write(item + '\n')
    with open('%s/embeddings.npy' % save_path, 'w') as f:
        np.save(f, np.asarray(out_emb))


if __name__ == '__main__':
    main(sys.argv[1:])
