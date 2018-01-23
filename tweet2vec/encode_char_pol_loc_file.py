import sys
import codecs
import batch_char as batch

from utils.pol_loc_tweets import RetweetDetector
from utils.preprocessing import preprocess
from utils.embedding import prepare_data_pol_loc_file, build_network


def main(args):
    # ignore tweets with very similar texts
    retweet_cache = RetweetDetector(cache_size_limit=1000000)

    data_path = args[0]
    model_path = args[1]
    save_path = args[2]

    # Test data
    Xt, meta = prepare_data_pol_loc_file(input_file=data_path,
                                         max_len=MAX_LENGTH,
                                         retweet_cache=retweet_cache,
                                         preprocessor=lambda data: preprocess(data['text']))

    # Model
    predict, encode, chardict, n_char, inverse_labeldict = build_network(path=model_path,
                                                                         m_num=int(args[3]) if len(args) > 3 else None,
                                                                         max_classes=MAX_CLASSES)

    # Test
    print("Encoding and saving...")

    numbatches = len(Xt)/N_BATCH + 1
    seen = 0
    with codecs.open('%s/embeddings.pol-loc.tsv' % save_path, 'w', 'utf-8') as output:
        for i in range(numbatches):
            xr = Xt[N_BATCH*i:N_BATCH*(i+1)]
            x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
            e = encode(x,x_m)

            for idx, item in enumerate(xr):
                output.write("\t".join(
                    list(meta[N_BATCH*i+idx][att] for att in ['id', 'datetime', 'persons', 'locations']) +
                    list(str(x) for x in e[idx,:])))
                output.write('\n')
            output.flush()
            seen += len(xr)
            print("progress: {}".format(seen))


if __name__ == '__main__':
    main(sys.argv[1:])
