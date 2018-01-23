import sys
import io

from utils.preprocessing import preprocess

# input and output files
infile = sys.argv[1]
outfile = sys.argv[2]

with io.open(outfile, 'w') as tweet_processed_text, io.open(infile, 'r') as fin:
    for line in fin:
        tweet_processed_text.write(preprocess(line.rstrip())+'\n')
