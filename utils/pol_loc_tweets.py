# coding=utf-8
import codecs
import json
import re
from collections import OrderedDict


#
# politician-location-file
#

def csv_to_dict(line, fields, sep='\t'):
    # ['id', 'datetime',
    #  'persons', 'locations',
    #  'persons_token', 'locations_token',
    #  'text', 'token',
    #  'dump_file']
    data = {}
    for idx, value in enumerate(line.split(sep)):
        if idx >= len(fields):
            break
        data[fields[idx]]=value
    return data


schema = ['id', 'datetime',
          'persons', 'locations',
          'persons_token', 'locations_token',
          'text', 'token',
          'dump_file']


def remove_mentions(token, mentions):
    """
    remove the mention offest in a token list
    :param token: list of tokes (of a tweet)
    :param mentions: expected list in the form of:
        ["start_token_index-end_token_index", "start_token_index-end_token_index", ...]
    :return: pruned token list
    """
    mention_token = set()
    for mention in mentions:
        # format of a match is "[start]-[end]"
        start, end = mention.split('-')
        for x in range(int(start), int(end)):
            mention_token.add(x)

    # remove the token in reverse order (from back to front)
    mention_token = sorted(list(mention_token), reverse=True)
    new_token = list(token)
    for del_index in mention_token:
        del new_token[del_index]

    return new_token


single_special_char = re.compile("^\W$", re.IGNORECASE)


def parse_politician_location_tweet_file(filename, unquote_data=False):
    data = map(lambda line: csv_to_dict(line, schema), codecs.open(filename, 'r', encoding='utf-8'))
    if unquote_data:
        quoted_attributes = {'persons', 'locations', 'persons_token', 'locations_token', 'text', 'token',
                             'person_probs', 'location_probs'}
        def unquote(data_item):
            for attribute in quoted_attributes & set(data_item.keys()):
                data_item[attribute] = json.loads(data_item[attribute])
            return data_item
        data = map(unquote, data)
    return data


__remove_rt = re.compile("^\s*(RT|VIA)?\s*(@\w+|\W+?)+", re.IGNORECASE)
__remove_link = re.compile("\s*http\\S+\s*", re.IGNORECASE)
__remove_trailing_dots = re.compile("[….]+\s*$")


def trim_tweet_text(text, trim_links=True, trim_rt=True):
    """
    trims the tweet text (i.e., removes links and
    :param text: the tweet text to be trimmed
    :param trim_links: if True, all links/urls are removed
    :param trim_rt: if True, all leading "RT @user: ..." are removed
    :return: the trimmed tweet text
    """
    if trim_rt:
        text = __remove_rt.sub("", text)
    if trim_links:
        text=__remove_link.sub(" ", text)

    text=__remove_trailing_dots.sub("", text)
    text = text.strip()

    return text


class LimitedSizeDict(OrderedDict):
    """
    this class enables a basic LRU (least recently used) cache functionality
    initialized as follows:
    cache = LimitedSizeDict(size_limit=2)
    it will store only the last 10 recently added cache entries (x->y)
    cache['x2']='y2' # --> cache = {'x2':'y2'}
    cache['x1']='y1' # --> cache = {'x1':'y1','x2':'y2'}
    cache['x3']='y3' # --> cache = {'x1':'y1','x3':'y3'}
    cache['x2']='y2' # --> cache = {'x2':'y2','x3':'y3'}
    """
    def __init__(self, *args, **kwds):
        """
        initialize LimitedSizeDict by specifying the maximal cache size as size_limit:
        e.g.: ...=LimitedSizeDict(size_limit=2)
        :param args:
        :param kwds:
        """
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value, **kwargs):
        OrderedDict.__setitem__(self,  key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class RetweetDetector(object):
    def __init__(self, cache_size_limit):
        self._cache = LimitedSizeDict(size_limit=cache_size_limit)

    def check_collision(self, tweet, append_to_cache=False):
        """
        checks if the tweet text is already contained in the cache
        :param tweet: the tweet data (text or tweet dict)
        :param append_to_cache: if True, the cache is updated with the text (as latest entry)
        :return: True if the tweet text was seen before
        """
        text = None
        if isinstance(tweet, str):
            text = tweet
        elif isinstance(tweet, dict):
            text = tweet['text']

        if not text:
            raise ValueError

        # remove links and RT indicators from text
        text = trim_tweet_text(text)

        # reduce length of the text (in case of a retweet of tweet 'xxx' as 'RT @realDonaldTrump: xxx' the message is
        # shrunk by twitter to a max length of 140 characters .. we try to reduce this problem by considering only the
        # first part of the tweet message
        text = text[:80]
        text = text.lower()
        # text_hash = sha1(text.encode('utf-8')).hexdigest()

        is_cache_hit = text in self._cache

        if append_to_cache:
            if is_cache_hit:
                # update position in queue
                del self._cache[text]
            # add text to the 'seen cache'
            self._cache[text] = None

        return is_cache_hit

    def check_collision_and_add(self, tweet):
        return self.check_collision(tweet, append_to_cache=True)
