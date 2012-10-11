#!usr/bin/python2.6.6

from collections import defaultdict
from operator import itemgetter
import math
import os
import sys

class ViterbiTagger(object):
    def __init__(self, train_file=None, input_file=None):
        self.train_file = train_file
        self.input_file = input_file

    def train(self, train_file=None):
        """Trains the tagger on a string of token/tag pairs.
        """
        if train_file is None:
            if self.train_file is None:
                sys.stderr.write("No training file given to Viterbi.\n")
                sys.exit(1)
            else: train_file = self.train_file
        else: self.train_file = train_file
        
        if not os.path.isfile(train_file):
            sys.stderr.write("Training file %s does not exist.\n" % train_file)
            sys.exit(1)

        self.toks = defaultdict(int)  # Token counts.
        self.tags = defaultdict(int)  # Tag counts.
        self.miss = defaultdict(int)  # Emission counts.
        self.tok_sings = defaultdict(int)  # sing_tw
        self.tag_sings = defaultdict(int)  # sing_tt

        x = None
        handle = open(train_file)
        for y in handle:
            self.__print_progress(self.toks[''], 1000)
            if x is not None:
                self.__count(x, y)
            x = y
        handle.close()

        # Token dictionary and default tag dictionary.
        self.tok_dict = set([tok for tok in self.toks.keys() if tok.find('/') is -1 and not tok == ''])
        self.tag_dict = set([tag for tag in self.tags.keys() if tag.find('/') is -1 and tag not in ('', '###')])

        # Tag dictionaries for each known word.
        # I tried using a generator function to do this dynamically, but it took ages.
        self.tag_dicts = defaultdict(set)
        for k in self.miss.keys():
            (tag, word) = k.split('/')
            self.tag_dicts[word].add(tag)

        # Precomputed tag transition probabilities.
        self.trans = {}
        for k in self.tags.keys():
            if k.find('/') is not -1:
                self.trans[k] = self.__p_tt(k)

        # Precomputed token emission probabilities.
        self.emiss = {}
        for k in self.miss.keys():
            self.emiss[k] = self.__p_tw(k)

        sys.stderr.write("\n")  # Newline because of print_progress.

    def __p_tw(self, key):
        """Computes the smoothed token emission probability.
        """
        (tag, tok) = key.split('/')
        
        lamb = self.tok_sings[tag]
        if not lamb:
            lamb = 1e-100
        backoff = float(self.toks[tok] + 1) / (self.toks[''] + len(self.tok_dict))

        num = self.miss[key] + lamb * backoff
        den = self.tags[tag] + lamb

        return math.log(num / den)

    def __p_tt(self, key):
        """Computes the smoothed tag transition probability.
        """
        (t1, t2) = key.split('/')

        lamb = self.tag_sings[t1]
        if not lamb:
            lamb = 1e-100
        backoff = float(self.tags[t2]) / (self.toks[''] - 1)

        num = self.tags[key] + lamb * backoff
        den = self.tags[t1] + lamb

        return math.log(num / den)

    def __count(self, x, y):
        """Counts a token/tag bigram
        """
        x = x.strip().split('/')
        y = y.strip().split('/')
        tag_key = '/'.join((x[1], y[1]))
        emi_key = '/'.join((y[1], y[0]))

        self.toks[''] += 1  # Total number of tokens/tags.

        self.toks[y[0]] += 1  # Unigram counts for p(y_tok).
        if self.toks[y[0]] is 1:
            self.tok_sings[y[1]] += 1
        elif self.toks[y[0]] is 2:
            self.tok_sings[y[1]] -= 1

        self.tags[y[1]] += 1  # Unigram count for p(y_tag).
        if self.tags[y[1]] is 1:
            self.tag_sings[y[1]] += 1
        elif self.tags[y[1]] is 2:
            self.tag_sings[y[1]] -= 1

        self.miss[emi_key] += 1  # Emission count for p(y_tok|y_tag).
        self.tags[tag_key] += 1  # Bigram counts for p(y_tag|x_tag).

    def tag(self, input_file=None):
        """Computes the best tag sequence for a string of tokens.
        """
        if input_file is None:
            if self.input_file is None:
                sys.stderr.write("No input file given to Viterbi.\n")
                sys.exit(1)
            else: input_file = self.input_file
        else: self.input_file = input_file

        if not os.path.isfile(input_file):
            sys.stderr.write("Input file %s does not exist.\n" % input_file)
            sys.exit(1)

        # Read the observation/tag pairs from <input_file>.
        obs = []
        handle = open(input_file)
        for o in handle:
            self.__print_progress(len(obs), 1000)
            obs.append(o.strip())
        handle.close()

        # Tag!
        viterbi = {'0/###': 0.0}
        paths = {'###': ['###']}
        for i in xrange(1, len(obs)):
            self.__print_progress(i, 1000)
            w1 = obs[i-1].split('/')[0]
            w2 = obs[i].split('/')[0]
            (cands, new_paths) = ({}, {})
            for t2 in self.__tag_dict(w2):
                emiss_key = '/'.join((t2, w2))
                try:
                    emiss_prob = self.emiss[emiss_key]
                except KeyError:
                    emiss_prob = self.__p_tw(emiss_key)
                for t1 in self.__tag_dict(w1):
                    trans_key = '/'.join((t1, t2))
                    try:
                        trans_prob = self.trans[trans_key]
                    except KeyError:
                        trans_prob = self.__p_tw(trans_key)
                    cands[t1] = (emiss_prob + trans_prob +
                                 viterbi['/'.join((str(i-1), t1))])
                (t1, prob) = max(cands.items(), key=itemgetter(1))
                new_paths[t2] = paths[t1] + [t2]
                viterbi['/'.join((str(i), t2))] = prob
            paths = new_paths
        path = paths['###']

        # Score!
        (known, novel) = (0, 0)
        (known_y, novel_y) = (0, 0)
        for i in xrange(len(obs)):
            (word, gold) = obs[i].split('/')
            if not word == '###':
                tag = path[i]
                if word in self.tok_dict:
                    known += 1  # Count the number of known tokens.
                    if tag == gold:
                        known_y += 1  # Count the number of correctly-tagged known tokens.
                else:
                    novel += 1  # Count the number of novel tokens.
                    if tag == gold:
                        novel_y += 1  # Count the number of correctly-tagged novel tokens.

        # Convert counts to percentages.
        self.total = 100.0 * (known_y + novel_y) / (known + novel)
        if known:
            self.known = 100.0 * known_y / known
        else: self.known = 0
        if novel:
            self.novel = 100.0 * novel_y / novel
        else: self.novel = 0
        self.perplexity = math.exp(-viterbi['/'.join((str(len(obs)-1), '###'))] / float(len(obs)-1))

        sys.stderr.write("\n")  # Newline because of print_progress.

    def __tag_dict(self, word):
        """Returns the appropriate tag dictionary for the given word.
        """
        if word in self.tok_dict:
            return self.tag_dicts[word]
        else:
            return self.tag_dict

    def print_score(self):
        print 'Tagging accuracy: %.2f%%  (known: %.2f%% novel: %.2f%%)' % (self.total, self.known, self.novel)
        print 'Perplexity per tagged test word: %.3f' % self.perplexity

    def __print_progress(self, current, interval):
        if not current % interval:
            sys.stderr.write('.')

def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: %s <train> <input>\n" % os.path.basename(sys.argv[0]))
        sys.exit(1)

    # All of the data is in tagging/; check there.
    default_dir = 'tagging'
    (train_file, input_file) = sys.argv[1:3]
    if not os.path.isfile(train_file):
        train_file = os.path.join(default_dir, train_file)
    if not os.path.isfile(input_file):
        input_file = os.path.join(default_dir, input_file)

    vtag = ViterbiTagger(train_file, input_file)

    sys.stderr.write('Training...')
    vtag.train()
    sys.stderr.write("Done training on %d tokens (%d unique tokens, %d unique tags).\n\n" %
                     (vtag.toks[''], len(vtag.tok_dict), len(vtag.tag_dict)))
    sys.stderr.write('Tagging...'),
    vtag.tag()
    sys.stderr.write("Done tagging.\n\n")
    vtag.print_score()

    sys.exit(0)

if __name__ == '__main__':
    main()
