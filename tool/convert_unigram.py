import sys
from enum import Enum
import struct
import math

INF = float('inf')
M_LN10 = math.log(2) / math.log10(2)

class State(Enum):
    BEGIN = 0
    DATA = 1
    UNIGRAM = 2
    NGRAM = 3
    END = 4

def read_unigram(filename):
    state = State.BEGIN
    unigram_cost = {}
    with open(filename, encoding = 'utf8') as fd:
        for line in fd:
            if state == State.BEGIN:
                if line.strip() == '':
                    pass
                elif line.strip() == '\\data\\':
                    state = State.DATA
                else:
                    raise Exception('unexpected line: {}'.format(line))
            elif state == State.DATA:
                f = line.strip().split('=')
                if line.strip() == '':
                    pass
                elif len(f) == 2:
                    pass
                elif line.strip() == '\\1-grams:':
                    state = State.UNIGRAM
                else:
                    raise Exception('unexpected line: {}'.format(line))
            elif state == State.UNIGRAM:
                f = line.strip().split()
                if line.strip() == '':
                    pass
                elif line.strip() == '\\end\\':
                    state = State.END
                elif line.strip()[0] == '\\':
                    state = State.NGRAM
                elif len(f) in {2, 3}:
                    unigram_cost[f[1]] = float(f[0])
                else:
                    raise Exception('unexpected line: {}'.format(line))
            elif state == State.NGRAM:
                if line.strip() == '\\end\\':
                    state = State.END
                else:
                    pass
            elif state == State.END:
                pass
    if state != State.END:
        raise Exception('unexpected end-of-file')
    return unigram_cost

def read_vocab(filename):
    vocab = {}
    with open(filename, encoding = 'utf8') as fd:
        for line in fd:
            line = line.strip()
            if line == '':
                continue
            fields = line.split()
            if len(fields) != 2:
                raise Exception('unexpected line in vocab: ' + line)
            word = fields[0]
            word_id = int(fields[1])
            vocab[word] = word_id
    return vocab

def build_cost_array(vocab, ucost):
    array_size = max(vocab.values()) + 1
    array = [INF for _ in range(array_size)]
    for word, cost in ucost.items():
        if word not in vocab:
            if word == '<unk>':
                print('ignore word <unk>')
                continue
            else:
                raise Exception('unexpected word: ' + word)
        word_id = vocab[word]
        array[word_id] = -cost * M_LN10
        print('{} {} {}'.format(word, word_id, cost))
    return array

def write_vec(cost_array, filename):
    with open(filename, 'wb') as fd:
        fd.write(b"VEC0")
        fd.write(struct.pack("<i", len(cost_array) * 4 + 4))
        fd.write(struct.pack("<i", len(cost_array)))
        for v in cost_array:
            fd.write(struct.pack("<f", v))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python3 convert_unigram.py <lm_arpa> <vocab> <unigram_bin>')
        sys.exit(22)
    
    lm_file = sys.argv[1]
    vocab_file = sys.argv[2]
    cost_file = sys.argv[3]

    vocab = read_vocab(vocab_file)
    unigram = read_unigram(lm_file)
    unigram['<s>'] = 0
    costs = build_cost_array(vocab, unigram)
    write_vec(costs, cost_file)
    print('ok')







