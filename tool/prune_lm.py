import sys
from enum import Enum

class State(Enum):
    BEGIN = 0
    DATA = 1
    UNIGRAM = 2
    NGRAM = 3
    END = 4

state = State.BEGIN
for line in sys.stdin:
    if state == State.BEGIN:
        if line.strip() == '':
            pass
        elif line.strip() == 'iARPA':
            pass
        elif line.strip() == '\\data\\':
            state = State.DATA
            print(line.strip())
        else:
            raise Exception('unexpected line: {}'.format(line))
    elif state == State.DATA:
        f = line.strip().split('=')
        if line.strip() == '':
            print(line.strip())
        elif len(f) == 2:
            if f[0].strip() == 'ngram 1':
                print(line.strip())
        elif line.strip() == '\\1-grams:':
            state = State.UNIGRAM
            print(line.strip())
        else:
            raise Exception('unexpected line: {}'.format(line))
    elif state == State.UNIGRAM:
        f = line.strip().split()
        if line.strip() == '':
            print(line.strip())
        elif line.strip()[0] == '\\':
            state = State.NGRAM
        elif len(f) in {2, 3}:
            print('{} {}'.format(f[0], f[1]))
        else:
            raise Exception('unexpected line: {}'.format(line))
    elif state == State.NGRAM:
        if line.strip() == '\\end\\':
            state = State.END
            print(line.strip())
        else:
            pass
    elif state == State.END:
        pass

if state != State.END:
    raise Exception('unexpected end-of-file')


