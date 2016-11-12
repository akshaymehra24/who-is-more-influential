# File for applying transformation to the original data
from os.path import join
from math import log

DATA_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Original Dataset'
OUTPUT_DIR = 'C:/Users/Patrick/PycharmProjects/who-is-more-influential/Python'

in_train = open(join(DATA_DIR, 'train.csv'))
out_train = open(join(OUTPUT_DIR, 'log_train.csv'), 'w+')
header = in_train.next().rstrip()


def log_xf(a, b):
    return str(log(a + 1) - log(b + 1))

# Read from input, apply transform
print >> out_train, header
for line in in_train:
    splitted = line.rstrip().split(',')
    row_data = splitted[:1]     # label
    a_features = [float(item) for item in splitted[1:12]]
    b_features = [float(item) for item in splitted[12:]]
    transformed_data = [log_xf(a_features[i], b_features[i]) for i in range(0, 11)]
    row_data.extend(transformed_data)
    print >> out_train, ','.join(row_data)
in_train.close()
out_train.close()
