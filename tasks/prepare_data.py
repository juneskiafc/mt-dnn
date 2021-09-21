import csv
import shutil
import sys
from pathlib import Path
csv.field_size_limit(sys.maxsize)

DATA_PATH = Path('../data/')

MNLI_TRAIN = DATA_PATH.joinpath('multinli_1.0_train.txt')
XNLI_DEV = DATA_PATH.joinpath('xnli.dev.tsv')
XNLI_TEST = DATA_PATH.joinpath('xnli.test.tsv')

CROSS_TRAIN = DATA_PATH.joinpath('cross_train.tsv')
CROSS_TEST = DATA_PATH.joinpath('cross_test.tsv')
MULTI_TRAIN = DATA_PATH.joinpath('multi_train.tsv')
MULTI_TEST = DATA_PATH.joinpath('multi_test.tsv')

# MNLI train set is also cross-lingual training set
if not CROSS_TRAIN.is_file():
    shutil.copy(MNLI_TRAIN, CROSS_TRAIN)

# cross-lingual test set is XNLI test set
if not CROSS_TEST.is_file():
    shutil.copy(XNLI_TEST, CROSS_TEST)

# multi-lingual train set is MNLI train + XNLI dev
with open(MNLI_TRAIN, 'r') as mnli_f:
    with open(XNLI_DEV, 'r') as xnli_f:
        mnli_rd = csv.DictReader(mnli_f, delimiter='\t')
        xnli_rd = csv.DictReader(xnli_f, delimiter='\t')

        with open(MULTI_TRAIN, 'w') as fw:
            fieldnames = ['id', 'label', 'premise', 'hypothesis']
            writer = csv.DictWriter(fw, fieldnames, delimiter='\t')
            writer.writeheader()

            for rd in [mnli_rd, xnli_rd]:
                for i, row in enumerate(rd):
                    writer.writerow({
                        'id': row['pairID'],
                        'label': row['gold_label'],
                        'premise': row['sentence1'],
                        'hypothesis': row['sentence2']
                    })

# multi-lingual test set is XNLI test
if not MULTI_TEST.is_file():
    shutil.copy(XNLI_TEST, MULTI_TEST)


