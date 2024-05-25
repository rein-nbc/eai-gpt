import numpy as np
import os
import argparse


def main(args):
    data_dir = args.data_dir
    portion = args.portion

    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    n_data = len(data)
    n_subset = int(n_data * portion)

    print(n_data, n_subset)

    data_subset = np.memmap('train.bin', dtype=np.uint16, mode='w+', shape=(n_subset,))
    data_subset[:n_subset] = data[:n_subset]

    data_subset.flush()


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', default='data/openwebtext')
parser.add_argument('-o', '--output_dir', default='data/openwebtext-subset')
parser.add_argument('-p', '--portion', default=0.1)

args = parser.parse_args()

main(args)
