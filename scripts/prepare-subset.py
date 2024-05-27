import numpy as np
import os
import argparse
import shutil

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    portion = args.portion

    data = np.memmap(os.path.join(input_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    n_data = len(data)
    # n_subset = int(n_data * portion)
    n_subset = int(0.1 * 9035582489)

    print(n_subset)

    data_subset = np.memmap(os.path.join(output_dir, 'train.bin'), dtype=np.uint16, mode='w+', shape=(n_subset,))
    data_subset[:n_subset] = data[:n_subset]

    data_subset.flush()

    shutil.copy(os.path.join(input_dir, 'val.bin'), os.path.join(output_dir, 'val.bin'))


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', default='data/openwebtext')
parser.add_argument('-o', '--output_dir', default='data/openwebtext-subset')
parser.add_argument('-p', '--portion', default=0.1)

args = parser.parse_args()

main(args)
