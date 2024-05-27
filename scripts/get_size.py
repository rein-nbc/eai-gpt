import numpy as np
import os
import argparse

def main(args):
    input_dir = args.input_dir

    data = np.memmap(os.path.join(input_dir, 'train.bin'), dtype=np.uint16, mode='r')

    n_data = len(data)
    print(n_data)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', default='data/openwebtext-subset')

args = parser.parse_args()

main(args)
