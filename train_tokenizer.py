import os
import argparse
import time
from natsort import natsorted
from tokenizer.ebpe import BPE,BPETrainer

def parse_args():
    parser = argparse.ArgumentParser("Entry script to launch training")
    parser.add_argument("--data-dir", type=str, default = "./data", help="Path to the data directory")
    parser.add_argument("--output-dir", type=str, default = "./tokenizer_model", help="Path to the output file")
    parser.add_argument("--load-pretrained", type =bool, default=False, help="Load trained tokenizer")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    load_pretrained = args.load_pretrained

    os.makedirs(output_dir,exist_ok=True)

    if load_pretrained:
        tokenizer=BPE(None)
        tokenizer.load_vocab(os.path.join(output_dir,"vocab.json"))
    else:
        tokenizer = BPETrainer(
            int(10000), min_freq=1,
            compress_threshold=0.3
        ).train_from_file(
            data_dir,
            verbose=True,
        )

        tokenizer.dump_vocab(os.path.join(output_dir,"vocab.json"))

    text = " Once upon a time, there is a wolf"
    print(tokenizer.encode(text))

    # print(f"decode: {tokenizer.decode([697, 297, 32, 659, 133, 32, 97, 32, 523, 101, 44, 32, 175, 510, 32, 195, 32, 97, 32, 545, 108, 102])}")

if __name__=='__main__':
    main()