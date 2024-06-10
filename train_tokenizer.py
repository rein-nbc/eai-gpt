import os
import argparse
import time
from natsort import natsorted
from tokenizer.regex import RegexTokenizer
from efficient_tokenizer.ebpe import BPETrainer

def parse_args():
    parser = argparse.ArgumentParser("Entry script to launch training")
    parser.add_argument("--data-dir", type=str, default = "./data", help="Path to the data directory")
    parser.add_argument("--output-dir", type=str, default = "./tokenizer_model", help="Path to the output file")
    parser.add_argument("--resume-training", type =bool, default=False, help="Resume training")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    resume = args.resume_training

    os.makedirs(output_dir,exist_ok=True)

    tokenizer = BPETrainer(
        int(10000), min_freq=1,
        compress_threshold=0.3,
        single_char=False
    ).train_from_file(
        data_dir,
        verbose=True,
    )

    tokenizer.dump_vocab(os.path.join(output_dir,"vocab.json"))

    text = " Once upon a time, there is a regulated"
    print(tokenizer.tokenize(text))

if __name__=='__main__':
    main()