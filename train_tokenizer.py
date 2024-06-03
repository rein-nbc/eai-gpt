import os
import argparse
import time
from natsort import natsorted
from tokenizer.regex import RegexTokenizer

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

    # open some text and train a vocab of 10000 tokens
    # for file_name in natsorted(os.listdir(data_dir)):
    #     file_path = os.path.join(data_dir,file_name)
    #     text = open(file_path, "r", encoding="utf-8").read()
    #     prefix = "trained_model"
    #     print(f"---Finished reading file from {file_path}---\n")
    #     t0 = time.time()

    #     # construct the Tokenizer object and kick off verbose training
    #     tokenizer = RegexTokenizer()
    #     if resume:
    #         print(f"Load checkpoint from {os.path.join(output_dir, f'{prefix}.model')}")
    #         tokenizer.load(os.path.join(output_dir,f"{prefix}.model"))
    #     print("---Start training tokenizer---\n")
    #     tokenizer.train(text, 512, verbose=True)
    #     # writes two files in the models directory: name.model, and name.vocab
    #     tokenizer.save(os.path.join(output_dir,prefix))
    #     resume=True
    #     t1 = time.time()

    #     print(f"Training took {t1 - t0:.2f} seconds")
    #         file_path = os.path.join(data_dir,file_name)
    
    text = open(data_dir, "r", encoding="utf-8").read()
    prefix = "trained_model"
    
    t0 = time.time()

    # construct the Tokenizer object and kick off verbose training
    tokenizer = RegexTokenizer()
    if resume:
        print(f"Load checkpoint from {os.path.join(output_dir, f'{prefix}.model')}")
        tokenizer.load(os.path.join(output_dir,f"{prefix}.model"))
    print("---Start training tokenizer---\n")
    tokenizer.train(text, 10000, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    tokenizer.save(os.path.join(output_dir,prefix))
    resume=True
    t1 = time.time()

    print(f"Training took {t1 - t0:.2f} seconds")

if __name__=='__main__':
    main()