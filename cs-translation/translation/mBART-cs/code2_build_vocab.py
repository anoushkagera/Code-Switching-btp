import argparse
from glob import glob

from fairseq.data import Dictionary
from fairseq.tokenizer import tokenize_line

# argparse is used to parse command-line arguments.
# glob is used to find files matching a specified pattern.
# Dictionary from fairseq.data is used to create and manipulate vocabulary dictionaries.
# tokenize_line from fairseq.tokenizer is used to tokenize lines of text.

def pad_dict(d: Dictionary, num_extra_symbols: int, padding_factor: int = 8) -> None:
    i = 0
    while (len(d) + num_extra_symbols) % padding_factor != 0:
        symbol = f"madeupword{i:04d}"
        d.add_symbol(symbol, n=0)
        i += 1

# main() function is the entry point of the script.
# It sets up command-line argument parsing using argparse.
# Three arguments are expected:
# --corpus-data: The path pattern (glob) to all tokenized corpus files (train, test, val).
# --langs: The pre-trained model languages.
# --output: The vocabulary file where the output will be saved.

def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabulary from corpus data.")
    parser.add_argument("--corpus-data", type=str, required=True, help="The path pattern (glob) to all tokenized corpus files (train, test, val).")
    parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--output", type=str, required=True, help="The vocabulary file.")
    args = parser.parse_args()

    langs = args.langs.split(",")
    ft_dict = Dictionary()
    for data_path in glob(args.corpus_data):
        Dictionary.add_file_to_dictionary(data_path, ft_dict, tokenize_line, 4)
    ft_dict.finalize(padding_factor=0)
    pad_dict(ft_dict, len(langs) + 1)
    ft_dict.save(args.output)

if __name__ == "__main__":
    main()

# Usage
# python build_vocab.py --corpus-data "./ft/*.spm.*" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output ./ft/dict.txt