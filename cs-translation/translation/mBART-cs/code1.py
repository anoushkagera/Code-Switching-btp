from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import os
import sys
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
import jsonlines

BASE_DIR = f'{sys.argv[1]}/preprocessed/'
DATA_DIR = sys.argv[2]+'/'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

def parse_iitb_file(file_en, file_hi, data_id):
    normalizer = DevanagariNormalizer()
    en_data = []
    hi_data = []
    with open(file_en) as f2, open(file_hi) as f1:
        for src, tgt in zip(f1, f2):
            hi_data.append(src.strip() + '\n')
            en_data.append(tgt.strip() + '\n')
    for i in range(len(hi_data)):
        hi_data[i] = normalizer.normalize(hi_data[i])
    assert len(en_data) == len(hi_data)
    print(f'total size of {data_id} data is {len(en_data)}')
    return en_data, hi_data

#make hindi english sentences seperately from the files

def parse_jsonl(file_jsonl, data_id):
    normalizer = DevanagariNormalizer()
    en_data = []
    hi_data = []
    with jsonlines.open(file_jsonl) as reader:
        for obj in reader:
            hi_data.append(' '.join([(x[1] if x[2] == 'hi' else x[1]) for x in obj['Hinglish']]) + '\n')
            en_data.append(' '.join(obj['English']) + '\n')
    for i in range(len(hi_data)):
        hi_data[i] = normalizer.normalize(hi_data[i])
    assert len(en_data) == len(hi_data)
    print(f'total size of {data_id} data is {len(en_data)}')
    return en_data, hi_data

# english and hindi sentences
# transliterates hinglish sentences
def train_dev_test_split(en_data, hi_data, dev_size, test_size):
    total_test = dev_size + test_size
    en_train, en_subtotal, hi_train, hi_subtotal = train_test_split(en_data, hi_data, test_size=total_test, random_state=42)
    en_val, en_test, hi_val, hi_test = train_test_split(en_subtotal, hi_subtotal, test_size=test_size, random_state=42)
    return en_train, hi_train, en_val, hi_val, en_test, hi_test

# Datasets

#for training, testing and validation
iitb_en_train, iitb_hi_train = parse_iitb_file(DATA_DIR+'iitb_corpus/parallel/IITB.en-hi.en', DATA_DIR+'iitb_corpus/parallel/IITB.en-hi.hi', 'IITB_TRAIN')
iitb_en_val, iitb_hi_val = parse_iitb_file(DATA_DIR+'iitb_corpus/dev_test/dev.en', DATA_DIR+'iitb_corpus/dev_test/dev.hi', 'IITB_VALIDATION')
iitb_en_test, iitb_hi_test = parse_iitb_file(DATA_DIR+'iitb_corpus/dev_test/test.en', DATA_DIR+'iitb_corpus/dev_test/test.hi', 'IITB_TEST')

dh_en_train, dh_hi_train, dh_en_val, dh_hi_val, dh_en_test, dh_hi_test = train_dev_test_split(*parse_jsonl(DATA_DIR+'mrinal_dhar.jsonl', 'DHAR'), 604, 604)
ph_en_train, ph_hi_train, ph_en_val, ph_hi_val, ph_en_test, ph_hi_test = train_dev_test_split(*parse_jsonl(DATA_DIR+'phinc.jsonl', 'PHINC'), 1374, 1374)

file_mapping = {
    'train.en_XX': dh_en_train + ph_en_train,
    'train.hi_IN': dh_hi_train + ph_hi_train,
    'valid.en_XX': dh_en_val + ph_en_val,
    'valid.hi_IN': dh_hi_val + ph_hi_val,
    'test.en_XX': dh_en_test,
    'test.hi_IN': dh_hi_test,
    'testph.en_XX': ph_en_test,
    'testph.hi_IN': ph_hi_test,
    'iitbtrai.en_XX': iitb_en_train,
    'iitbtrai.hi_IN': iitb_hi_train,
    'iitbvali.en_XX': iitb_en_val,
    'iitbvali.hi_IN': iitb_hi_val,
    'iitbtest.en_XX': iitb_en_test,
    'iitbtest.hi_IN': iitb_hi_test,
}

for k, v in file_mapping.items():
    with open(f'{BASE_DIR}{k}', 'w') as fp:
        fp.writelines(v)