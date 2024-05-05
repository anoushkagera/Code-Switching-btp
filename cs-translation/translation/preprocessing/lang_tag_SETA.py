import jsonlines
from three_step_decoding import *
from nltk.tokenize.casual import casual_tokenize

tsd = ThreeStepDecoding('lid_models/hinglish',
						htrans='nmt_models/rom2hin.pt',
						etrans='nmt_models/eng2eng.pt')

dataset = []
dataset_t = []

with open('../../DataSets/SETA/Dhar/s-enhi.txt') as f:
	for line in f:
		line = line.rstrip()
		line = casual_tokenize(line, preserve_case=True, reduce_len=True, strip_handles=False)
		dataset.append(line)

with open('../../DataSets/SETA/Dhar/t-en.txt') as f:
	for line in f:
		line = line.rstrip()
		line = casual_tokenize(line, preserve_case=True, reduce_len=True, strip_handles=False)
		dataset_t.append(line)

valid_idx = [i for i in range(len(dataset)) if dataset[i] != [] and dataset_t[i] != []]

dataset = list([dataset[i] for i in valid_idx])
dataset_t = list([dataset_t[i] for i in valid_idx])

assert(len(dataset) == len(dataset_t))

with jsonlines.open('../../DataSets/SETA_PRE/mrinal_dhar.jsonl', mode='w') as writer:
	for i in range(len(dataset)):
		if i%10 == 0:
			print(f"{i}/{len(dataset)} completed")
		isSingleWord = False
		if len(dataset[i]) == 1:
			isSingleWord = True
			dataset[i].append('.')
		temp = list(tsd.tag_sent(' '.join(dataset[i])))
		if isSingleWord:
			temp.pop()
		writer.write({
				"English": dataset_t[i],
				"Hinglish": temp
			}
		)
