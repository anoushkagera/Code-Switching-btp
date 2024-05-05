from sys import stdin
from nltk.tokenize.casual import casual_tokenize

for line in stdin:
	line = casual_tokenize(line.rstrip(), preserve_case=False, reduce_len=True, strip_handles=False)
	print(' '.join(line))