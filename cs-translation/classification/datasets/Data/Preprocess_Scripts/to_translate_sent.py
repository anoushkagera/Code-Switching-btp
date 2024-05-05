import jsonlines
import csv
import glob

files = sorted(glob.glob('/home/bhavya24/PreTranslated_Data/Sentiment_EN_HI/*'))

f_write = open("/home/bhavya24/delete/sent_to_translate.txt", "w")

def write_line(line):
	f_write.write(' '.join(line.split()[:400]) + '\n')

for file in files:
	with open(file, 'r') as f_r:
		for line in f_r:
			temp = line.strip().split('\t')
			assert(len(temp) == 2)
			assert(temp[1] == temp[1].strip())
			assert(temp[0] == temp[0].strip())
			write_line(temp[0])

f_write.close()