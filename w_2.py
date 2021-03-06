import numpy as np
import copy

def suffix_array(text):
	suffixes = []
	indices = []	
	for i in range(len(text)):
		suffixes.append(text[i:])
		indices.append(i)
	indices = [x for _, x in sorted(zip(suffixes,indices), key=lambda pair: pair[0])]
	return indices, suffixes

def burrows_wheeler_transform(text):
	text = list(text)
	M = []
	for i in range(len(text)):
		x = copy.deepcopy(text)
		y = x[-i: ]
		del x[-i: ]
		x = y + x
		M.append(''.join(x))
	M.sort()
	bw = ''
	for string in M:
		bw += string[-1]
	return bw

def inverse_burrows_wheeler_transform(bw):
	last_column = list(bw)
	labels = {}
	lc = []
	for i in range(len(bw)):
		lc.append((last_column[i], i))
	
	fc = copy.deepcopy(lc)
	fc.sort()
	for i in range(len(lc)):
		lc[i] = lc[i][0] + str(lc[i][1])

	for i in range(len(fc)):
		fc[i] = fc[i][0] + str(fc[i][1])

	mat = []
	for i in range(len(bw)):
		mat.append([fc[i], lc[i]])

	i = 0
	l = []
	text = ''

	while len(l) != len(bw):
		find = fc[i]
		index_find_lc = lc.index(find)
		symbol = fc[index_find_lc]
		l.append(symbol)
		i = fc.index(symbol)

	for j in l:
		text += j[0]
	return text

def patterns_input(file):
	read = open(file)
	patterns = []
	for line in read:
		l = line.strip()
		l = l.split(' ')
		patterns += l
	return patterns

def last_to_first(bw):
	last_column = list(bw)
	labels = {}
	lc = []
	for i in range(len(bw)):
		lc.append((last_column[i], i))
	
	fc = copy.deepcopy(lc)
	fc.sort()
	for i in range(len(lc)):
		lc[i] = lc[i][0] + str(lc[i][1])

	for i in range(len(fc)):
		fc[i] = fc[i][0] + str(fc[i][1])

	mat = []
	for i in range(len(bw)):
		mat.append([fc[i], lc[i]])

	ltf = []
	i = 0

	while len(ltf) != len(bw):
		find = lc[i]
		ltf.append(fc.index(find))
		i += 1
	return ltf, last_column

def bw_matching(last_to_first, bw, pattern):
	top = 0
	bottom = len(last_column) - 1
	top_index, bottom_index = 0, 0
	while top <= bottom:
		if len(pattern) > 0:
			symbol = pattern[-1]
			pattern = pattern[0: len(pattern)-1]
			x = last_column[top: bottom + 1]
			if symbol in x:
				top_index = bw.find(symbol, top, bottom + 1)
				bottom_index = bw.rfind(symbol, top, bottom + 1)
				top = last_to_first[top_index]
				bottom = last_to_first[bottom_index]

			else:
				return 0

		else:
			return bottom - top + 1


def bw_matching_all_patterns(last_to_first, bw, patterns):
	l = []
	for pattern in patterns:
		l.append(bw_matching(last_to_first, bw, pattern))
	return l
'''
bw = 'AGGTCAGAGGAGTACTGTGGTTCGCAGCGCATGTCCGCGCTGTGCGATTTGGTGCACACGATGTTGGGCTTTTACTGATCATGGACTGGCGAACCGACGCCCGACTCCCTTAATGGGTGTGCTTTCCAAGCGGAGACTGTGCACGACTAGCTCCTCATATTCTCAATATGACACCCTCCATTTAGAACGGAGTGTCTACGACCCTATATTCAAAGGAAGTCGGCGAGTATACTTCGCTAACCCCGCATTTGATGTTCCGGCTTCACACACCCATACTAGCAATATCTCACTGTAGCATCACCATACTCGTCGCGATCTAAATCCCCGTCGCCAGCAGGACATGCGGGCACGGGCACAAGCTGTGGTCCTCAGGATTCCTTGGGCATCCGCTTTTTCGTTGGACGG$CAGGAGATGCCGCGATCTCTACGCATGTGAATAACGTACGGCTGATCGTCGGTTCTAACCCGGTTGCGGGAAGAGTCTATCTCTCTCGGAGCTACTCCGTGTTTAAAACGCTAATAATAACGGATCAAGGGGTAGTCCGCTGGGATATACGCGTCCCTAATGAGAACTGTGGAAAGCCTTGTTCCCCATTCGTCTAAGTTACCTATGTCCGTAGCGTTCCGGTTTGACTGTGGTCGGTATCCGTCCTCTTCGAGGCTGGATATGGTAGATTACCCGCTAAGTCTGAAAAGAGCTAGGGTACCGCCGTCGACGCTGTAAGAGGATCTACTATACGTAGGCACCATCTAGTCAGTGACCCGACATTCACACATGCTGCACTAAGGACCTGTTGTACCGAGCCCGTACGTTTCGTGTCACCCAGTCCGGAGTCATGCAGAGACAGCTCCACCGTCGTAACGACCGAACCCTTCTCGTCGTTCACCTCGAACGCCGACTATGATCTACGGTTAACCCAACATAGTTACAGCTTCGCGAAGTGCGAGAACGCTACAGCTCGGTGTACCTGAAGAACCAGCGTCATGGTACTGGCTATGGG'
ltf, last_column = last_to_first(bw)
file = 'patterns.txt'
patterns = patterns_input(file)
l = bw_matching_all_patterns(ltf, bw, patterns)
for i in l:
	print(i, end = ' ')
'''		
'''
bw = 'G$CAGCTAGGG'
print(inverse_burrows_wheeler_transform(bw))
'''
'''
text = 'ACCAACACTG$'
print(burrows_wheeler_transform(text))
'''
'''
text = 'cocoon$'
print(suffix_array(text))
'''