import copy
from w_2 import *

def count_of_symbol_matrix(last_column):

	symbols = list(set(last_column))
	symbols.sort()
	nums = [i for i in range(len(symbols))]
	x = zip(symbols, nums)
	symbols_dict = dict(x)
	count = []
	for symbol in symbols:
		l = [0]
		for i in range(len(last_column)):
			if last_column[i] == symbol:
				l.append(l[i] + 1)
			else:
				l.append(l[i])
		count.append(l)
	return count, symbols_dict


def first_occurrence_matrix(last_column):
	first_column = list(last_column)
	first_column.sort()
	first_occurrence = {}
	for i in range(len(first_column)):
		symbol = first_column[i]
		if symbol not in first_occurrence:
			first_occurrence[symbol] = i
	return first_occurrence


def better_bw_matching(first_occurrence, last_column, pattern, count, symbols_dict):
	top = 0
	bottom = len(bw) - 1
	while top <= bottom:
		if len(pattern) > 0:
			symbol = pattern[-1]
			pattern = pattern[0: len(pattern)-1]
			x = last_column[top: bottom + 1]
			if symbol in x:
				top = first_occurrence[symbol] + count[symbols_dict[symbol]][top]
				bottom = first_occurrence[symbol] + count[symbols_dict[symbol]][bottom + 1] - 1
			else:
				return 0
		else:
			return bottom - top + 1

def better_bw_matching_all_patterns(first_occurrence, last_column, pattern, count, symbols_dict):
	l = []
	for pattern in patterns:
		l.append(better_bw_matching(first_occurrence, last_column, pattern, count, symbols_dict))
	return l

bw = 'CTTGCACGACCTGTACGCCGGGTAGCAGTCTGTATTCCCCGCAACTCGCTCACTTGTAAACCGGTCTCCAGCCACAATCGGGAGACCCTTAATATTGCAACTCTGATACTTCTACGCGCCCACTCGTCCAGTACCGACATGTCATGAACGTTCTTTCGCCATCGGTGCTAAGGAATGACATCAACCGATCAGCCAAATAGTATGCGGCTAAAAGAGCTAGATATGCAACGCCAATTCTCCAGGGTTGGGGCACCCTATATTCACCACGTTTCATGATTCGACCTCTTAAGTAGGAAGCGATCAACGTCCTTAGGTCCGGTGTAGAGGCCTACCATACCTCCAAAGACCTTTAACGATCGCTAGTATTGGGGGCCCGTTAAGTTGACATCCATCCTTCAGACGCCACGTTGAGCCCAGGTCCAGTGAGCCTTGTTGAAGGTGCCACGCCCGGTGATCTTCTTTTCCTCACTGTCTTTGAGAACGGGAAAAGAAACCTCCTCGTCCGGCCTCGCAATCGTCAACTACATAGAATGCATGCCACCCATGCGGGCTGAAGGGTACATGAATTCCCTCATTAGTCCAGCCAGCACTATCTTAGAGATTCAAGCCCAGACTGCGGATGTCACAGTCCATTAGAAGACGGGAGCAGTTTTTGCGGTTAGCCTCGTAACACCGGAGTGTGGACATGCTACCCTCTCACTTTCACGGAACGTGTATATGTGTGAATAGACCGATCACCAGACGGATGCTAGTTTAATCCGCTTCCCTTGATCGGTCATCAACATTAAATCATCATTGAGACTAGCGAAACAGCCGAGATTATCGGGGAAATACGTTGTGACTATGTCCTCCAGGCATACATCTTTCTTTCAGGACACCTAGCTTAACCAGGCGTGTCGGCAGAAAATGGATATCGCGGTAAGTGTTACGTCAAATTTCTAACGCAACCAACGTTCACGCCGCCTGGCTGAGAAAAACGAGTTGCTGGGAGAGAACGCTTCGGGACCCCTGAACTTACCGAGTCCATACTTTGCACCGATCTGTTTCCTCAGACAGGTTGAGCCGAATGGTTGATCAAAAATAACGCAGGTTCAGGTTTAACCCGTTTGCCCCGGGCTACATCAGTGACTACCCGGTCACGACTTCGGCAGTGCTGGCCAGGTTTTTTCGTACCGATATTCCATGCATGCTAGCGAAGGATGTCAGTGGCCGGCGGCGGATAGATCTCGTCCTCGATCCGCGTCTATGTTGCCAGATCATTCCCCCTAGAGGCTGGTGCCGACTTATAAAACACTCACTTGGCGCAGCAGGATGAGTGTTTGCGTTGGCCTTTACGAGGCTATACTTTTGTATGGACTAATCTTCGCTCAAACGGCGCCAGGCACTGAGCCGCGGCGTGGGCACCTGGCACTCAAGATGTCTTGCCCGCAAGTAGGCTGGCTGCCCTTGAGTGATCACTGTTAATCCACTTCCCTGCTTTGAAACAACCATCACCGCATGAACACTAGCGTCCTCGTGGTGAATCACCCGGGTGTCAATTTATTATCCAGCTGTTATACGTAATCAACGGCCTTGCGCCCTTAACGCTAACATGCACCTACGCATAATTGCAACACGGGCGTTGGCTTGCGAAAGTATCAAAAAATAAAGATGAAAGGAACAACTGGAAATAACCACCGTCGACGTCTCCTGGTATGGAAAAATCACAGGTTCGCCTACATGGGACTTGGCGTCGATAATCATGCGAAAGAAAGGCTTTTTGTCACTCGTTGGCGTATTATGCGGCTAGCAAGCCAATCACGACCATAAATCCACGATAGGTAGATACAAGATTAGTTAGATATAGAGGGCTCTACGCACACAGAGAACCTGCAATATGAGTCCGGGAAACTCCGCCATGATACGGTGGACGGGCTGTTGGTAATGCTTACAAGGTGATTGCGAAAATAGAAGGAGTGTCGACACGGGCGTATCCCATTTCTTAAAAGTACACGGGGGCAACTTTGAGGTGCGGCGTAAGTAACCAGTAGAACCTTGGTAGGCATAACGTTCCTAGATGTCAGTTAGAACGTCGTGTGCTTATCCACAGCCTTTATAGTTGTCTGCTGCCAATAATAGACCATCGTACGGATTGATCTGGGACGCGAACTAAACGTGTTTCACCGCTGGATGGGTGATTTGTGGCCGTGCGTAGTTAGGCAAAATCGAAGCTTCGTCGCATGGAAGCAGTCCTTAGAAGCCCATAGGAAAGGTGGAGATACAAACCCTTACAAGACACCTATGCGGTAGGTCAAAACCCTGGGGCGCGTCTAGGATAAACAGACCGGTAGAGCGGTCGCGTACGGGTTTCAATACACACTCCGGACGACAAACCTCATTTATCTGAAGCGATGCACATTGCGCGCGTACCACGCTACAATCAAGGTGGATGCGGAACAAATTCTTCCAGGCTTTGCTGAGGCGGCATAGACCCCCTGTTGATGATCTTCCGGGCAACGTAATCAGACCGAGTCCGGCCAGTACATGCGAAGAGTGGACCATATGTAGCACGCGTAGACAGCACAGAGTCAATCGACTTTGGAAGATGATTAAGCCCGAGCGTCTGAGAACACCCCGGGTAATGTCGAAGACAGTAGCGGTTGCGTCTCTTCTCGTGCTTGGGGGAAGTGGCCACACGGGTTGTCTCTGATTAATACGGGTAACGCCCTCATTGGCTCGAATTTGCTCGGGGTTATACGTAGCCTGGGTGAGGCAACCTTGGCCACTGCAACCCAGCCATGTAAAACCGCGCAAGTGAATGTGTTGGTAGAAAGTGCTTCAATGCAACGTCCACGAGGCACCACAGAGGTATCTCTGGAGAAAGCCGAGAACCTCTTGCAACGCCCAGAACGTCCGAGCCGACTGAGAATAATGCGTCCTGGGCACTGTGAGTCTGAGCGGAAGATATAGCTACTTGCGGCCTCCACCTGTCCCCAGGACCAACACCCCGAGATCGCACGCTAGTAGACGTTGTAGCCTGGGGTATCACGGGCTTTCCGTAGGCTCCTGCGCAACCCCTGAGCCCGACCCCACATTCATGTGGCGTACCGGAAATCTAGCCAGCTGGCATGGCTTCACTCCATGAAGAGATTTACTAAAGTTAGAGCAAGGTGAAAAGGCAGGCGCTTAGTAGACCTCGTCCGAATGTGCGCCCCCAACGGGCGCTTACTGGCGCCTACCATGCTACGCGGGGTCAAACGGCGGAGACAACTGAAGTCTCACTCACGCCCGTCTACAGAGAACAGCATCCCGCTCCTAGGGACAAAAACGGCCTGGAAGGGGATCAACGCTGCTTCGACAGGAACATACATCCGTTAGCAGCGCCTGGTCTGGCATAATGATGTTTAAACATACTGTAGTAGATGGGGCGAGTTGGAGGGTTTTTGCAAACCTAACGAGTTAGGTTGCAAACCCATTGGGTACAAGTACGGTATTGTATGCGCACATATGCGTAGTTACGTACCGTGTCATAACGCCCTTGGCAGTCAGAGTGAATACTACAGAATCGTATGACTCTAGGCCTTGCCTTTGTCCATTCCGATATTTCCTAACGGTTGGTACCGACGAGCCTAAATTACTCCCCTACTCACGACGAGCTCTAGCCCGTAAATTATAATCGTGCATCCCGCGAAGAGACCGCTTCTGTTACTAAATATTGCGGTTAGTCGTGCTCTTAGCATGATTTAACCATAACTCACATGGCGTCAATTGAGGGATCTTCATGCGACCCGTAAACTTGATTTGAGGAGAAGTCCACCCCAGCTCCTATCGTTATTTTTAATTGCCTGTCAGTTAGCATCGACGCCGTAATTTCTCGGTTACCAGATGATCCGCCCCTGACTTAAGCATTGTACCTTTAACTAGTCTCAAGGAGAAGCTTGAGAAGCCCGTTTGACTCCAATAAACAACCGAAAAACGGGTGAGTCAAGTAATGACGCGATAGCCTACTGCCCCATAAATCTATAGTCATAGTCCGAAAGGGCCCCCAGACGAACGACCATTCGAGGCACCTATGGAATGCCCCCTTGTTGGTGTCGCTATCGCAACGGGCGTACACCTGCGCTCCCTAATTAGCGTCTACTCAAAATCAAGTTACCTTAAACCTGGCCTGGATACGGTGAAGTTGTTGTTGTGGGCTCCCCTACACTATTACCGAGACTGTCGAGAATACGTGTGCAGCCGGTGGTGACTGTATAGGCTGCTGCATCTAAGGAAGTAAAACGGTCTGCGCGACGGCTTATCCGATGAAGTTAGACGTCAGCCAGCACCAAGCAGGGTACCAGCTATCCCAGGATCATTGGCTTTATCAGAATCGGATACCTAGGCTCGCTAACGAGCATCTCCGGTACGTTCGGAAACTAGTTTGGTACAGTTATAGGGCCATAATTCCAAATTGCTTCACGTAAATCGTCTTTGTCTGTCGGTATTCTATTCTTGAAAATCTCGATGCCTACCAAGGTAATAGTGAGGTGACCGGGTTCAAGTCCTACACTTTCGCGCTAACATGTGCACTCTAGACCTCAATAGGCACAATGGAGCGATTCAAGCGCCTCTATCCCCCTGTATAATGAATAGATCACACAGGGCTTATTCCAAAGGATCCCGACGTTGGTGGACGCTGTCCGCCAACTGGAAACGGCGAGGACACTTTGGTTCGGGATCAAGACATTTACCGGGATATTAAGATCGATGAAATCACACGAGAACCGACCAATTCGGCAAATGGGGGTTAATACGAGGCCTAGTGGTCCTAAACGATAAGGATCCTTAAAACAAATCCTTGGGTATCAAACCTTGACACCTCGTTGACATAAGCTTGTTGTGAGCCGTACGTGAGAAAACCTAGGGCGAGTCGAACCAATGAGCTGTTTTCAGATTTAGATTATCGAGAGGTTCACCACAGCCCTGGTCCGACACATCTACCGTGTGGCAGATAACGTCCTGCAATAAGGATCTTTGCAAGGTTTTCTTTGGGATCCAAAGAACGATGAGCATGTCTCATGTTCGCTCGCGTATCCCGCAGGTCCGTCCCTGTGTCAACGCACGAGTAACCGATATCGGTGTTCGAGGTTATGCTTTAATCCTGGAAGTGATCCCCCCTTTTGGAAAGCTTCACTATAGCCCGTTGCCGATTAGGTCAAGAGCTTCGTCCGCCAACTCGGTTGGGAAGCCCGAGTGGCCCTGACCCGGTTTCGGAGGGGTAACTAAATTGCACCAGTACCGACCCCTCATAGTCGTTCGCAAAGCTCCGCAGTCAGCCGTAGCACGCTGTCCTACGTCAGGCCGTACGTAAAAAGACGACCCAGCCTACTCTCTTAGAGATGTCCTATTGCAGTACATACTCAATTTTGAGACGCTTGCCATTTAGCAGCTCTAATGGGTCATATGAGTGAGAGTAAAGTATTTGCATGACACTTTTCCTTAACATATAGATTGTGGACGTGGGAATATCACACCTACATCGTCAGTAACTACCCTACCGGGTGAATTCGGGTAGCGCGGACATACAATTCAAATACGTACAATAAGATTCCGGTGTCAAACCGCTACAGCGCTCTTCCCCGATAAGCTGTATCTGCGGGCACAGCGTCTCACGATGACTACTGCTCTGCCGTTTTCGCTCATAAACACCCATAGTGCAAGCCTGCAACATGGAACATCACTCTACCAGCAATTATAGGGACATGCTAATGGGTACAAATCCAGTCGCCCGCCCGACTCGCCCCTTAAGTTCCCCGAGAGCTATGTTGTATATCGTCCGCTCAAGACTCTGGGTAGTCAAGTCCGCAAAGGACGTCCTTGTCGGCTTATCACTCCTCCCCGCAAGCCCTCTGGCGTGACGTCTGGGTTGTTTTCCTACTCGCCCGTAGTCTAATTTACTCCCCTCGTGGACGTATCCAAAATCTCCATGAGTGGTAGGAGGGTGCCGACTCCAGTTTTGCTCCATTCCGTCCAATGCCCCAAGTCCGACGCCCAAAGTTCTGATTAATAACCCTGCGGGCGAATAGAGCGTACCTTGGGTTGATTATATTCTGAGGTTTGGAGTAACGATCGCTAGGTGCAGATGTTGTTTGAGTACTAAATTGGCTGATGTTTGCAACCGCTTCCACGAAAGAGCAGCTAGCACGCAGAAAATAACGAGCCGGTTCACAGGATGCAGGCTACTCCGCCGCACAACTTCGCCCTTATCAGTCCAATAGACGATTCGCACTCAGAAATCGGTAAGCGGGTGATGGGCCTCGTCGTATTTCGTTCCGTAGTAATCCAATTCAGCGATTAATATTTGTCGTGGAGCGGATCCGATGCCCAACCAGTGTGTTAATGGGCGTCCCAATAGGATGATTGCCAGAATACATCACGGACAAGTTATCTGTAGGGAGCTGAACGTGCTTTAGCGATGAGTCACACGTGTAAAGTGAGCGATTTGGAGGTACTCTCCTTCGAGTGTCGAAACACCAATCCCAGAAAAGATTTACGCTCCCTCAGACGTAATCGGTCTATATGATGAGTGGCACCAAATAGGGCGGAGTAGTCTTAAGGGGCTGCACTCGCCCATAAGGCCTGCTGGGTAATGCCCTCATACCCAATCCACCTCGGTACGGTAGGGTGCTGAGGCATGCTACGGACGAGATTGGTCGTGTGCAAGTAAATCTCAGTTGTGTAGATGTCTACACCTTAATGGTTCTGTAGCGTACAGACACCCCAATTGGTTGAGAATACGAGACTGGCATAAGATTCAGTATACGCAGTGACCTTATAATCACTCGACAAGTGCCTGGTGTCGAGTCTTTTGACAGTAATCCTCAGCGGAGAAGAGGCACTCTATTACTAGATCTGCTTCATTATTTTGGATGGTATTTGGGAACGGCTAGGGCCTAACGATCTTGGTCGCAGAAGATTCTAATTTGGACTCTATCCGGCCGGGCACGAACGATCTCTACAATTGCAGCGTGGCAGCGATCAATCGAGTGATTTCCGGTCGTAAGCGGACCACGCCGTCCTTCTAAGGTTCCCCGACCACGAGGAACCAACGACTGACGTTCCGGAAGCACTAGAGCGCCCGGCAACAGCCGGGCATCCACAACGCGGACATGATGTCTCTCGACTGTGCCTGTAGATCATAGCCGCTCGTCTACCCCCTTACTTTTGGGGCCCAATTGCAACCAATATGGGTCACGTTCTGACCGCCTATAGGGCACGCGGGTGGAATAGGTGGCGAGTGTAAACAGCGCTGATACAATATCACATAAAACGACTTGGTCAGTAACTGCATCGCCTTCCCCTTCTACGTCCTTCGAATCAGGCCCTGGTTCATTTTAGCGACTGGGTACATACAATCCTAATCACGGTCTCGCTTGCTGCACAACAGACGCGTCTAGAGTCCAGAGGTATAGTCGGCGTGGTCTAGACATCAAGAGAAGTGCTGACCAAAGTAAAACCGTGGAAAATTTTATGAACCCTGACTTAGATGAGAAAGAATGGGCCGAATTCTACCCATGTCATACCTTTCTCGCACGACCAGAAATGCCTTGCGGCATGTAAGCGATAAAATGTATTCTCACCATAAGAAAGCCGGTAAGTGTGACCTGGACCGATCCTGTGAAGCACTGAACGTATTTTCTGAGCTCTCTGGACAAGGATACATTCCTGCACGTTTCTAATAATGTAGGTCAAGTTGTGGGGTCTGGCGTGAATTAGTCGTCCCTCCCTTTATACAAAGATTGGTTTTTCGCAACACGGCGTCACCTGCGCACCTCCCGCAGGCCCAAGGTGTTTAATACCAAAGACCCCGACTACTACTGACTCGAGTTCCGACATTCGAGCTGGCTCCTATGACATGAGCCGACGGGCAGGGGAGGGATGCCCGTCCGACCCATTCCGTTGCCGATAGTGCCCAAAGGTGCCGCATCGTTGCTACGATAAGTCCGGCCAAGAAGTATTAGGCTCGCTGATTGTGTGCTATCCCGTTTTTGCGGGGGCCCCTGACCCCGTCAGCCGCATCTGGCTATATCGCTGCTTCAATCACGGTCGGACTTGTCAATAAGCCGTGGCATTAGGAGACTGATACGAAAAGGAGATGACTATTAGGAACCGGCCATAGAATACTTTACAACCCGTGTCGGTTCAGAATAGGTCCCCATTTGTGTAGGACGCGGGGGGCGGCGGGGAACTGTTTTTGTGCACTGAGAATTGCGTGAGTACAACGAAAGGCTTAACAATTCGGCTCAGACCCGGGTAACCCGTGGGTAAGTCTAAAAAGAAGGGCGTCATGGAGGACGTTTATAATTGGTCGACACCAGGCAAGACGTAAAACCTTTTCCGGTACGAGTTTCGCATCGGGCTCGCCCTAAGGACGTGCCCTAGCTCATCTACATTCGGTCTGCAGAGAACTTGAGTGTTAGTAGGTACTGGTTATTATAGAACGCATAGATATAGTTTATGGTTCGCCCTGCGTCTGAACTTAGTTGCGTGAACCGCCTCGTGCTCGGCGGCTAGCTAATGTGCTGGGTGTCTTATATCCCGTTGAAAAGCCATAATCTGTCCAAACGTCGCTCCGATTAATACCCTCTCTGGTACCACGGGCCCGCGCCTCTAGGGTTTTTGCTTATAGCGGGTCGTGTAAGGATACGGGGGCTCCGAGCGGTCTATATTAACGTTAAGGGGACACACATGCCGTCGTCCACTAAGTTGGTAAGGATGATTTTACTGCGTGTCTTTAAGAGTCGCTCCCGCACAACGTACAGTAGGCGCCTTCTAGTTCTGGCGAGCAAAAAAGTGTTGGGAAGTTCCGAGAAGGAGACAAATGAGCCACATAATAATGGATGGTATTACACCGATAGGAGTTGATCGTAGAATTAGTTGGCCTTTAGTGGAATGGAAGCAAACTAGTGTACTCCCAGAGCTTAATAGAGAGGTAATCGGATCTATCCGGTAAAGAACTGTCTACACGAGTGATCCTGTCACAACCCATTATTTGCCTTTGCCGTGTAACCCAACCCCATAATGACCGTATGATACCGCAATGGGCATCTTTGACAATACCGAAAACTGGTTTTGTGAGTCCACTGACTGTATAATAAGGAACCGATGATATTGAGGTGTGCTGCGGCTGGCCTGAACTGAATACTTAAGAATAACGCATTAGTAGTTTCCCCCGGGCCACAAAGATCTGCAAAGATTTTCCACCCGGCCCGTGTCAATTACCATGATGGGGTTCAGAATTCAGCACACGATAACACTTGTCAATTTGGGCCAAGAATTCAAGATAACTAGGATTCCTCATGGTGCCCCTCCGAATGCTGCTTTTGGTAAGGGGTAAAGGACCCTGACGAACTGACCTACAATGTCCTGCGACCAGTCACTAATGTTCGAGTGCTGTTACACGCGCGAAAAGGTGCACCTCCTCGCAGATAGACACGTCCTGGATACCTGGGTTGTTATGGGCAGCTGGGAGGAGCTTTATCAACCTTGCTACCGGGGCAACAAAACGATTCCGCTTTTGCCGCCAATACGTTTTCAGAACCGGTAGTATCCTATTTTTATCACCGCAGGGTGCAAGGCTCTCAAGCCGGCTGTCAAGGGCAGTGCATGCACCACTGGTTAGTAAAGGCTACGTTGCAGAATGGCTGTCAATAGCCAAATGGGGGATTTCAACTTGGTGTTTTTTAATGGAACGCATAATCAATTGGCCCAGTTTCTTACTCGTTCCTATTCCGACGATCAACTTCGAGCCATACTCGATAGATGTGCAATCAACGTTAATCCCAGCATCTTTCATGCTGCCGCAATCTTTATCAAAAGGGGATAAAACTTCACTTACCGAATGCGTCACAAATCTGGGTGCGCTCTCTGGAAGTGCGACTGGGTACAACTTCGCACTAGTATGGAGCGGTTTTGCTGAGCGAGGGGGCAGGCCTTGTGGAAAAGATGGAAAACGGCATGCATGCAGTAGTTTCATTACTATATCCCAAGCGTTAGCTGTAGAGAGCTAGGCCTGACGCGAGGATCTTGGCTCTTATAGTCAAACAACACTATTTAACATCGTCGGTGTAACTACATTTACACAGTCTCTGGACTTTGCGGAAGACGCCCCCGCAACGTCTGCGTGCTGATCTGAGTCCCGAGCGCAATGTGGCCAGATTACAAGCCAAGGACACCTTTGTAGTAGGAGAAGGCCCCATCGAAAGAATAAAACAATGAAGCAGTACCCACACTTTGAAGAGAAGTTTTCCGATACTCCGGCGGCTTATCAACCCCTCTCACGAGTCTTCTTCTTCCCCGGCTACACAAGCCCCTGGTGGTCGTCTTAAGGCGAGGTTCGCAATGTCGTCTGGCGGATGTACGCTCTTGGCTAGTCCTTACAGCTCTTGAGCAACTACGATCTTGGTCAGGCTCTCGAGTCCCCGTGGGGGTGCCCACATTTTGGTGGAGGAAAAGTGAGTGAGTAGGGGGCATAGTGGTCCTAGCCAGTATATTTCTCATAGTTCTCGGTAGGCTCCAGGTCGTTGTATGCCGTACGGGTAGCCAATTCTTGGTCATGCGACCGGTAGCGCAGAAACAAGTCCGTGCATACCTGTCTCTTACCTTGCGAGCATAAGGCTGTCAGATGCGTCATAGTGCTTGCGTGTTATTCTACGGCGGTGTGACTAACCGAAAACTTTAAGAATCTTTCTGACAACTTAGTCCTACTGTGATAGCCTCTTTAAAGTTATAGGTGATGGCCCGGAAATGGTCAGTATATATAGGCGCCGGGTATTCTAGAAGCGCCTTTGCACTTGCTTCAATGCTCATATATGATAACTGATGTCACTCACGTAATGGTCTTTACTATGGTAAAGTCTGGCCATTGATGATGAGGGAGCAAGAACACTAGAATCGCCTCCAAAATGAGGGGATAAACTTTCTGTACTTACACATGGAGCTTCCCTTCCTTATTCTACGGCATCTTACAAGGACATCGTAACTAAAGTGGGATAAAGGGGTCACTACCAAAGACGCGGAGGCGGTAATCTTCCCGCTGCCCACTACATAGATAGTACCCGGAGCCGTCCTGTCGGTAAACCGCGGGTTTGGCTCCAAACCGGTCTCACACTACTGACGTCTAATGGCTGAAAATCAGGACCCTCGCTGTTTAGAGATGACAACCTAGCCGCTTGGAGTGTTGCGCCTAAGGGCAAATCGCGCAACCTCGTGAGGACTTGCGAGTTTAAGTTAGGTGCCTACATTCTTCACATGCAACTCTGTCCACGGAATGTACACCGCATCAAGCGTGTCTCTGCCCGGTTTGCGAACATAAAGACGTTGAGCTCTGGAATGTACCTGTGAAGAGAAATGCCCCATTACAGTTGGTTACTCCTGACATTGGCCATACAACCGCCCTGCGGAATATCGTAATCAGACTACGGCTCCGCCCCAGTGCCCCTGATAGGATCCAAGTATGGTTGACCTAGGCCCTATGCGCTATTCCTCGGCCCTGATACGGCCGATAACCAGATGAAACGGGGGAATGTCTACTTCGACGGACTTGGCAGGCGATGTTATTAGGTGAGTAGCCCAACAGTGCTAGGAAGGCGAGCCATTGCACCCGCTGTAACAACATGTAACCAAAGTATTGATACTTTGCTCCCTAAAACTGGACAGATTACGACCCCTATTCGGCGTGATCCACAACTCAAATACGCCGCGACGACTCCACACCATTTTAATTTGTGGTGCGCGGTCAGTCAGCTTAGCCTAACCCTATTCTAAATCGACCCGTCTCATGACAGCACGTCTGGTACATGGATCGTCCTGGGCGACTAATATGCAGGGGGTAGGGTCATCGACGTACCCGCGTCTTTAAGACATTTGTAGTGTTTATCGATAAGTGTTCATTGACGTAGTGGACTCGGGTATATGGATCTATCCCTAAGTAACTTTTATCCCTGCGCCCGAAACCCCCAAGATACATCAGGGAAACTTGTAACCCAAAGCCGACCTCCCACAATAGTGTGGTTGTGTCCGCAACCGTAGCAAGTTTGCGAGGAAACATTGTTTCTGTATTGATTTCCTTTTAGGTAAGCGGTCTCACACCTATCCTACTCCAGACTCCAGAGGAGCTCACATATTATTTCTCCGCCTTATATGAAAAGCAAATGTACTTCCCTATGTGTTGCCCCCATTACTGGTCAAGCAAAAGGCTCTACTACTTCCAGTGGCGGCACCCGGCAAGGCAGCGCATTTCACCGGGGAGAACCAAATACAGGTATTGGACGTACATAACCCTAGGGAGATTTCCCTTCCTTCAACCGATACCCTAGGTCTGAGTCACGAAACCCTCGGCAGCAAGATATTCCAAAGCACCACTCCTACACGTGCCAGAGGTTCAATTTTGGAGAACTTGATGCCCACCCAGTAGGTCGATACGTGAACGCGACCACTTGTTCAGAATGCACTTCATCGTGTCGGACTACGCCAACTGAACCGAAATACCAGGGGAGCCCACTCCCGGGCGAGTGACTTCAACTGGTCCAGCCGAAGGTGAGGCTGCGATATGCAAGACCGAGACGTGTACGACACGCGATTGGAAGTTGAGATGCATCTTCACTGGCCGGAGTTAGCTCGGCGCTCCTCATGGCGCAGGGTCGACCACAAGAAACTGCATAGTGCAGCATTTGCAGCGCGTCATTTGTATCGAACTCCTACACATTTTTATGATACAGACATGATTGTTTCGGACCGGAAAGGGATAGGTCACCCTAAGACCACGAGGTGCTCAACCAAGCTCTCAGCTCGGGGCTTGGAGCCAGCTTTATAAGTCGATTGGGAGCCGGAAATCTCGATTTTACAGGGCCACAAGTATAGCCGATCACGGGCAAAATCCTTGATTTAAGCTCGCTCGCACCTTAGAGGTTAGGTAGTGGATCACGGCGCTAAAAGGATGTTAAATCCAGCGTATGTGTCCATAGTTCCGTTCCCTAGTATGACTTGACGCACATCATGAGAGCACCAGCAACAGTCTTTCTAGTGGCCAGTTTTAATGCCCTCGGGGTGTCAGCTCCTGCTGGAAACACATTCGTCAGCGGGGTCCAACCTATAGTTGATGAAAATTCCATGTCTTCCGTATGATTACCGGAGAACGTGACTGTCACAAACCAAGTTCTCCCATGGCACTCTGGCTAATTGTTGGGCAATTACCCACGTCTAGACCTTAAAACCTCTGCCCATCAGATTGATATGACGGGCCCATTCATCGTTCCGTTGCGCCTCGCTGCCAGCCTCGTTTGACCGGCTACCAATCTGGGCAGTATGTTCTGTCTGTAAACATACTGCAGGGGTACGTTGACATTAAGTGCCAATGCAGTCCATGATATACATCCAATTGCTCGGGAGACCGCGTGTTAGCGACGAAGTGCGTACGAGTGGTCCCATTATCTTAGGACATCAGTACATGTGCGAACGTCGACTAACCCAATGCCTATCGAGTTTCCAGTTAAAACCGGCCCCCAATAACCTCGATCCCTGGGCCTCTGTCAGCCAAAAGATTGAACCGCAGTCTGTGAATGACACGGGAGATCCAGGGTGTATACCATGTAAACATTCAATACCAGACAGCCGGCCAGGACGAAACTGCAGTAGACGTCAAAATACGCCCCGCGACCCGCCCTTGATACCAACAACAAGCATGGGCCTTGACGTGATCGACGGAAAAATGGTACGTGACGCGCACGATGAAGGTTTAGCTATCAGATATCCCTTGCGGTTCGTGCGCAGCTTCATTTCTAATGAAGCTCACCAAAGGTAGCTAACTAAACAGGGGGGCTCACCTCAATCGAAAGGAGTAAGCGAGTCGCCTTAGAGGGCCCGCGCTATCCACATTGTCAGAAGACATCGGAAGATAGTCATAGGGCTGGTCTTATCTGAGCAAGCTCCGGACACGTACGCGCAGCGTGTATCGGCCCCCCGAGGCGTATCGCATGCCGTCCTGTACCAGAACGAGAGAAGCATGCCTGTTCGAGGATAGATCTACAAACTAGAGCGCATGCAAGGCGGCACTGCACTGCACTCAAAAAGAACAGGAAAAACGCGAGCCTCGCGAGTGGGCCTAGTTGACCAGGATTAGTCCAATTTCTGCTCGGACCAATTTTATTGTTTTACATGATAATCACCCCTCCTAGGGGGTGTCGGCTGACTGACATTAGAATGAGCTGAGTGTGTGGTTGCCTACCTCCCCAATAGTCCGTCTAATAAAGAAGCAGTCCGTTATGGATCAGAACCACTAGTTGTCACGCTCTGCCTTGAAGTAGCTGTCTCAGTGGGGTCGTTGGAATACTTAATAGAATACTGGTTACAGGTGCAATCTCTCCTACCGAAGTCGGGGATGTTTGATGTGTACTATAACCCGTAAAAGCGTGCCTAGGCAGGGAGATCTTTAACGTTATTCCACACGCTTGGACTTTAAGATCTTGGAGATGTTAGTACTTAGTCGCGGACCGTACATGGGGTCCGCGACTGATCAGCCTCACTCTTCCCGGAAGAAATTCTCGCCCGTAAACGCCGTGCCCTCGTTAAGGCTAGACGTATCTCTGGGCGTTCCAACCGGCTGATCAGCACACCCGTTGCCTCACGTAAGGGGAGAAAACCATTGCGTGTAAGTGTTCGCCACAATTTTTACAACGAGGCCTAGTCGGTGAGTTCTGTCTTGTACACAAAAAATAATGGCCCCGGTAATGATTCACGAAAACTTCCAGACCAGCGAATAAATAACATCGCTTCCTCAATGCCGGTCTGGTCAGAAGGATACTGGGAGATTCCCCCTGCAGGTAATGAGGTTGCTAGAGGCTAACGGTGAAGCAAAGTGATCGGCCGATAGTGAGACGTTTCTCTCGGGAATCCCCCGACACACTGTCTCCTACCCAACTGGTCCAAGAACGCAATAAGTTAAGATCGCTGTCTGCCCCAATTTTCCTGGCAAGGACACATTGCAGGGGTAAACTGCTAGGGTCCATTATGAGAGGAGGTAATAAATTCGACACGAGCGTACTTACGAGGACCGCGCCACTGGAAGGCGCCCCTAGTGTGCGCAACATCCACGGGCCGGTACCATTCAACGAACGTCTCCCTGCATGCACAAAACGGGTAAGTTTACCCGTCAAGTGCCATGTCGGGACTAGCAGTCATATATATTTTTAGGAGAAACACCCGCCCTCACTACCCGGCACGCTCTGATGAGCATAAAGCCAGGACACGCACTAGATTGTGTCCGGTGAGTAGATGAAATTCTTACCAATCGGCTCCGACCGACGGCAACTCAAGTATGTTCCCACGCTCACGTAATGTTCACATCTCAGAGATGACGCCCACGAGCATGAGAGGAAAAGTAAGCTAGAATTACCTGGCGGACGCGGGGCACTGAGAGGCTAGGAGGTACCGATTCCTCCATTACGTTATGGTAGGTGCACTATAATTCCCGCGAGGTCATCTATGAGACTAGATCTTTCGCGATCTTCATACGTACACCTTCATAACTAATACGGAGACGGCCCTGCTTGAGGTGTGTGCGTACATATGGCAACTCACCCCCGAATGCGTATTCCCGGACTAGTATCTCATACCTGTCGTCCCACGCCACATTCGACTTGTGTCACCAGTGGAACAGGATTCGCTCGTACGTCGGGATTCATCTGGTTGTGGTTAACGACTGCGTCTCGGTACCAGCCTGAATGTGTTTAGTCCATTCCATACGTAAATGTTCCCCCGTTTCAACACCTGTGAGCAATTAGATTTAACCTCGTGTTAATACTCGCGAGAGTGTGATATCCAAATCGGACATTAGGATATATGGTTGCGACCAAGCTTCCGGAGGGCGTCCTCACCACGCGGTTCTAAGTCAAAATCCCCTACCTGCAGACAGAAATGGGCGGCGGGCCAGGCTTGCTGTCACGGAACCGTGGTAGTGCTGGGTCTTATCTTGTCGGACTAAAGCATGATATGTTGCCTAGTATGTGACCAAGTCAGCTTTAGTAGTACGGTCGCCTATGCTATCCAGGGCGAGTACATCGGCACTGACATTATTTGGTACCGTACAGTCGAATTCTCGACAGCTTGTCTTTCTTGGTGATATTTCAACCAACAGCCGCACACGCTTATGATTAATATTGTCGAAGCACCCGAATAAAATTAGCTGGGCGTTGAGACCAAGCCGTCCGAGTAAATAAGTCAGTGCACGACACTGCAGAGGGTGGAGGCATTTCAATCCCTACTTCCCCGGTGTTCGCTGAGCATCTTTACCTTGAGCACAGACGGCTTCCCAACTTTAATCTCGGGAGCCCGCCTAGAGTGTATTCCCCTATTGTTTTCGGAAGGCCAATGGCAACGCTATCACACCACAAGCTAACGGGAATATCAGCTTGGACAGTGGTAACGGCGTTGCTTTGATCTATGGAAAGGAAAGTGTTTTACCTCGACTTCTAGTATATTAGGTGAAGTCCCAACCCCTATACGGTTGCTGATAACCGACGCATGTCAGTGTAGTCAATAATTCGGCAGTCCACTTGCGCGAACAGCCAGTATGGCCTCGTTACAAGGCAGTGAGTGTAAGGATTGCTGCAATATGGATCTCCTGAACAGCCTTCCTTACAGAAACATGATCTCGACAAGGGCGACGTCTGCCGTGGCAAACCAGACG$ACCCCCAGCTATCAGGGGGGTGATGATACTAGATTAATTGCGCTCTCTGGGTATTCAGACGAGTCATTACGGGTAATGTCTACCGCTCTTTCCCCGTAACATAGGCGCTGACCGCTAATCGATCTTGATATCCTACTGTGCAGATACAGCCGTGCAGATTGTGTCACCGACGCGGTGGATACACTGACGCGCACACCCACAACGTCGAAATCTGTTCTGTACCCTGGTTTGCAGTCCCAGGTCTACACAGGAAGCACCGGGATCGGTGGTCAAGGCGTACTCGCTGCGTGTCGACGTGACGGGTAGGGATGAACTGAGGCGGATTGAGTCAACAACAAGAGGACGGTAAGCCGTCAGATCTATTTTTAAGGGTGGCCAGCACCCTACGGCGTGCCATAAGAGACCCATACAGTCTCTGAACAAGATGGGCATCCCACACTGTTATCGACACTTCTTGAACTATGGCTTCAGATGGATCTCAACGCTCTGCGGACGCCGAGCGAAAGATGCGAAGGTATTTACAAAGCCTATGCCATTCCAGAGCTTCGCAAGACGCACCCTCGGAGTGATAAGACATCAATTCAATTAGATACAGGGAGGTATCCCGAATACTTACCTTTAAGTTACATCTTATGCATGTAAGCTAAATATGTGTAGCGCCGCCCAAAGCGTATCGTAGTGTGCGCTCGTGCCGAGCAAAGGGGTTCGACGCCCTGAAGCGCCGATCGGGCGCTTGTGCCTCTGAGGCTAGCGGCTTGGTGCTTTATTTGCTGCTATGCTACGGCTCGTCTAAAGCCAGGGTTGACAGAGCACCCGAATTAAGCGGTTTCCCAGATGAAGCGAGCAGCGTTAGTTCCCCCGGAAGTGTCTCTCACGGTCTTAGTGACCCGACCCCGGCCAGCTATGATGTTTTTGACTACTCATCACATTTTGGGCTAGAATCTGCCTAAGCACCAATGCTATCCGGGCGCATGGTTCCTATTATATGAATACGACGCGGCTTTACGGCGTGTCTTGTTAAGCTTAGTTTTCATGGCGAGTAGGGGTCGCCGACGATACGAAGTATCATGTCGACGGATCCAAACGTAAGCAATGAACCAGCGAGGTTAGAGGCATAATTAGGTACCTACTTGCATTCTTGTCTAGAAAATCTGTGAAATAGCACTTGAGAGACAACCCCGGAGTCCCCTATTGCCAAATTATGTGCGTGTACGGCATTATACCCCTGATATTCGCCCCTCAACATGCGCGGAAAATACGGCATGTAGTGTCACATTAACTCCCGCCGGATCGGGTACACTTCCTAGTGCTGACGGCTATGCTGCTATCTCCAGATAATCCGACCAGCCACAAACGATCCATATAACCCAGCTCTTAGGTGGGGCTATATACTTCAACCCGAGTGTCAAATGTAGCGTCGGAAAGAAGGAGTCCTCTTGATCTGTCAAACTCTCCAGGACCAAATTATAGCTCACCGTTGTGGGTTTCCGGAGGCACTAAGGGGCAACAACAAGACGTATTATGAGGGGGCATTACCCTGAGGGTAGGTCTCAGGGCCCCCCGCAGAGCTGATGCGGTGCTATCGGAAACTCTTGCTGGAAGAAGAAATTAACATCACATTTGCGTCCAACGACATGCCATGTCGATCTATGAGGACCCCCAAGCGCAAGCTGTAAAGGTATCGAACTTTAAAGTAGCGGCGTAAAGAGCAATACGCGAGCGAAACAATGTTCATCTGTTAAACCCGGTCACTAATAACGCAGGCTTCCGAAGGACACGAGAACGTGGACCGTTACTTATGATGGTATTGTCCATAGTACAATACAGCTAGTGTGCAGGCAGGAAGAACGATGCCTGAACCGCTTAGCAGGTGGATAGACGTGGTACCGGTAAATTGGCGTGATCGGAAGAGCAAGTCGTCCGTACAGGCTACAGCAAGTGTGTGTTGCCCTCCACAAGAACTCCGGCGTCTGCGGTCTCCAGTGATTCGACTGTATGGCTCGGATGGGGGTAGCCCTACAGGGGCTCGTGTCTAGCCCGAAGTGTGCACAAAACGGATCTGGTACATCACACGACACTGTGACCCAGGAGGTAGACATACGTCTTCCACTCACGAGTCGCTTTTTACAAGCAAGTCATCGACACGCAATTTCAGTGACTACTATCGATGAATTAGACTAGGGTGTACATTACGGACAATTCAGGTCATCTAGCAGCAAAAAGTTTGTATCCGACGAGCAGCCGAAACCGCCTGAAAATGGAGATCCGGAGTTTGGGTTTCGACTTGCTTCTCCTCGACTGGGGACCTACAGCTGCCACATATGGAAGAGGTGACACCTTTACAGAAAAGCCATGTCAGTAGTACGAACAGTAACTGTTGCTAATAAACTTTACGGCAATTATTTAGTGGGCAAACGCTTGGCGTGGTCTCAACCGAACTAACAAAAGGTCGTAGTTGACATGCTAGACCGCGTCCGAGTACGAGGGGGTACTAGGGTGGCGTATTGTATAGTAATGGGCCTTCTTTTTTAACCGCCCTGTCTGTGGTTATTAGTCCTCACTTCATCTGGGAGGAAATCCCCGAAAGCGGACTCGCAACGGAAGCTTACAATGCGATTGAGCAGCATTAGACCCACACCAACTAGCTATTATATTAGATACCGCAAGCGCGCTAAAATGTGAAAGAGTATGCATCTGCCTCGGGGCTCCAGATCCGTAATACAGATCCGCCCCCGGCCACTCGCCTAAAAACATGCTGACTCTCCACTTCTGCCAGAATTGTCGAGCTGATTAACCCCTATACCAACTAAGTTTATATTCGTACGTCATAAGAAGTATGCACTGGTCCCGGTCGCAAAAATAAACGGAGAGGAAGCTAGGTGGGACTGGTGCGAATTCCCTTAACAAAACAATATCTGCCTGCAGTACCTATCGAAAACCATGGTGAGCTATTTACACACGCGCATCATTGACGTGTCCGGTTCTAGGAGGCCAATGCACGAGGTAAAGGAACCTCGGGAAGCTCAAAATTCTTTTTTACTCACTAACAAAAACGTTAGATTGTAGCACGTCGAAAAAACTATTTCAAGTGTATTCGCTTTTAACGGCTTACATACGTTCTCTGACGACGACTTCTCGGTTTACCTCAATGAATTTGCTCCCGATGCCTTTGGCAGGAGCCGACTATCAGGATGCGCGCAGACGTCGGGTACTAGCTCCCAGCAAAATGCGTTAATCGGAGACACACTGTGTCCAGTGGCCACTTTTCTTTATCCTGCTGTACAAGGAATAACCA'
file = 'patterns.txt'
patterns = patterns_input(file)
count, symbols_dict = count_of_symbol_matrix(bw)
first_occurrence = first_occurrence_matrix(bw)
l = better_bw_matching_all_patterns(first_occurrence, bw, patterns[1], count, symbols_dict)
for i in l:
	print(i, end = ' ')
