
import math
import networkx as nx
import copy
from itertools import groupby
from operator import itemgetter


#from stackoverflow
def my_round(n, ndigits):
    part = n * 10 ** ndigits
    delta = part - int(part)
    # always round "away from 0"
    if delta >= 0.5 or -0.5 < delta <= 0:
        part = math.ceil(part)
    else:
        part = math.floor(part)
    return part / (10 ** ndigits)

def hmm_parameter_estimation(sequence, alphabet, path, states):
	#form transition matrix
	transition_matrix = {}
	total = {}
	count = {}
	for state in states:
		transition_matrix[state] = {}
		total[state] = 0
		count[state] = 0
	for state in transition_matrix:
		for _state in states:
			transition_matrix[state][_state] = 0

	#form emission matrix
	emission_matrix = {}
	for state in states:
		emission_matrix[state] = {}
	for state in emission_matrix:
		for x in alphabet:
			emission_matrix[state][x] = 0

	#calculate transition probabilties
	for i in range(len(path) - 1):
		if path[i] == 'B':
			total['B'] += 1
		elif path[i] == 'A':
			total['A'] += 1
		else:
			total['C'] += 1
		transition_matrix[path[i]][path[i+1]] += 1

	for state in transition_matrix:
		for value in transition_matrix[state]:
			if total[state] == 0:
				transition_matrix[state][value] = my_round(1/len(states), 3)
			else:
				transition_matrix[state][value] = my_round(transition_matrix[state][value]/total[state], 3)

	#calculate emission probabilities
	for state in states:
		count[state] = path.count(state)
	
	for i in range(len(sequence)):
		emission_matrix[path[i]][sequence[i]] += 1
	for state in transition_matrix:
		for x in alphabet:
			if count[state] == 0:
				emission_matrix[state][x] = my_round(1/len(alphabet), 3)
			else:
				emission_matrix[state][x] = my_round(emission_matrix[state][x]/count[state], 3)
	return transition_matrix, emission_matrix

def input_alignment(file):
	strings = []
	with open(file) as f:
		for line in f:
			l = line.strip()
			strings.append(l)
	return strings

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def profile_hmm(theta, strings, alphabet):
	#check which columns are to be removed
	ratios = {}
	total_number_of_columns = len(strings[0])
	for i in range(total_number_of_columns):
		ratios[i] = 0
	number_of_strings = len(strings)
	for string in strings:
		for i in range(len(string)):
			if string[i] == '-':
				ratios[i] += 1
	removed_columns = []
	for column_number in ratios:
		ratios[column_number] /= number_of_strings
		if ratios[column_number] >= theta:
			removed_columns.append(column_number)

	#make hmm graph
	G = nx.DiGraph()
	#number of match states = total number of columns - length of removed_columns (node names - m1, m2, ..., mn)
	#number of insertion states = number of match states + 1 (i1, i2, ..., in)
	#number of deletion states = number of match states (d1, d2, ..., dn)
	number_of_match_states = total_number_of_columns - len(removed_columns)
	#add nodes
	nodes = []
	nodes_match = []
	nodes_insert = []
	nodes_delete = []
	for i in range(number_of_match_states):
		nodes_match.append('M' + str(i+1))
		nodes_insert.append('I' + str(i))
		nodes_delete.append('D' + str(i+1))
	nodes_insert.append('I' + str(number_of_match_states))
	nodes = ['S', 'E'] + nodes_match + nodes_insert + nodes_delete
	G.add_nodes_from(nodes)
	#add edges
	G.add_edge('S', 'M1')
	G.add_edge('S', 'I0')
	G.add_edge('S', 'D1')
	G.add_edge('M' + str(number_of_match_states), 'E')
	G.add_edge('I' + str(number_of_match_states), 'E')
	G.add_edge('D' + str(number_of_match_states), 'E')

	for node in nodes:
		for _node in nodes:
			if node[0] == 'M' and _node[0] == 'M' and (int(node[1]) < int(_node[1])):
				G.add_edge(node, _node)
			elif node[0] == 'I' and (node == _node):
				G.add_edge(node, _node)
			elif node[0] == 'D' and _node[0] == 'D' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)
			elif node[0] == 'D' and _node[0] == 'M' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)
			elif node[0] == 'M' and _node[0] == 'D' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)
			elif node[0] == 'I' and _node[0] == 'D' and ((int(node[1]) == int(_node[1]) - 1) or (int(node[1]) == int(_node[1]))):
				G.add_edge(node, _node)
			elif node[0] == 'I' and _node[0] == 'M' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)
			elif node[0] == 'M' and _node[0] == 'I' and (int(node[1]) == int(_node[1])):
				G.add_edge(node, _node)

	states = ['S', 'I0']
	for i in range(len(nodes_match)):
		states.append(nodes_match[i])
		states.append(nodes_delete[i])
		states.append(nodes_insert[i+1])
	states.append('E')



	#emission matrix
	emission_matrix = {}
	for state in states:
		emission_matrix[state] = {}
	for state in emission_matrix:
		for x in alphabet:
			emission_matrix[state][x] = 0

	columns = [i for i in range(total_number_of_columns) if i not in removed_columns]
	count = {}
	for state in states:
		count[state] = 0
	#emission values for match states
	for x, y in zip(columns, nodes_match):
		for string in strings:
			if string[x] in alphabet:
				emission_matrix[y][string[x]] += 1
				count[y] += 1
		for item in emission_matrix[y]:
			emission_matrix[y][item] = my_round(emission_matrix[y][item]/count[y], 3)
			if emission_matrix[y][item] == 0.0:
				emission_matrix[y][item] = int(emission_matrix[y][item])

	#emission values for insert states
	r = ranges(removed_columns)
	merge = []
	for _range in r:
		merge.append([i for i in range(_range[0], _range[1] + 1)])

	merge.sort()

	#convert merge to insert states
	merge_to_insert = []
	m = copy.deepcopy(merge)
	i = 0
	j = 0
	c = copy.deepcopy(columns)
	c_ = copy.deepcopy(columns)
	while i < total_number_of_columns:
		if i in c:
			i += 1
			j += 1
		else:
			a = m.pop(0)
			c.insert(i, a)
			c_.insert(i, nodes_insert[j])
			i += len(a)

	for x in range(len(c)):
		if c[x] in merge:
			merge_to_insert.append(c_[x])

	count = {}
	for state in states:
		count[state] = 0

	for x, y in zip(merge, merge_to_insert):
		#print(x, y)
		for position in x:
			for string in strings:
				if string[position] in alphabet:
					emission_matrix[y][string[position]] += 1
					count[y] += 1
					#print(emission_matrix[y][string[position]], y, string[position])
		for item in emission_matrix[y]:
			emission_matrix[y][item] = my_round(emission_matrix[y][item]/count[y], 3)
			if emission_matrix[y][item] == 0.0:
				emission_matrix[y][item] = int(emission_matrix[y][item])

	#transition matrix
	transition_matrix = {}
	for state in states:
		transition_matrix[state] = {}
	for state in transition_matrix:
		for _state in states:
			transition_matrix[state][_state] = 0
	nodes_match.sort()
	nodes_delete.sort()
	strings.sort()

	#column to state for each string
	column_to_state = [[] for i in range(len(strings))]
	for k in range(len(strings)):
		column_to_state[k] = ['' for i in range(len(strings[k]))]
		j = 0
		for i in range(len(strings[k])):
			if i not in removed_columns:
				if strings[k][i] in alphabet:
					column_to_state[k][i] = nodes_match[j]
				else:
					column_to_state[k][i] = nodes_delete[j]
				j += 1

	for i in range(len(strings)):
		for c, insert_state in zip(merge, merge_to_insert):
			for column_number in c:
				if strings[i][column_number] in alphabet:
					column_to_state[i][column_number] = insert_state
				elif strings[i][column_number] == '-':
					column_to_state[i][column_number] = '-'
		column_to_state[i] = ['S'] + column_to_state[i] + ['E']
		column_to_state[i] = list(filter(lambda a: a != '-', column_to_state[i]))

	#transition values values
	count = {}
	for state in states:
		count[state] = 0
	for conversion in column_to_state:
		for state in conversion:
			count[state] += 1
		for i in range(len(conversion) - 1):
			#print(conversion[i], conversion[i+1])
			transition_matrix[conversion[i]][conversion[i+1]] += 1
	for state in transition_matrix:
		for value in transition_matrix[state]:
			if count[state] != 0:
				transition_matrix[state][value] = my_round(transition_matrix[state][value]/count[state], 3)
				if transition_matrix[state][value] == 0.0:
					transition_matrix[state][value] = int(transition_matrix[state][value])
	#print(merge, merge_to_insert)

	return states, transition_matrix, emission_matrix


file = 'alignment.txt'
strings = input_alignment(file)
alphabet = 'A B C D E'.split(' ')
theta = 0.311
states, transition_matrix, emission_matrix = profile_hmm(theta, strings, alphabet)

#HOW DO I FORMAT THIS UGHHHHHHUHHSDKHDKJHKSHKH K SO IRRITATTINGGGGGGGJSGDJHGJGHJbjhn!!!!
value = ''
value += '\t' + '\t'.join(states) + '\n'
for state in states:
	l = state + '\t'
	l += '\t'.join([str(transition_matrix[state][state_]) for state_ in states])
	value += l + '\n'
value += '--------\n'
value += '\t' + '\t'.join(alphabet)
for state in states:
	l = '\n' + state + '\t'
	l += '\t'.join([str(emission_matrix[state][x]) for x in alphabet])
	value += l
print(value)
'''
    for i,s in enumerate(state_name) :
         l = s + '\t'
         l += '\t'.join([format("%.3g" % __transition[i,j]) if __transition[i,j] != 1 else '1.0' for j in range(len(state_name))])
         ret += l+ '\n'
    ret += '--------\n'
    ret += '\t' + '\t'.join(alphabet)
    for i,s in enumerate(state_name) :
         l = '\n'+ s + '\t'
         l += '\t'.join([format("%.3g" % __emission[i,j]) if __emission[i,j] != 1 else '1.0' for j in range(len(alphabet))])
         ret += l





for state in states:
	print('\t' + state, end = '')
print('')
for state in transition_matrix:
	print(state + '\t', end = '')
	for _state in transition_matrix[state]:
		print(transition_matrix[state][_state], '\t', end = '')
	print('')

print('--------')

for x in alphabet:
	print('\t' + x, end = '')
print('')
for state in emission_matrix:
	print(state + '\t', end = '')
	for x in emission_matrix[state]:
		print(emission_matrix[state][x], '\t', end = '')
	print('')
'''

'''
sequence = 'zxzyyxyxyxyyyyxzzzzyyzzxzzxxzyyzzyzzxxyxzyxyyyyzxzzyzzzyyxzyxxyxyzxyxzzyyyxyyzzyyzxzxzyzyxyxyzyxzyzz'
alphabet = 'x y z'.split(' ')
path = 'CBCBABCABBBBAACBAACCCAAABABBCAACBACABAABCCAABABABCBBCBBBBCCAABBACBCCCABCBBCCCCBCACACAAAACACCACACACCB'
states = 'A B C'.split(' ')
transition_matrix, emission_matrix = hmm_parameter_estimation(sequence, alphabet, path, states)

for state in states:
	print('\t' + state, end = '')
print('')
for state in transition_matrix:
	print(state + '\t', end = '')
	for _state in transition_matrix[state]:
		print(transition_matrix[state][_state], '\t', end = '')
	print('')

print('--------')

for x in alphabet:
	print('\t' + x, end = '')
print('')
for state in emission_matrix:
	print(state + '\t', end = '')
	for x in emission_matrix[state]:
		print(emission_matrix[state][x], '\t', end = '')
	print('')
'''


