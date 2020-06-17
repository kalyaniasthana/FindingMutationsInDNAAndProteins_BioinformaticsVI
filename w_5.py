
import math
import networkx as nx
import copy
from itertools import groupby
from operator import itemgetter
import sys
import numpy as np
from w_4 import *


def max_with_index(my_list):
	index, value =  max(enumerate(my_list), key=itemgetter(1))
	return index, value

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

def hmm_hidden_path(sequence, emission_matrix, transition_matrix, states, alphabet):
	#node scores and backtrac matrix
    scores = {} 
    backtrack = {} 
    for state in states:
        scores[state] = [0]*len(sequence)
        backtrack[state] = [0]*len(sequence)

   
    for state in states:
        scores[state][0] = 1*emission_matrix[state][sequence[0]]


    for i in range(1, len(sequence)):
    	for state in states:
    		weights = []
    		x = []
    		for state_ in states:
    			weights.append(scores[state_][i-1]*transition_matrix[state_][state]*emission_matrix[state][sequence[i]])
    			x.append((scores[state_][i-1]*transition_matrix[state_][state]*emission_matrix[state][sequence[i]], state_))
    		scores[state][i] = max(weights)
    		for j in x:
    			if j[0] == scores[state][i]:
    				backtrack[state][i] = j[1]

    #print(scores, backtrack)


    #find max value at last position
    weights = []
    x = []
    back = []
    for state in scores:
        weights.append(scores[state][len(sequence)-1])
    for state in scores:
        if scores[state][len(sequence)-1] == max(weights):
        	current_node = state
        	back.append(state)

   
    i = len(sequence)-1    
    while i > 0:
        for state in states:
            if backtrack[current_node][i] == state:
                current_node = state
                back.append(state)
                i -= 1
                break
            
    return ''.join(back[::-1])

def viterbi_learning(iterations, sequence, alphabet, states, transition_matrix, emission_matrix):

	for i in range(iterations):
		path = hmm_hidden_path(sequence, emission_matrix, transition_matrix, states, alphabet)
		#print(path)
		transition_matrix, emission_matrix = hmm_parameter_estimation(sequence, alphabet, path, states)
		#print(transition_matrix, emission_matrix)
	return transition_matrix, emission_matrix


def soft_decoding(sequence, transition_matrix, emission_matrix, states, alphabet):
	m, n = len(sequence), len(states)
	forward = np.empty(shape = (m, n), dtype = float)
	backward = np.empty(shape = (m, n), dtype = float)
	cp = np.empty(shape = (m, n), dtype = float)
	#print(transition_matrix, emission_matrix)
	#print(states, alphabet)

	#calculating forward values
	for i in range(len(states)):
		forward[0, i] = emission_matrix[states[i]][sequence[0]]/len(states)
	for i in range(1, len(sequence)):
		for j in range(len(states)):
			emission_value = emission_matrix[states[j]][sequence[i]]
			s = 0
			for k in range(len(states)):
				s += transition_matrix[states[k]][states[j]]*forward[i-1, k]
			#print(s)
			forward[i, j] = emission_value*s

	#calculate sink value
	sink = 0
	for value in forward[len(sequence) - 1]:
		sink += value

	#calculate backward values
	for i in range(len(states)):
		backward[len(sequence) - 1, i] = 1
	for i in range(1, len(sequence)):
		for j in range(len(states)):
			bk = 0
			for k in range(len(states)):
				weight  = emission_matrix[states[k]][sequence[-i]]*transition_matrix[states[j]][states[k]]
				bk += backward[-i, k]*weight
			backward[-i-1, j] = bk

	#calculating conditional probabilities
	for i in range(len(sequence)):
		for j in range(len(states)):
			cp[i, j] = my_round((forward[i, j]*backward[i, j])/sink, 4)
	#print(forward, '\n', backward, '\n', sink)

	return cp
'''
sequence = 'yyzxxxzxxz'
file = 'transition.txt'
states = 'A B'.split(' ')
alphabet = 'x y z'.split(' ')
transition_matrix = input_transition(file, states)
file = 'emission.txt'
emission_matrix = input_emission(file, states, alphabet)
cp = soft_decoding(sequence, transition_matrix, emission_matrix, states, alphabet)
for state in states:
	print(state, end = '\t')
print('')
for l in cp:
	for i in l:
		print(i, end = '\t')
	print(' ')
'''

'''
sequence = 'yxxzxxxyxyzxxxxzzxxxyxyzxzzxxxzxyzxxzyyyxzxzxzzyyxyzzzyzyxyxzxxzyyxyyyyzyxxyzyyzyxzyyyyxxzxzzzyyyxzx'
file = 'transition.txt'
states = 'A B'.split(' ')
alphabet = 'x y z'.split(' ')
transition_matrix = input_transition(file, states)
file = 'emission.txt'
emission_matrix = input_emission(file, states, alphabet)
iterations = 100
transition_matrix, emission_matrix = viterbi_learning(iterations, sequence, alphabet, states, transition_matrix, emission_matrix)
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

def profile_hmm(theta, strings, alphabet, pseudocount = 0):
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
			if node[0] == 'M' and _node[0] == 'M' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)

			elif node[0] == 'M' and _node[0] == 'I' and (int(node[1]) == int(_node[1])):
				G.add_edge(node, _node)

			elif node[0] == 'M' and _node[0] == 'D' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)

			elif node[0] == 'I' and (node == _node):
				G.add_edge(node, _node)

			elif node[0] == 'I' and _node[0] == 'D' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)

			elif node[0] == 'I' and _node[0] == 'M' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)
			
			elif node[0] == 'D' and _node[0] == 'D' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)

			elif node[0] == 'D' and _node[0] == 'M' and (int(node[1]) == int(_node[1]) - 1):
				G.add_edge(node, _node)

			elif node[0] == 'D' and _node[0] == 'I' and (int(node[1]) == int(_node[1])):
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
			emission_matrix[y][item] = emission_matrix[y][item]/count[y]
			#if emission_matrix[y][item] == 0.0:
			#	emission_matrix[y][item] = int(emission_matrix[y][item])

	#emission values for insert states
	r = ranges(removed_columns)
	merge = []
	for _range in r:
		merge.append([i for i in range(_range[0], _range[1] + 1)])

	merge.sort()

	#print(merge)

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
			emission_matrix[y][item] = emission_matrix[y][item]/count[y]
			#if emission_matrix[y][item] == 0.0:
			#	emission_matrix[y][item] = int(emission_matrix[y][item])

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
				transition_matrix[state][value] = transition_matrix[state][value]/count[state]
				#if transition_matrix[state][value] == 0.0:
				#	transition_matrix[state][value] = int(transition_matrix[state][value])
	#print(merge, merge_to_insert)

	#pseudocount
	#print(states)
	if pseudocount > 0:
		#modifying emission_matrix
		for state in emission_matrix:
			s = 0
			if state[0] == 'M' or state[0] == 'I':
				for x in alphabet:
					emission_matrix[state][x] += pseudocount
					s += emission_matrix[state][x]
			if s != 0:
				for x in alphabet:
					emission_matrix[state][x] = my_round(emission_matrix[state][x]/s, 3)

		#modifying transition_matrix
		from_S = ['I0', 'M1', 'D1']
		s = 0
		for state in from_S:
			transition_matrix['S'][state] += pseudocount
			s += transition_matrix['S'][state]
		for state in from_S:
			transition_matrix['S'][state] = my_round(transition_matrix['S'][state]/s, 3)

		to_E = ['M' + str(number_of_match_states), 'I' + str(number_of_match_states), 'D' + str(number_of_match_states)]
		s = {'M' + str(number_of_match_states) : 0, 'I' + str(number_of_match_states) : 0, 'D' + str(number_of_match_states) : 0}
		for state in to_E:
			if state == 'M' + str(number_of_match_states):
				transition_matrix[state]['E'] += pseudocount
				s['M' + str(number_of_match_states)] += transition_matrix[state]['E']
				transition_matrix[state]['I' + str(number_of_match_states)] += pseudocount
				s['M' + str(number_of_match_states)] += transition_matrix[state]['I' + str(number_of_match_states)]
			elif state == 'I' + str(number_of_match_states):
				transition_matrix['I' + str(number_of_match_states)]['E'] += pseudocount
				s['I' + str(number_of_match_states)] += transition_matrix['I' + str(number_of_match_states)]['E']
				transition_matrix['I' + str(number_of_match_states)]['I' + str(number_of_match_states)] += pseudocount
				s['I' + str(number_of_match_states)] += transition_matrix['I' + str(number_of_match_states)]['I' + str(number_of_match_states)]
			elif state == 'D':
				transition_matrix['D' + str(number_of_match_states)]['E'] += pseudocount
				s['D' + str(number_of_match_states)] += transition_matrix['D' + str(number_of_match_states)]['E']
				transition_matrix['D' + str(number_of_match_states)]['I' + str(number_of_match_states)] += pseudocount
				s['D' + str(number_of_match_states)] += transition_matrix['D' + str(number_of_match_states)]['I' + str(number_of_match_states)]
		for state in to_E:
			for item in transition_matrix[state]:
				if s[state] != 0:
					transition_matrix[state][item] =my_round(transition_matrix[state][item]/s[state], 3)		

		#other states
		for state in states:
			if state == 'S':
				continue
			elif state in to_E:
				continue
			elif state[0] == 'I':
				s = 0
				to_states = ['I' + state[1], 'M' + str(int(state[1]) + 1), 'D' + str(int(state[1]) + 1)]
				for state_ in to_states:
					transition_matrix[state][state_] += pseudocount
					s += transition_matrix[state][state_]
				for state_ in to_states:
					transition_matrix[state][state_] = my_round(transition_matrix[state][state_]/s, 3)


	

		#print(states)
		m, n = len(states), len(states)
		for i in range(m-1):
			a = int(min((i+1)/3*3+1, n))
			b = int(min((i+1)/3*3+4, n))
			for j in range(a, b):
				x = states[i]
				y = states[j]
				print(x, y)
				if x[0] == y[0] and x != y:
					continue
				transition_matrix[x][y] += pseudocount

		for state in states:
			s = 0
			for state_ in states:
				s += transition_matrix[state][state_] 
			if s != 0:
				for state_ in states:
					transition_matrix[state][state_] = my_round(transition_matrix[state][state_]/s, 3)

	else:
		for state in emission_matrix:
			for x in alphabet:
				emission_matrix[state][x] = my_round(emission_matrix[state][x], 3)
				if emission_matrix[state][x] == 0.0:
					emission_matrix[state][x] = int(emission_matrix[state][x])
			for state_ in states:
				transition_matrix[state][state_] = my_round(transition_matrix[state][state_], 3)
				if transition_matrix[state][state_] == 0.0:
					transition_matrix[state][state_] = int(transition_matrix[state][state_])



	return nodes_match, nodes_delete, nodes_insert, states, transition_matrix, emission_matrix


def sequence_alignment_with_profile_hmm(nodes_match, nodes_delete, nodes_insert, states, alphabet, transition_matrix, emission_matrix, x):

	#use viterbi graph to find optimal hidden path
	#construct viterbi graph
	
	states.pop(0)
	states.pop(len(states) - 1)

	m = int(len(states)/3) + 1 #number of rows 
	n = len(x) + 1 #number of columns

	#l = [666 for i in range(n)]
	match_node_scores = np.empty(shape = (m, n), dtype = float)
	delete_node_scores = np.empty(shape = (m, n), dtype = float)
	insert_node_scores = np.empty(shape = (m, n), dtype = float)
	#l = [None for i in range(n)]
	match_backtrack = np.empty(shape = (m,n), dtype = tuple)
	delete_backtrack = np.empty(shape = (m,n), dtype = tuple)
	insert_backtrack = np.empty(shape = (m,n), dtype = tuple)

	match_backtrack.fill(None)
	delete_backtrack.fill(None)
	insert_backtrack.fill(None)

	match_node_scores.fill(666)
	delete_node_scores.fill(666)
	insert_node_scores.fill(666)

	match_node_scores[:,0] = 777
	delete_node_scores[0,:] = 777
	match_node_scores[0,:] = 777
	insert_node_scores[1:,0] = 777


	#I0 row values
	insert_node_scores[0, 1] = np.log(emission_matrix['I0'][x[0]]*transition_matrix['S']['I0'])
	insert_backtrack[0, 1] = None
	for k in range(2, n):
		insert_node_scores[0, k] = insert_node_scores[0, k-1] + np.log(emission_matrix['I0'][x[k-1]]*transition_matrix['I0']['I0'])
		insert_backtrack[0, k] = ('I', 0, k - 1)

	#M1 row values
	match_node_scores[1, 1] = np.log(emission_matrix['M1'][x[0]]*transition_matrix['S']['M1'])
	match_backtrack[1, 1] = None
	for k in range(2, n):
		match_node_scores[1, k] = insert_node_scores[0, k-1] + np.log(emission_matrix['M1'][x[k-1]]*transition_matrix['I0']['M1'])
		match_backtrack[1, k] = ('I', 0, k - 1)

	#D1 row scores
	delete_node_scores[1, 0] = np.log(1*transition_matrix['S']['D1'])
	delete_backtrack[1, 0] = None
	for k in range(1, n):
		delete_node_scores[1, k] = insert_node_scores[0, k] + np.log(1*transition_matrix['I0']['D1'])
		delete_backtrack[1, k] = ('I', 0, k)

	#first column delete scores
	#delete_node_scores[1, 0] = 0
	#delete_backtrack[1, 0] = None
	for l in range(2, m):
		delete_node_scores[l, 0] = delete_node_scores[l-1, 0] + np.log(1*transition_matrix['D' + str(l-1)]['D' + str(l)])
		delete_backtrack[l, 0] = ('D', l-1, 0)

	#second column insert scores
	for l in range(1, m):
		insert_node_scores[l, 1] = delete_node_scores[l, 0] + np.log(emission_matrix['I' + str(l)][x[0]]*transition_matrix['D' + str(l)]['I' + str(l)])
		insert_backtrack[l, 1] = ('D', l, 0)

	#second column match scores
	for l in range(2, m):
		match_node_scores[l, 1] = delete_node_scores[l-1, 0] + np.log(emission_matrix['M' + str(l)][x[0]]*transition_matrix['D' + str(l-1)]['M' + str(l)])
		match_backtrack[l, 1] = ('D', l-1, 0)

	
	def M_score_relation(l, k):

		M = match_node_scores[l-1, k-1] + np.log(emission_matrix['M' + str(l)][x[k-1]]*transition_matrix['M' + str(l-1)]['M' + str(l)])
		D = delete_node_scores[l-1, k-1] + np.log(emission_matrix['M' + str(l)][x[k-1]]*transition_matrix['D' + str(l-1)]['M' + str(l)])
		I = insert_node_scores[l-1, k-1] + np.log(emission_matrix['M' + str(l)][x[k-1]]*transition_matrix['I' + str(l-1)]['M' + str(l)])
		score = max(M, D, I)
		if score == M:
			return M, ('M', l-1, k-1)
		elif score == D:
			return D, ('D', l-1, k-1)
		else:
			return I, ('I', l-1, k-1)

	def D_score_relation(l, k):

		M = match_node_scores[l-1, k] + np.log(1*transition_matrix['M' + str(l-1)]['D' + str(l)])
		D = delete_node_scores[l-1, k] + np.log(1*transition_matrix['D' + str(l-1)]['D' + str(l)])
		I = insert_node_scores[l-1, k] + np.log(1*transition_matrix['I' + str(l-1)]['D' + str(l)])
		score = max(M, D, I)
		if score == M:
			return M, ('M', l-1, k)
		elif score == D:
			return D, ('D', l-1, k)
		else:
			return I, ('I', l-1, k)

	def I_score_relation(l, k):

		M = match_node_scores[l, k-1] + np.log(emission_matrix['I' + str(l)][x[k-1]]*transition_matrix['M' + str(l)]['I' + str(l)])
		D = delete_node_scores[l, k-1] + np.log(emission_matrix['I' + str(l)][x[k-1]]*transition_matrix['D' + str(l)]['I' + str(l)])
		I = insert_node_scores[l, k-1] + np.log(emission_matrix['I' + str(l)][x[k-1]]*transition_matrix['I' + str(l)]['I' + str(l)])
		score = max(M, D, I)
		if score == M:
			return M, ('M', l, k-1)
		elif score == D:
			return D, ('D', l, k-1)
		else:
			return I, ('I', l, k-1)

	#I1 row scores
	for k in range(2, n):
		insert_node_scores[1, k], insert_backtrack[1, k] = I_score_relation(1, k)

	#D scores second column
	for l in range(2, m):
		delete_node_scores[l, 1], delete_backtrack[l, 1] = D_score_relation(l, 1)

	#all other nodes
	for l in range(2, m):
		for k in range(2, n):
			match_node_scores[l, k], match_backtrack[l, k] = M_score_relation(l , k)
			delete_node_scores[l, k], delete_backtrack[l, k] = D_score_relation(l, k)
			insert_node_scores[l, k], insert_backtrack[l, k] = I_score_relation(l, k)

	#score for end node
	#backtrack
	backtrack = []
	M = match_node_scores[m-1, n-1] + np.log(1*transition_matrix['M' + str(m-1)]['E'])
	D = delete_node_scores[m-1, n-1] + np.log(1*transition_matrix['D' + str(m-1)]['E'])
	I = insert_node_scores[m-1, n-1] + np.log(1*transition_matrix['I'+ str(m-1)]['E'])
	score = max(M, D, I)
	l, k = m-1, n-1

	if score == M:
		backtrack.append('M' + str(m-1))
		current_node, l , k = match_backtrack[l, k]

	elif score == D:
		backtrack.append('D' + str(m-1))
		current_node, l, k = delete_backtrack[l, k]

	else:
		backtrack.append('I' + str(m-1))
		current_node, l, k = insert_backtrack[l, k]	

	backtrack_matrix = []

	while True:

		if current_node == 'M':
			backtrack.append('M' + str(l))
			backtrack_matrix = match_backtrack

		elif current_node == 'D':
			backtrack.append('D' + str(l))
			backtrack_matrix = delete_backtrack

		elif current_node == 'I':
			backtrack.append('I' + str(l))
			backtrack_matrix = insert_backtrack

		if backtrack_matrix[l, k] == None:
			break

		current_node, l, k = backtrack_matrix[l, k]

	backtrack = backtrack[::-1]

	return ' '.join(backtrack)


'''
file = 'alignment.txt'
x = 'EEBEABDCEEABCCCEEBDEDCADEDACCDCBBEECDBDACABDADCBEE'
strings = input_alignment(file)
pseudocount = 0.01
alphabet = 'A B C D E'.split(' ')
theta = 0.359
nodes_match, nodes_delete, nodes_insert, states, transition_matrix, emission_matrix  = profile_hmm(theta, strings, alphabet, pseudocount)
#print(transition_matrix, emission_matrix)
print(sequence_alignment_with_profile_hmm(nodes_match, nodes_delete, nodes_insert, states, alphabet, transition_matrix, emission_matrix, x))
'''
'''
file = 'alignment.txt'
strings = input_alignment(file)
pseudocount = 0.01
alphabet = 'A B C D E'.split(' ')
theta = 0.359
nodes_match, nodes_delete, nodes_insert, states, transition_matrix, emission_matrix = profile_hmm(theta, strings, alphabet, pseudocount)

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


