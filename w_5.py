
import math

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


