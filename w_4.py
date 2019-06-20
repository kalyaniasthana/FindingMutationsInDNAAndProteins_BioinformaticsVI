
def input_transition(file, x):
	transition_matrix = {}
	for i in x:
		transition_matrix[i] = {}

	for key in transition_matrix:
		for i in x:
			transition_matrix[key][i] = 0

	with open(file) as f:

		for line in f:
			l = line.strip()
			l = l.split('\t')
			if l[0] == 'A':
				transition_matrix[l[0]]['A'] = float(l[1])
				transition_matrix[l[0]]['B'] = float(l[2])
			else:
				transition_matrix[l[0]]['A'] = float(l[1])
				transition_matrix[l[0]]['B'] = float(l[2])

	return transition_matrix

def probability_of_hidden_path(path, transition_matrix):
	pr = 0.5
	path = list(path)
	for i in range(len(path) - 1):
		pr *= transition_matrix[path[i]][path[i+1]]
	return pr

def input_emission(file, states, alphabet_x):
	emission_matrix = {}
	for i in states:
		emission_matrix[i] = {}
	for i in emission_matrix:
		for j in alphabet_x:
			emission_matrix[i][j] = 0

	with open(file) as f:
		for line in f:
			l = line.strip()
			l = l.split('\t')
			for j in range(len(alphabet_x)):
				emission_matrix[l[0]][alphabet_x[j]] = float(l[j+1])

	return emission_matrix

def probability_of_outcome_given_hidden_path(sequence, hidden_path, emission_matrix):
	pr = 1
	hidden_path = list(hidden_path)
	sequence = list(sequence)
	for i in range(len(hidden_path)):
		pr *= emission_matrix[hidden_path[i]][sequence[i]]
	return pr

'''
file = 'hmm.txt'
path = 'BABAAABABBBBABAABBABAAABABBABAAAABAABABBBAAABAAAAB'
states = 'A B'.split(' ')
transition_matrix = input_transition(file, states)
print(probability_of_hidden_path(path, transition_matrix))
'''
file = 'hmm.txt'
sequence = 'zzzyzzxzzyyzxyxxxyyzxyyxxxyzzzxyzzzzyzzzxxxzyyzxyx'
hidden_path = 'BAABAAAAABAAAAABAAAABABBABAAAABAABABAABBAAABBABBAB'
states = 'A B'.split(' ')
alphabet_x = 'x y z'.split(' ')
emission_matrix = input_emission(file, states, alphabet_x)
print(probability_of_outcome_given_hidden_path(sequence, hidden_path, emission_matrix))