
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
			l = line.split('\t')
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


file = 'hmm.txt'
path = 'BABAAABABBBBABAABBABAAABABBABAAAABAABABBBAAABAAAAB'
states = 'A B'.split(' ')
transition_matrix = input_transition(file, states)
print(probability_of_hidden_path(path, transition_matrix))