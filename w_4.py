
import networkx as nx


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

def optimal_hidden_path(sequence, emission_matrix, transition_matrix, states, alphabet_x):

	G = nx.DiGraph()
	number_of_rows = len(states)
	number_of_columns = len(sequence)
	number_of_nodes = number_of_rows*number_of_columns + 2

	#add nodes
	for i in range(number_of_nodes):
		#print(i)
		if i == 0:
			G.add_node(i, state = 'source', score = 0)
		elif i == number_of_nodes - 1:
			G.add_node(i, state = 'sink', score = 0)
		elif i%2 == 0:
			G.add_node(i, state = 'B', score = 0)
		else:
			G.add_node(i, state = 'A', score = 0)

	#add edges
	for i in range(number_of_nodes):
		if i == 0:
			G.add_edge(i, i+1, weight = 0.5)
			G.add_edge(i, i+2, weight = 0.5)
		elif i == number_of_nodes - 1:
			break
		elif i == number_of_nodes - 3 or i == number_of_nodes - 2:
			G.add_edge(i, number_of_nodes - 1, weight = 1)
		elif i%2 == 0:
			G.add_edge(i, i+1, weight = 0)
			G.add_edge(i, i+2, weight = 0)
		else:
			G.add_edge(i, i+2, weight = 0)
			G.add_edge(i, i+3, weight = 0)

	#assign weight to edges
	i = 1
	node_state = nx.get_node_attributes(G, 'state')
	
	for u, v, weight in G.edges.data():
		if u == 0:
			continue
		if v == number_of_nodes - 1:
			break
		state_u = node_state[u]
		state_v = node_state[v]
		if v%2 == 0:
			i = (v//2) - 1
		else:
			i = (v//2)
		G.edges[u, v]['weight'] = transition_matrix[state_u][state_v]*emission_matrix[state_v][sequence[i]]
	
	#assign score to nodes
	G.nodes[0]['score'] = 1

	for node, attr in G.nodes.data():
		if node == 1 or node == 2:
			G.nodes[node]['score'] = G.edges[0, node]['weight']
			continue
		if node == 0:
			continue

		incoming_edges = G.in_edges(node)
		weights = []

		for edge in incoming_edges:
			weights.append(G.edges[edge[0], edge[1]]['weight']*G.nodes[edge[0]]['score'])
		G.nodes[node]['score'] = max(weights)

	#bactrack to find hidden path
	backtrack = []
	current_node = number_of_nodes - 1
	while current_node != 0:
		node_score = G.nodes[current_node]['score']
		incoming_edges = G.in_edges(current_node)
		for edge in incoming_edges:
			u = edge[0]
			v = edge[1]
			if node_score == G.edges[u, v]['weight']*G.nodes[u]['score']:
				backtrack.append(G.nodes[u]['state'])
				current_node = u
				break
	backtrack = backtrack[::-1]
	backtrack.remove('source')
	return ''.join(backtrack)

sequence = 'yyxxxzzyzyzxzyyyxyyxzzyzyzxxzzxyzyzxxxzyzzxxyxyzxyxyxzxzyyzyxyxzxyzzyzxzyzxzzxzyzzxxxyxyyyxyyzzzyxyx'
file = 'transition.txt'
states = 'A B'.split(' ')
alphabet_x = 'x y z'.split(' ')
transition_matrix = input_transition(file, states)
file = 'emission.txt'
emission_matrix = input_emission(file, states, alphabet_x)
print(optimal_hidden_path(sequence, emission_matrix, transition_matrix, states, alphabet_x))
'''
file = 'hmm.txt'
path = 'BABAAABABBBBABAABBABAAABABBABAAAABAABABBBAAABAAAAB'
states = 'A B'.split(' ')
transition_matrix = input_transition(file, states)
print(probability_of_hidden_path(path, transition_matrix))
'''
'''
file = 'hmm.txt'
sequence = 'zzzyzzxzzyyzxyxxxyyzxyyxxxyzzzxyzzzzyzzzxxxzyyzxyx'
hidden_path = 'BAABAAAAABAAAAABAAAABABBABAAAABAABABAABBAAABBABBAB'
states = 'A B'.split(' ')
alphabet_x = 'x y z'.split(' ')
emission_matrix = input_emission(file, states, alphabet_x)
print(probability_of_outcome_given_hidden_path(sequence, hidden_path, emission_matrix))
'''