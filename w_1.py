import networkx as nx

def trie_patterns_input(file):
	read = open(file)
	patterns = []
	for line in read:
		l = line.strip()
		patterns.append(l)
	return patterns

def trie_construction(patterns):

	G = {}
	labels = {}
	root = 0
	new_node = 0
	G[root] = []
	labels[root] = []

	for pattern in patterns:
		current_node = root
		for i in range(len(pattern)):
			current_symbol = pattern[i]
			suc = G[current_node]
			for successor_node in suc:
				if len(suc) > 0:
					index = suc.index(successor_node)
					if labels[current_node][index] == current_symbol:
						current_node = successor_node
						break
			else:
				new_node += 1
				G[new_node] = []
				labels[new_node] = []
				G[current_node].append(new_node)
				labels[current_node].append(current_symbol)
				current_node = new_node
					
	return G, labels

def trie_contruction_print(G, labels):
	edges = []
	file = open('trie_output.txt', 'w')
	for node in G:
		for i in range(len(G[node])):
			line = str(node) + '->' + str(G[node][i]) + ':' + labels[node][i]
			file.write(line + '\n')
	return edges

				
file = 'trie.txt'
patterns = trie_patterns_input(file)
G, labels = trie_construction(patterns)
trie_contruction_print(G, labels)