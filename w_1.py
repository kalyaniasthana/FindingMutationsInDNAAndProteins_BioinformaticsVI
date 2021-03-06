import copy
import collections
import sys
sys.setrecursionlimit(10000)

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


def out_degree(G):
	out_degrees = {}
	for node in G:
		out_degrees[node] = len(G[node])
	return out_degrees

def in_degree(G):
	in_degrees = {}
	for node in G:
		if node not in in_degrees:
			in_degrees[node] = 0
		for key in G[node]:
			if key not in in_degrees:
				in_degrees[key] = 0

	for node in G:
		for key in G[node]:
			in_degrees[key] += 1

	return in_degrees


def prefix_trie_matching(text, G, labels, i):
	itr = 0
	symbol = text[itr]
	v = 0
	out_degrees = out_degree(G)
	while True:
		#print(symbol)
		#print(itr)
		found = 0
		w = None
		for edge in G[v]:
			index = G[v].index(edge)
			label = labels[v][index]
			if label == symbol:
				found = 1
				w = edge
				break

		if out_degrees[v] == 0:
			return i

		elif found == 1:
			itr += 1
			symbol = text[itr]
			v = w

		else:
			return 'No matches found'


def trie_matching(text, G, labels):
	x = 0
	y = len(text)
	indices = []
	#return G, labels
	
	while True:
		match = text[x:y]
		#print(match)
		rt = prefix_trie_matching(match, G, labels, x)
		if isinstance(rt, int):
			indices.append(rt)
		x += 1
		if x == y - 1:
			break

	return indices

def modified_suffix_trie_construction(text):
	
	trie = {}
	root = 0
	new_node = 0
	trie[root] = []
	symbol = {}
	symbol[root] = []
	position = {}
	position[root] = []
	leaf_labels = {}
	#colors = {}
	#index = text.index('#')
	for i in range(len(text)):
		current_node = root
		for j in range(i, len(text)):
			current_symbol = text[j]
			suc = trie[current_node]
			for successor_node in suc:
				if len(suc) > 0:
					index = suc.index(successor_node)
					if symbol[current_node][index] == current_symbol:
						current_node = successor_node
						break

			else:
				new_node += 1
				trie[new_node] = []
				symbol[new_node] = []
				trie[current_node].append(new_node)
				symbol[current_node].append(current_symbol)
				if current_node not in position:
					position[current_node] = []
				position[current_node].append(j)
				current_node = new_node

		if len(trie[current_node]) == 0:
			leaf_labels[current_node] = i

	return trie, symbol, position, leaf_labels


def maximal_non_branching_paths(trie):
	out_degrees = out_degree(trie)
	in_degrees = in_degree(trie)
	paths = []
	temp = copy.deepcopy(trie)

	for node in trie:
		if (in_degrees[node] != 1 and out_degrees[node] != 1) or (in_degrees[node] != out_degrees[node]):
			if out_degrees[node] > 0:
				visited = copy.deepcopy(trie[node])
				while len(visited) > 0:
					stack = []
					stack.append(node)
					edge = visited.pop()
					stack.append(edge)
					while in_degrees[edge] == out_degrees[edge] == 1:
						u = trie[edge][0]
						stack.append(u)
						edge = u
					paths.append(stack)


	used = []
	cycles = []

	for key in temp:
		if key not in used:
			cycle = []
			cycle.append(key)
			curr = key
			while (in_degrees[curr] == out_degrees[curr] == 1):
				u = temp[curr][0]
				cycle.append(u)
				curr = u
				if cycle[0] == cycle[-1]:
					cycles.append(cycle)
					for i in cycle:
						used.append(i)
					break

	for cycle in cycles:
		paths.append(cycle)

	return paths

def modified_suffix_tree_construction(trie, symbol, position, leaf_labels, text):
	pos = {}
	length = {} 
	new_trie = {}
	paths = maximal_non_branching_paths(trie)
	#print(trie)
	#print(paths)
	for path in paths:
		#print(path)
		x = len(path) - 1
		if path[0] not in new_trie:
			new_trie[path[0]] = []

		new_trie[path[0]].append(path[x])

		index = trie[path[0]].index(path[1])	
		pos[(path[0], path[1])] = position[path[0]][index]
		length[(path[0], path[1])] = x + 1
		#print(new_trie)
	file = open('trie_output.txt', 'w')
	patterns = []
	for edge in pos:
		s = text[pos[edge]: pos[edge] + length[edge] - 1]
		file.write(s + '\n')
		patterns.append(s)
	return patterns, new_trie, pos, length, paths


def longest_repeat(trie):

	def dfs(root, label):
		if (len(trie[root]) > 1) and (len(patterns[-1]) < len(label)):
			patterns.append(label)
		for edge in trie[root]:
			index = trie[root].index(edge)
			s = symbol[root][index]
			dfs(edge, label + s)

		return None

	patterns = ['']
	root = 0
	dfs(root, '')
	return patterns[-1]

def trie_input(file):
	read = open(file)
	trie = {}
	for line in read:
		l = line.strip('\n')
		l = l.split(' -> ')
		if l[1] == '{}':
			trie[int(l[0])] = []
		else:
			x = l[1].split(',')
			x = list(map(int, x))
			trie[int(l[0])] = x
	return trie

def colors_input(file):
	read = open(file)
	c = {}
	for line in read:
		l = line.strip('\n')
		l = l.split(': ')
		c[int(l[0])] = l[1]
	return c

def trie_coloring(trie, colors):

	while len(trie) != len(colors):
		for node in trie:
			#print(colors)
			if node not in colors:
				children = trie[node]
				found = 0
				for x in children:
					if x not in colors:
						found = 1
						break
				if found == 1:
					continue
				else:
					colors_children = list(set([colors[c] for c in children]))
					if len(colors_children) == 1:
						colors[node] = colors_children[0]
					else:
						colors[node] = 'purple'
	return colors


def longest_shared_substring(trie, symbol, leaf_labels, text):
	colors = {}
	index = text.index('#')
	for leaf in leaf_labels:
		if leaf_labels[leaf] <= index:
			colors[leaf] = 2
		else:
			colors[leaf] = 1

	def color_node(node):
		if node in colors:
			return colors[node]

		children = trie[node]
		color = 0

		for child in children:
			if child in colors:
				color |= colors[child]
			else:
				color |= color_node(child)

		colors[node] = color
		return color

	for node in trie:
		color_node(node)

	def dfs(root, label):
		if (len(trie[root]) > 1) and (len(patterns[-1]) < len(label)) and (colors[root] == 3):
			patterns.append(label)
		for edge in trie[root]:
			if colors[edge] == 3:
				index = trie[root].index(edge)
				s = symbol[root][index]
				dfs(edge, label + s)

		return None

	patterns = ['']
	root = 0
	dfs(root, '')
	return patterns[-1]

def shortest_non_shared_substring(text1, text2):
	patterns = [text1]
	for i in range(len(text1)):
		for j in range(i+1, len(text1)):
			p = text1[i: j]
			if (p not in text2) and (len(p) < len(patterns[-1])):
				patterns.append(p)
	#print(patterns[-1] in text2)
	return patterns[-1]

text1 = 'AAAATAAACAAAGAATTAATCAATGAACTAACCAACGAAGTAAGCAAGGATATACATAGATTTATTCATTGATCTATCCATCGATGTATGCATGGACACAGACTTACTCACTGACCTACCCACCGACGTACGCACGGAGAGTTAGTCAGTGAGCTAGCCAGCGAGGTAGGCAGGGTTTTCTTTGTTCCTTCGTTGCTTGGTCTCTGTCCCTCCGTCGCTCGGTGTGCCTGCGTGGCTGGGCCCCGCCGGCGCGGGGAAACGCGCCATTCTACGTATCGTAAGCTTAAGTCCCAATGCCCACCTCGCTTACCTTGTCAGCTTATTTTTGCTGGATTGAACAGCAAGGCTTTCGATTGAGACTTCAGGCTCTGGCCGATGAATATGTGCGTAGCAAGTAGGTCTGGGTATTGTGGAACCTCAGCTTGACTACTAGAGATACACATCTGCCGAGTGATCGTAGTAGATAAAGTGACAGCGGAAACAGTGATTTTATTCCCGGTAAGCCTGACTGCCCTAACACTGGGCCAGGGTAACCAGTGACATTATACCGGCCTATGGCCATAGCGGTAAGAACAGCACCTGCTAAAAGGTTCGTATATACTGAGCCATCAGGGGGGTACCCCTGTTTCGACCAAAGACAATGGTCTTAATCTCCATGC'
text2 = 'AAAATAAACAAAGAATTAATCAATGAACTAACCAACGAAGTAAGCAAGGATATACATAGATTTATTCATTGATCTATCCATCGATGTATGCATGGACACAGACTTACTCACTGACCTACCCACCGACGTACGCACGGAGAGTTAGTCAGTGAGCTAGCCAGCGAGGTAGGCAGGGTTTTCTTTGTTCCTTCGTTGCTTGGTCTCTGTCCCTCCGTCGCTCGGTGTGCCTGCGTGGCTGGGCCCCGCCGGCGCGGGGAAA'
print(shortest_non_shared_substring(text1, text2))
'''
text = 'TTGAATGACTCCTATAACGAACTTCGACATGGCA$'
trie, symbol, position, leaf_labels = modified_suffix_trie_construction(text)
print(len(leaf_labels))
'''
'''
text1 = 'ACAACCGGTTGGGTTTTCCGGCATCTGACGCGCCCCGCCCGTGAAGGTTTTCTGATTTGATAAGGGTCGATGGCTTAAACATCGCTTAGCATACTGAGCCGTTGGGGATTAAGACCCGTAAACCATGCATTCCTTTAGGGACCGGTGTTGTTTTACGGCGGCGTGCGAGAAAAGGAATAAGGTCAACGCCGAAATTACGAGGTCTAATCCTACGTACGTCTGGGGGTCCTTATTACAAGGAGTAGACTGGCCCCTCACATTGCCTGATGCCCACATGATTGATGACGCGCGCACTGCTTTTGACGGTAGCCACGCCTGGAATCTCGTTGCCGGCGCGCTCTTAAGCGCCGACAGTCCCGTCACATATGCTGATGAAAGCTTTAGAGCTTATAGACATAGCCGGGTACCGCACAGTGTGTCTGCCTCGTAAACTTTTTAGGTCGGGGAGACCTGGTAAGATAGTCGGTCTTCCTTTAGTTTTACCTGAGTGGACCATACTTGTCCTAGTTTTTCTGCGTAGTCTCCCTAATTAAGGTGAGGGGCTTCTGCGTTAGATAAATGCATCTTGTTTTCCAAGAGATATAGTCTCACATTAACGAACTGTCGCCTTAATGTCGTCTGAGATTACCAGGCGGGCCACTCGGCTACGTAATATGGATAGGATTCATACACAAATTATGCAGGCGCGTGCGCAGACTTGGTGCTTTCCCCCAGGAATTACGGCTTGACGAGAGTGATGCTGGATGTCTTAGGCGAGTAAAGGCGAAAGGCCGCAAGGGGTAAACCTGTCTAGAACTGAGCCGACTTACGCTGCGAGCTCCAGAAACCCTTCGCGTACCCATGATTTCATCGTCCTGACTACAGAGTATCGCGTGATAAAGGGCTGGCGACGCAGTCCGCTACGATGAGTGCTGTAATATCTAAATCCCCTCTGTTACAATATCTGAGCACGAGGAGTAAGAGGTCCAGTGTACTATGAAACTTTCTATAGTTTCGTGTG'
text2 = 'GCTTGAGAGACGTACTAACCTTGAACCGAAATGAGTCTCACCCAGTCCTCCCACGTCGGAGAAGGGGGTTGTGATTAGGACCGATGTGAGTACCACATACCTCTGACCATGCCTGCCGTCGTTATGTCCTCTTTGAATCGGAGCTCTAACTGGCCCTTTACTCGAACCCGACAAGGGAGTGCTTTTCTGCTCAAAAGGGAGTACATTGACCCAACCACGTTTGGACAAAAAGACTTGAATGCTACACTTAACCGTTTGAGGCCAACCGGAATCAAAGTCGCTACTCACAGACAAGGGGTACTGCTAATCTCCGGATCCATCGTTTCCACGATTGACAACCGTTCATCGCTTACCGAAGTTCGTGCGTGAAATCCCCAGTCTCAGGTGGTCCCCCCCGGTGGTCTAGCTTTAAGAAGTAAGATATCTACCAGCTTCTGTCGGCTGGCCTACACTATATGATTAACGCATAACATTCTTTGCAAGCGAGAAGATAGAGTATCCCGCTGACCGGCCGCATGGAGTTCTACATATTGCTATCTGTTCCGAGGTCGCGAATGTCCACGGAAGATCAGATCACTGAATTATAATCTGACGCGCGAACCAGGGTACTCTTACATGCCGATCACGTCTTTTTAAACGGTCGGGTTCTCTCGACCTACTCGATAAACTCAGGATTATCGGTGGGCGCTACTTACCCCTCATTTTCGCGGCCTTCGTTGCTCGGAACGCGATAGGAGTACTATGCCACTGAGGTTCAAAGCCATAAGCCGCCCATCGGGCGTCTTGACAGCTACGCGTATTTCAGGGGCCACCATTTGGCCAAGGTTTCTGGTCCCGAGCCCGCACCGACCGAGACTGGAGCCCTTTGATAGGCCTGGAGGATTCATCTGTCGTCTGCCAGCCTGGAAGTCATCTTGATTGAACAACAATACTATCCTCCGTCACATTCACGAACCAGCCTTGTCCTAAGAGCGATTGCTCGCGTTAACGGCCCGATGTC'
text = text1 + '#' + text2 + '$'
trie, symbol, position, leaf_labels= modified_suffix_trie_construction(text)
print(longest_shared_substring(trie, symbol, leaf_labels, text))
'''
'''
file = 'nodes.txt'
trie = trie_input(file)
file = 'colors.txt'
colors = colors_input(file)
colors = tree_coloring(trie, colors)
for key in colors:
	print(str(key) + ':' + ' ' + colors[key])
'''
'''
text = 'CAGCCGTGTAGTCAGGTTGGCGCTTGATGTCCCAGGTGCCACGATCGGCGCCTTGTCCTCCTTTCGAGTCGTGAGACTGGTGGACCGTGGACCCTCCTAGGTACAGGACCGTGTACAGTGCTCGGGGCAGGCCCAGCGGAACTCGCAGTGCAATCGCAATCACAAGTTATCAGTAGGTAACAGGACCACAAAGGTGATCCGCTGTGAGGTAACTCAAAACGCCGCGCCTACGGTTATTTATGTCCAATACTAGCTAAGTAGACGTAATCTCCATAGGAGATCCACTCCGAGCTCATACTAAGGCTCAATAAATCGTTTCTACCTGGGGTCATGCCTAGGTCTCACTTCGGACGGCGAGTAAGTGTGTCTTTCATCTGTCTCCTGCCCATAAGGCATGTCACTTATTTATACTTTCTCCCGCGATGACTGGAGGTCGCTCGGTTGGATTAGAGTTTGCACCGGCCAGTTGTTAGCCGTTGCAGGGGTGATTTTCCGCCCATACATGATAAGAGCTTTTAGTGTTACTAGTCTGACTTTTCGTATGCAGTCAGGATTAGCCGATTTTCCGCCCATACATGATAAGAGCTTTTAGTGTTACTAGTCTGACTTTTCGTATGCACTGCGTTCGCCCTACGATCATATTCACTAAAACGTCGTTATCATTATGGACTAGTGGGATTTTCCGCCCATACATGATAAGAGCTTTTAGTGTTACTAGTCTGACTTTTCGTATGCAAGGTGTCTAGTATAGCAGCTAAAGCCGAATCGAGAAGTCAGTCCTCTGCGAATGCTCCAGTACCTGTACCTTGTAATCTTAGTCCACAGCAAGGTCTAGTCTAAGTTGTGTACAGCAAGATTCAACTTACTATTGCCGTTTAGACAAGTGTGACGCCTAAGTTTTGTACCATCAGAAGCGAAAGCACAGCAACACAACATTCAATGGCCGGTTCCACAGGTATATTTCACAACCAGTTCCCTGATCGATCGGTCGTCCGGGCTTCCTTCCTAGTGACGTCGCAACAGGGTTCGCCAGTTGCCGAGCTAACACTTTTGGGGCCTGGGTCAGGATATTTTCCATACGCCCGCTGGGCAGCAGACGCTTCTAAGTATCTTGCACTAGCCGGTACAGTGCCCATAGGGAAGATCAACCCTGGGTGCTCTAGGGATCCATATTCAGTTC$'
trie, symbol, position, leaf_labels = modified_suffix_trie_construction(text)
print(longest_repeat(trie))
'''

'''
text = 'GTCTTGTATCAGCAGGAGAAAACAGCATAATTCCTTAGCATGGTTTTTTCAGGATAGATATAGTTATGGCCAGACCTAAAACGCGGACATAGCCTACCATATATTTTCTCAAGGAGGCAAATTTCTGAGGCACGGCACCGTCCAACGGGTCGGAGGGTCATTCAGAGTTATAGGTAACAACTAATACCTTACGCCGGAACTCGCCTCCATATGAGCTTGGGGCTCGACTCCTAGATTGCGTCCACATCATTGAGCTTACTTGAATAATGTGTCTGGGTAAATCCTGAATCCGGGGCGGCTCCCTCTGATGAGACCAGGTTATCGAGCGCCGACAATTTCACAAACTCAATACCGAATAGTCTCGGATAGGATCAATACTGCTTAAACCAACAACACTCTGGCTGTCTCGACTTAAGACTTTGCGTGGCTTAGGCCGACGGGCGGCTTAGTGTAGCTATATGACTTGATCTAGGTGAATTTGAGACAAATTGGCGCGGCTCAATGCGATACGTTCTATCTTCGTTACGTGCTCCGACTTATCGACGCCCTGCCAAGCGCGGACGAGTACTTGCTATCATACGTTTGATCTGTGTAAAACTGACAAGGAACGCCACACTGGCCTGTAATCATTTGCGTAGATCTCAGGTGGGCCACTCGTATCGCACCTGCACACAATCCGGGACTAGATAGGGCATCACTGATTCGACAAGAACTGTAGGAGCGGCGTATAGGGAGCATACCGGGTCACGCTTTTTCCGAAACTTTCCACCCATGTCTGGTGCTAGAGTCGAGGGGTAGCGGTGGCACCGCTACGATCCAAGCGGATTACTCTCGCTATTACAGATCCAGCGTCCACTATAATGACTTTCTATGAGCCCACGTCAGATGCCGGAAGGCATGAGATGGTTCTGTGAACCTCTCTTACCGTTTTCGAGTACATGGCCTAT$'
trie, symbol, position, leaf_labels = modified_suffix_trie_construction(text)
patterns = modified_suffix_tree_construction(trie, symbol, position, leaf_labels, text)
for pattern in patterns:
	print(pattern)
'''
'''
file = 'trie.txt'
patterns = trie_patterns_input(file)
text = 'ATGCGGAACACATGGTCTGTACACGTGTAAGAGTACTTGTTTATTTTCACGCTTTGGATATCGAGTCCTATTCCCCCTCTGCCCACTACAAATTGTCACATTATAACGTGTAATTATAAACAATGAGTTTACTTTAGTGCGTCTTTTGATCCTTAAATCCGCCCAGTCATATGCATGAGCGGAGGACGGCTTGGTACGTAGTGCGCCTCTTAACTTGCGCCAGCGCAGCCGGAAACCCAGTATAAACATCCGCTTCCCTCACTCCGACGGATCTATGGTTATACTAACTAATTGTGGTGTGAGGAGAGCGGGCCCACTGAGTTCCAAACCGCGCCACCTTCTATAATTTTTGATTAGTGAAGGCCCAGCTTTCACTCGATGTACTTTTCCTGTGCACCGCACAAGGCGTTAACAGGTTTGTACCTGCTGGTCTCCTTAAACTCACCGCAGGAGGTCGGCCAAGTGACTGCTAAGCTCTAGGCTCGTGCTATAATAAAGATTGCGACGACAGAGCGCTAACACATCCGGTCAGTCTTCCATAAACCCTGACTCCGGTCCCGAAGGCTTGCGGTCTAATAGGGCCGGCTACTCCTCTCGACCTCGCTCCAGGGACGAAGTGTTAAAGGGGCTACCCAGCTTATGATCCCGTCCGTACTTGGTGGCTACACCGTGGCCTACGGCCGACCGAGACACTACTGATTGTGATTTAAAAATTAGGCCTGGTAGCGTGAACGAGCAGCGGTACCACGAGAGAACAGGGGTGCATCAAGGAAATTAGGCCACACAGTAGGAACTAGCCATATAAGGAGCAACCTATTGTTCCGTCCTTCTGGGCAGCTGGCCTGGTCGGTTTTCATCCGAAACTACAAGGAACCGCTGGAACCGGTAGTTAGTCCCAGGCAGCTGAACTGACCACCGTCTATTGCGGATTAGATGCACTGTCTGTAGCGCCGGGGGGTTACAAGGAACGCGCCAGGATAGAATAGATCCGCTCGAGCGAGAGTACTATTCTCCGTTGTTACTCCCTGCATTATAGCAACGGTGCCTAAAGCAGATGAGACTGATTGCGTGGCGAACGCTTCACACGTTTGGTAAAAGATCCCCCCGGCAAGCTGCACGCCCCGTTTGGCGAACCGACACACAGGAGGAGTTAATTCGAATGTACCGATCAGAACGGCGTGTTTGCATTAACTGCATGATGATTATATAGCTCGTTCAGGGCAGAAACTACTTATGGCATAATATCCTCGCAGAGCAGTTATAATGGCGAGAACACCTGGGTCGTGGTTGGCGATCCTCCCACGTCCAGACTCAAGAAGTCTATCCACTCACAGCTACATTTCGCAACAGGTACCGAACGTGGCCCATGGTACATCAACGAGTGTTGTATCTATATGTACCTACTTTACAGTGAACCTAGCTCTCGACGTGTTGGTCATTTGATAGATGGAATGTAGTGGTACCGATATTATTCCTGCTGGTAGGTAAATTGCGTCAGTTAACTTTTCAATGATAAGCACTACAAGAGCTCTTATCGCTCGCCAGTAAACGCAGAAAAACGAAAGCGCTGATCCGCGACGTGGCATATAATAGCCTAATGCCAAAAGTCCGGTTACATAGCCTGTCCATCACGGCTACGCAGGACTAAGCGGTCTAGATGACAGGAGAAATTCTGTAATGTCTGTGCCGGAGCCGGTTTTCTTAAATTTGTTTCTTTCGTCTATCAGGTGAACGCTTACAACTAGTGGGCCATGCAGAGTGGTCCACATCACACCAGTCTATGGCAGGTATAGCGGTTGGAACTAGATTTATCTCTTTAAAAAAGACAACGGGGGAGTCCCTAAGCCTTCGCACACACGATGACACACGAATAGAACCATTACAGAGGCGAGTCGGATGGCCCTCAAGCGTTTGCGAGGCGAGACGATCTAAGGTAAGTAAAAATACGCATTGCGCCCGGATTATAGCACATCCAGCTCTACTTGAAGGCGCTTAACTCGGCGTATTATAGGTCAATGTTGAATAGATGCCCCAACAAGCAGAATCTAGTCATACACGGACAAATAGTGCTCGCGCCACGGTTTCAGGACGAGGTGTACCCACTTTCACGGTACCACTCAGTTACTGCACCCGCCGCCCGCCGCCGCCGTCGCCAAATGCTTTCTTTTATGGCCCGTACTGATAAATAAAGATCGGATGCTCTGGCGGTCGCCGTCCACAACAGTGATCTTGTGCCACTCCGTCGCAGTCGCGGATGGTAGTCCAGGGCCCGTATACAGCTCTGACAATTCACTCTTTCCACTCTCATGCATTCCCTGGTTAGTGTTACAGAAGCGAGCGCCCGCCGCCCGCCGCCATGGCCAAGCACTCTTCCCACTGGAGGTTGTATACCGACCTTGTTGAAACGTGCGTATAAGCTACAAGTGGTAAACTGGTGCGTTATATGCCACCTACACTTCGACCCGCGGAACTCCTGTGGGATAACTTGAGGTTCGACTGGCGCTCCCTATGTCTTGCAGCGGATAAGTCCGTCGAAGGCGCCCCAGTAGAGGAGACACGAAAGAACCCATGAGATATCTCATCTTTTCTTTGAGAAAGCTGCAAAGGAACCGTTTTGGTATCCGGCCTACAGGCTTGACGATACTATATGCCTACAACCTCTCGTATAGTTGTAACCGGTAAACTGAGAGTGGAAGGCGAGCCTGGCCGAGGGCCGGAGGTGGGCACAATACCGCTAATGCATGGGAGGATAGTTCACGTCAGGGACTTTGGGCTAGCCCATCGTGTTTGTAACCCTTTGGCCCCCTTAGGTCTTCTCTGCGTTAATTAGTCATGAATCTTACTAATTCGCGATTGCACTTGCATCAAACTGGTTGGAATATTCCTGCTGGCTGTGGCCATTACCTAGTACTCCCTTAGACAACGTTACTCTGCTTATTGGGATAGCCCGCTTAAAGCTGGCCATTGCACATCCGTCGGTTCCGACCTCCGACCTCCGAAATTTGGGGGCGAGCGATACATGGGTGAGCATCTGGACCTAGTTCGGTAAGTGTAATTCTACGGCACGGAGCCGGAATCCTCTCTCATTGGGCTACTTTTCGGGGTTGAGGTCAGAATCTCTTGCCGGGGCCTTTGTCGTCCGACTGATGACGAGGACAGGTCTCAGAACACCTTTGTACGCGATCCTATTTGACGTCGCGACTGTATCCGCAGCTACATGATAATGTTTTGTGCTCATTGGTTAGTTAGGCAAACGGGAGGTGGCCGACTAACTTACAGGACGCAAGACATATATGTTACTACTCATGGTTGCAAAGCCATATTCTCAAGTCACCAGCTCGGAAGGGAGTCTTATCAGCCCCAACCGCTAGTCTATAGGTTGACGCTACCCAATGCCAAGCCCGCCTCAGATCTCTGCTCGAAGTGATAGTCCTACGCCTAATCTCTTGGAGGAGCAGGGGGAATGCCCGTACGAGTATATTCACGTCAACGGTTAAGACACGCACGGCTTAACATGCTAACCTAGACTATATTGAAGAATTAAGAGATGGCTAGTTGCAATAAAGAACCGATCGATTAGTTCGGTGCCGTAGTCCTGGAGATCCGCCGTAGTTCGCTTCTGGGTGCCATAGCTTCTTGGTCTAGCGGCTCGTCCCAAAGCCAATACAGTACACCTTGCCAAATAGTAAATAAGTGGATCCAGTCACGCCGAACAGAGAACCCTACCCCCAGTTCGACGGTTCGCGTAGTAAAGAGGTCGCATTCGCCGTCGCTCCTGGGTCATTGACGTAAGCATTAAAGTTAAGTAATACCCTTAACATCTAATACCGTTCTGTGCTCCGTGCAATAGAAACGGGCCGCGAAGTTGTGACATTTGCATACTGGGCCTTAAAGAGTAATTCTGTGAGCGACGAGCCCAGCGGCGCATCTACTGCCGGCCATCTCATTTGTTGGCTATCATCTCACCGTATTGCCCCTGGTTTAGTCGCGCCGGACAACCATGGTGGACTAATTCCCCGGGTGCTGTCTGTATACGGCTTCTCCTTGCCCGACGTGGCTATCCTATTCATATGTTATTGCTTGCCAAGTGCGCGGGCTCGTCACTACCACAATATGTAGTTCCACTGGGTCGCAACAACGCTACTCTTATGTAAGAAATCTTTTTAGTTCAGGAAGCCACGGCCGACCCGGGGGTTAGATGTCCAGGGCCTCACTCATAACGCTCGTACTTTGCGGATTGCGGATTACTACTTTCTCTTTTCCGCGCTACAGTTGCAAGTTAATGATGAATTGAAAGGCGTGTATCATGAGTTGCTACTTTAATTGAGTGGTTTGATGTCAACAGCCCCGGTACGGTCATTCAGCCATAGACTAGCCCGGAGTGTTGCGCGTCTTGCGTAGAATCAGGAGTTCCGTCACCCAGAGGGTGGTACGAGTGATCTATACCTTTCGAGTCTAGTTACCCGACTATCCTCTCCATCTACCCGCCTTATTGCTACGCAGCCGCAGCGATGTGCCTCATGTTACGATCCGTGAGGAGGGTTGGCACAATAATCCTGTGGCGGATATTACCCATAATGGTTAACTTTTACCACGCTGGTCGTCTACTGTTGAAGACCTGGCTGAATTGCTACAAATCTGCTAGGTAGGTGCTATCAGTCTAGTCTTCTACTTTCCTTCTACAACTAGGGGATCCAGCGTATAGCACATCTCATACAAGAATACATGACGACCCCAGACCGTCACGATATCTACTGTATCAATTAATGGCCCGAGTGATATGAAGAGGGAGGTGAACTGCATGTAGTGACGCTGAGATTAGACTCAGCGTACGAGGTGAGCAGTGCACTGTGACGCCTCAACGGGACTCCTTCTGAAGCATATGCAGGGCTAAACTGGGAATATTAATGGTTAGCTGTTAACTACTAACATTCGGTAAGTATGGGTGCCGCAGTACCTGTTGGTGTCGGCGCTATGTTAATACCCGGTGTGCAGAAGGGCTGCGGGGGTGCAATAAGGGGGATTCTCAACCACCTCGCTGTACCTCTTCCTGTGTCCTCGCGGTGTTCTTTTCCCTCGCAATATGTGCCTTCGTGTTCCCCTTAGATTGCGTCGATTGGGGCGGAGTTACCAGGCAAATGCCGGCCCATTATACTACGTTGAACAGCACTTTGACGGATCACTTCAGGAGTAAAAGCTAGCACACGGCGGCCACCAAGAACCGGAAGCAACCTCCCCGCCGATCGACGCGAGCGGACACCCGTATTAATTGGGGCCTTTCGGAATCGCACCGACAGCTGCCGCATTGCGGATTAGAACCCCTTCCCATTTCTACGCTGTAGAATAGTTGCTCCTTATCAAATTTGGCTGTCGGGCCGGTGTCCACCGGGCTACATGTCACACCTATGATGCGTCCATTTAACGTTATAGCACATAGCACATTATTCGATAATCTTCGATAGAACGTCCATCCGACCAGAGAGAACTGGTGGACTGTCGTGGGCCCGTTGCGGATTGCGGATTGGGGATCTATAGCACATAGCACATAAATATTCTTCATATAGCGTAGCTCGAAAGGGCGGTTCTCGAGTCGCTTGACCACAGACTCGTAAGCGTCTATCCCAGCGGTTCGGAAGCGCATCGTTCATCTTAATGGTTGATCCGACCCCGGATTTTCCTCTTGACACACTAGGAGCTGAAGAGCGGGATCGATTGGTTAACGGAGTTTAAAGCGAAACGCACCTTTCAACAACCTGGCCAGACCTTCCATACTACCTCGTACCATACAGCGAATTACATTATAATGTTGTAGGAGAGACGTGCGGACGACGATACGATAGAGGTGTCTAGGACGAGGAGTCGTCTTAGATTACTGGAAAACCACGAAAGGGACGGGGCAAGTGCATCAGTACCGACGTAGTAGTCCTCACAGGATCATATCCCTGGAACCCCCACATGGTAGCACACAATCCAGATCTCGTATCTTGGGCGTACTGAAAAGAGTCGCCAATCATACCGGTGCCAATACCGAAGCTGGCATTGCGGATTCACGTCCTGTGCTTAGCGTTTTGCTGGAAAAAGAGTGTCAAGAACGCAAGCTATTTTAGCAAAATTGAAGTCACACTGTAAGTACTCAGGGTCCAATTCTCCTCGGGTCATTGTTAGGTCGGAAAATGACTGGGACAGGTGGCCGCTATGCTGTCATAGCACCACACAACAGTTGACAGTGCCCCCCGGCGTCGAAGAAGGATGGAGTTGTGGTGCCTAAAAAACTGAGTCCTTTACGCACATTGTTGCTGACATCGGGGGTACTTGATCATCTCATTCTCCCGTTTGGTCGCATATCATACGAAATCGACTTGAGGGATGCCCGAGCTTGTAGCCTTGGTACGACTTGGTAAGCCTAAGACCGCTTTATCGTCCATTGACCTGCTGTGCAATATGACCAGCAGGTCACCGTATCTAAAGACCGGTTACTGTAATTGCGGTCCAAGCCCAAGCCAGGAACGTCCTACACGTGGCTTCTTAACATGTATTCCAACATCTCGTTTTAAAATAACTCAGGTTTTGGATCATGAGGCGGTAGAGTTATCCGGCAGGTAGCTGCTCGACTCGGTCCTCATGAAACATACTGACATCTTGAAGTGTTCCATAAGGCGGACGTATGAAATTCCACTGGTACATATATTGCTTCGAATGCACGCGCGGTATTCAGAAATTTGTTTAACCCAGGATCTTATTTTCGCTAACCGAGAATGCGCGTTTGCCCCTTCTGCTTACGGCAGACTTTACATTTGCAAACCCTGTACTGTGAAGTCGGTAGAGTTCGAATTCGTTGCGAGGACGTGTCAACATTACGCCGGTTTCCACACCTACTGTAATCGAGATACACTACGGATATTTACGGCCGACACAGTCTTTGGATTGGATCCTCTTTATGGTGCAAGTAGGTCGCCTCAATTATAGTATATGACTAGGCCTCAATATGCGTTTCAGCTAAGGTACTCGCTTTGATCATCGTAAGCAGTATGATGAGTCGGTGGTGATCGCCCCGCAAGTAGCTCCAGCAGGAAAGCGCTCACTTGGGGGTGTACGGTCGTCGCTCGCAAACCCACCCGCCAGGAAGTAGCCTGCGGGTGGGGGGGCTGTCCGTAGTTGACAGTTGACAGGCCGTGACAGTTTTCTGTCAACACAATTCTCGCAAACACGCGTAGAACGACTGCGAGCATTGCATAACTTCCCGACCTCCGTTATGCCACCACCGTCTAGCATAAGCTACTAGGCTGCGATTTTGTACGTACTGAAGCGTTACCGACTTGGGGCGGCGGGAGTATGTTACTCAGCGCTTTATTGGCCCAAGCGTAATCGTGATCGTGGGTAAGTGTCAGTTGCTTGCTAGGTAGGCATGCAAAAGTTGAGCAATCACTATTCAGGTGTAGCTGTTGTCGCAGCCTGCTTATCAATGGTACACAAAGTCCCTTGCACGATTATATCCCGCCGCCCGCCGCCTATCGAGCAATCCCGGATTTTTAAATCCGAAAGGAGTTACTCCTGGTGCACATTTTCCATTACTTTGCACCAGTTTGAAGATATAACCGACGGTGCTAGTACTTACAGTAATCGTCCGATGAGTGGTAGATCCAAGGCATTTGACGGGGCAGCAGGGACTCTACGGCCACGGTAGAGGTTGCGAATAGTAGAGACTTCACCTGCGAATCTAAAGACAATAGTGGGTTCATGGTACTTTCACCCTTAAGTTCGGGTTGCGATGATTAGAATTAGGTTATAGGGTGTGGCAGACATGGTGGGGTTCTGAGCTACTTCCACAAATAACTCCAGGACGAGCGTGCTTCCTTGGACTCGTGGCCTTTAGCCAGGACATATGACTGCTTATGCTTCTTTTCGACCAAAGACGGGGTCCAGCCCATGAAATTAGTTATACATAGCACATAAGCTATCCTACAGCACCGGTGAGGAGCATTAAGTTGCCTTTTCGAATCGAGGATTACGCTCGCCAAGTTCCAGCGACTTAGCATTACATATGAAATGGCCAATAATTAATGGTTCCTCATTCCCTCGGCCGCGGTGAACCTCTCTGGGTTCGAACTCGGTCACGTATGTATCTTTTGACAATGATTATGATCCTATGTCTCTGGGGGATAAGCTGACAGCGGGACAATGGATGGGCCTCATCAGCCGCTTTACCAGAGATAGAAATCACACTAGGCTTCCTCTCCACCACCTACTTTATACCGAGCGAGGTAGTATATTGTGGTGCAACGGTTAGCATTGATTTCACAGCAGCGAGGACCCGTTAACTCGGTGGATAGATTAACGTCTAGTCCGACCCCTCCATCGTCGCAGTATTAATAACCCAGTTTTCTTGATTATCCTCGACACGGCGCATTATGTAATGACTTGAAGAGGTGTTCTATGAACACCATCTCACCATATTCCTCGTACACCGATGTGAGATTATTTAGTCGAGGAAGAAATGATAGTTGATGTACGATATATAGCAGACGTAGCCGCAAAATAAGATGAAGGCTTGACCCCCAAGCAATTATCGGTCGTGGGCTCAACCCCGTACCCCAGGCAGGTTTGAAACCGACGGGTAAAATCCCAGCGAGTCCGTACTAGAATACCGCATCTGTGGGGTCCTCTCGTAGCAGTACAATGATCATAGGTAACCGTTACGGTATAGCTTTTCGCGTGCCCGAACTCCGAAGGAACCTGGAGTTTATGACGAAGGCCGGACACAACAAATCTCCGACCCGACCTCCGACCTCCAGTTGACAGCTTGTAGTCTGGCACCTAAGTCTTAGTCATTCCTAGGACTACGGCCGACGACATCTCCGAAGCAGCTAAACATTGGGCCCGTTTTGACTGTTGTATCATACAGTAGGGGAGACAGGTGAAAGGATGGGTTCGGTAGACTAACTTCGCAAACTTGCAACTGGTCCGAACCAGTGCAACTAGGTAGATACCTTAGAAGAAGCACCCCCTGCTTAATGAATGTTAAGAGTGATCTGACCCGCGTCCGTCGGTACAATACCTTACACAGACGACCAATATGTGAGCGGGAGCTAAAGCTGATCCGCTGAGAGTGCTACCCTCCAATTATGTTCCGAACTCTTTGTTTAAGTGAAGCCCCAGCGAGTTATTTGCGAGAGAGCTGGACCTGTGGAAGGAGGTTACATAGGTCCTTATGTCGATCGCCTATCACATCAACAGTTGCAGGTCACGTGGTGTGCAGTCGTCCGAGCATCCCGAGAGAGTCAGGATAAAAAACAGACGTTAATGGTCAACACTAGCCGTACTATCAATGACAGTATTAATGGTTAATGGTTATAATATATACAGACCCGTTCCCGGACGCCGTCGGAGCCGGATAGAGGAGTCACTGCCGGGAACAGCGGGCTACGTCGCGATGGTTTCTCGCCCGCCGCCCTAGTGTGTATTTGTCGATTCAGCCTATTGTGCACCCCGCTGACCTCAGCCAAATTCTTGAGTATCGGCGTCCGCGACTCAAGCTTAAAATGCCAGCTATTCAATGGACAATGGCTTGCCGTGAGCGGTGCAACAACGGCGACACCCAAGGTCGCCAATCGGCCCCAACAGCTCCGGGCTAAGGGCTTGATGGTAACTGGTGGATGTCCGTAGAGGTTTATGAAGGATGAGGGACGTTACTTGGACACAGATGACGGAGAGATAGTATTGGTGCGACCTAATCCGCGAGTTGACAGTTGACAGCAGACGCTCGGCACGCTCAGAAATGCGCATCTCGCTCGAAACA'
G, labels = trie_construction(patterns)
indices = trie_matching(text, G, labels)
for j in indices:
	print(j, end = ' ')
print('')
'''
'''				
file = 'trie.txt'
patterns = trie_patterns_input(file)
G, labels = trie_construction(patterns)
trie_contruction_print(G, labels)
'''