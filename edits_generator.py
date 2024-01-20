import random

random.seed()

def get_alphabet(char_list)-> str:
	alphabet = set()
	for w in char_list:
		for c in w:
			alphabet.add(c)
	return "".join(alphabet)

def generate_edits(w, alphabet):
	result = []
	for i, c in enumerate(w):
		result.append(w[:i] + w[i + 1:]) # delete
		for l in alphabet:
			result.append(w[:i] + l + w[i:]) # insert
			if c != l:
				result.append(w[:i] + l + w[i + 1:]) # replace

	for l in alphabet: # insert in the end
		result.append(w + l)

	i = 0
	while i + 2 <= len(w): # flip all bigrams
		result.append(w[:i] + w[i+1] + w[i] + w[i+2:])
		i += 1

	return [edit for edit in result if len(result) > 0]

def generate_k_random_candidates(w, k, alphabet) -> list[str]:
	candidates = []
	for e in generate_edits(w, alphabet):
		candidates += generate_edits(e, alphabet)
	return random.choices(candidates, k=k)

def generate_edit_efficient(w, alphabet):
	operation = random.randint(1, 5)
	if len(w) < 2:
		operation = random.randint(1, 4)

	if operation == 1: # delete
		i = random.randint(0, len(w)-1)
		return w[:i] + w[i + 1:]

	elif operation == 2: # insert
		l = alphabet[random.randint(0, len(alphabet)-1)]
		i = random.randint(0, len(w))
		if i == len(w):
			return w + l
		return w[:i] + l + w[i:]

	elif operation == 3: # replace
		l = alphabet[random.randint(0, len(alphabet)-1)]
		i = random.randint(0, len(w)-1)
		return w[:i] + l + w[i+1:]

	elif operation == 4: # multiply
		i = random.randint(0, len(w)-1)
		m = random.randint(1, len(w))
		return w[:i] + m*w[i] + w[i+1:]

	# operation 5 - flip
	i = random.randint(0, len(w) - 2)
	return w[:i] + w[i+1] + w[i] + w[i+2:]

def generate_k_random_candidates_efficient(w, k, alphabet) -> list[str]:
	candidates = []
	for _ in range(k):
		first = generate_edit_efficient(w, alphabet)
		if len(first) > 2:
			candidates.append(generate_edit_efficient(first, alphabet))
		else:
			candidates.append(first)
	return candidates