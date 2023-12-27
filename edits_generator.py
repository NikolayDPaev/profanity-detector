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