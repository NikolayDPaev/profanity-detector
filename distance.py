from collections import defaultdict, Counter
import numpy as np

def levenshteinDistance(s1, s2):
    M = np.zeros((len(s1)+1,len(s2)+1))
    len_s1, len_s2 = len(s1), len(s2)

    for i in range(len_s1 + 1): M[i, 0] = i
    for j in range(len_s2 + 1): M[0, j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            M[i, j] = min(
                M[i-1, j] + 1,
                M[i, j-1] + 1,
                M[i-1, j-1] if s1[i-1] == s2[j-1] else M[i-1, j-1] + 1
            )
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                M[i, j] = min(M[i, j], M[i-2, j-2] + 1)

    return M[-1, -1]

class Edit_suggester:
    def __init__(self, n, dictionary):
        self.dictionary = dictionary
        self.postings = defaultdict(lambda: [])
        self.n = n
        for token in self.dictionary:
            for ngram in self.get_ngrams(token):
                self.postings[ngram].append(token)

    def get_ngrams(self, token):
        ngrams = []
        i = 0
        while i + self.n <= len(token):
            ngrams.append(token[i:i+self.n])
            i += 1
        return ngrams

    def ngram_spellcheck(self, word, jaccard_threshold=0.7) -> str | None:
        """Return the word from the postings structure, which is closest to the input word."""
        union_of_postings = sum([self.postings[ngram] for ngram in self.get_ngrams(word)], [])

        suggestions = []
        for suggested_word, occurences in Counter(union_of_postings).most_common():
            denominator = len(suggested_word) - self.n + len(word) - self.n
            jaccard_coeff = occurences / denominator if denominator != 0 else 0

            if jaccard_coeff > jaccard_threshold:
                suggestions.append(suggested_word)

        if len(suggestions) == 0:
            return None

        return suggestions[np.argmin([levenshteinDistance(x, word) for x in suggestions])]

    def ngram_all_closest_words(self, word, jaccard_threshold=0.7, tolerance=0) -> list[str]:
        """Return n closest words to the word."""
        union_of_postings = sum([self.postings[ngram] for ngram in self.get_ngrams(word)], [])

        suggestions = []
        for suggested_word, occurences in Counter(union_of_postings).most_common():
            denominator = len(suggested_word) - self.n + len(word) - self.n
            jaccard_coeff = occurences / denominator if denominator != 0 else 0

            if jaccard_coeff > jaccard_threshold:
                suggestions.append(suggested_word)

        suggestions_distance = list(map(lambda x: (x, levenshteinDistance(x, word)), suggestions))
        minimum_distance = min(map(lambda x: x[1], suggestions_distance))

        return [suggestion_distance[0]
                    for suggestion_distance in suggestions_distance
                        if suggestion_distance[1] <= minimum_distance + tolerance
        ]
