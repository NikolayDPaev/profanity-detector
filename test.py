def map_substrings(sentence, substrings):
    result = []

    for substring in substrings:
        index = sentence.find(substring)
        assert(index != -1)
        result.append(sentence[:index + len(substring)])

    return result

# Example usage:
sentence = "This is a sample sentence."
substring_list = ["is", "sample", "sent"]

result_list = map_substrings(sentence, substring_list)
print(result_list)