import random

def mask_word(word, mask_percentage):
    """Mask the given percentage of letters in the word."""
    letter_indices = {}
    for index, letter in enumerate(word):
        if letter not in letter_indices:
            letter_indices[letter] = []
        letter_indices[letter].append(index)

    unique_letters = list(letter_indices.keys())
    num_to_mask = max(1, int(len(unique_letters) * mask_percentage / 100))
    letters_to_mask = random.sample(unique_letters, num_to_mask)

    masked_word = list(word)
    for letter in letters_to_mask:
        for index in letter_indices[letter]:
            masked_word[index] = '#'

    return ''.join(masked_word)

def process_file(input_file, output_file):
    """Process each word in the input file and write masked versions to the output file."""
    with open(input_file, 'r') as f:
        words = f.read().splitlines()

    with open(output_file, 'w') as f:
        for word in words:
            for mask_percentage in [20, 40, 60, 80, 100]:
                masked_word = mask_word(word, mask_percentage)
                f.write(f"{masked_word},{word}\n")

input_file = 'train.txt'
output_file = 'train_aug_masked.txt'
process_file(input_file, output_file)

input_file = 'test.txt'
output_file = 'test_aug_masked.txt'
process_file(input_file, output_file)