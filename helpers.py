import random


def generate_unique_index(index_range, already_generated):
    while True:
        i = random.randint(*index_range)

        if i not in already_generated:
            return i
