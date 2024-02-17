# return ACGT frequencies
def get_dataset_path():
    return os.path.join(os.path.join(os.path.dirname(__file__)),
                        'datasets',
                        os.path.splitext(os.path.split(__file__)[1])[0] + '.txt')


def load_dataset_naive(dataset_path, lines=False, binary=False):
    read_code = 'r'
    if binary:
        read_code += 'b'
    if lines:
        return open(dataset_path, read_code).readlines()
    else:
        return open(dataset_path, read_code).read()


def get_nucleotide_frequencies(dataset):
    valid_results = ('A', 'C', 'G', 'T')
    results_dict = {nucleotide: 0 for nucleotide in valid_results}
    for nucleotide in dataset:
        if nucleotide in valid_results:
            results_dict[nucleotide] += 1
    return ' '.join(map(str, (results_dict[nucleotide] for nucleotide in valid_results)))


if __name__ == "__main__":
    problem_function = get_nucleotide_frequencies

    sample_dataset = 'AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC'

    assert problem_function(sample_dataset) == "20 12 17 21", "Incorrect answer to sample dataset"

    import os

    dataset_path = get_dataset_path()
    problem_dataset = load_dataset_naive(dataset_path, lines=False, binary=False)

    print(problem_function(problem_dataset))
