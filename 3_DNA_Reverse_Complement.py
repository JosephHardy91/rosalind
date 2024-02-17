# return ACGT frequencies
import os


def get_dataset_path():
    return os.path.join(os.path.join(os.path.dirname(__file__)),
                        'datasets',
                        os.path.splitext(os.path.split(__file__)[1])[0] + '.txt')


def get_problem_number():
    return int(os.path.splitext(os.path.split(__file__)[1])[0].split("_", 1)[0])


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


def get_rna_transcription(dataset):
    return dataset.replace('T', 'U')


def get_dna_reverse_complement(dataset):
    valid_results = ('A', 'C', 'G', 'T')
    reversed_dataset = dataset[::-1]
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement_dict[nucleotide] for nucleotide in reversed_dataset if nucleotide in valid_results)


# ----------------------------------------------------------------------------------------------


if __name__ == "__main__":
    PROBLEM_NUMBER = get_problem_number()

    problem_functions = {1: get_nucleotide_frequencies, 2: get_rna_transcription, 3:get_dna_reverse_complement}

    sample_datasets = {1: 'AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC',
                       2: 'GATGGAACTTGACTACGTAAATT',
                       3:'AAAACCCGGT'}
    sample_answers = {1: "20 12 17 21", 2: 'GAUGGAACUUGACUACGUAAAUU',3:'ACCGGGTTTT'}

    assert problem_functions[PROBLEM_NUMBER](sample_datasets[PROBLEM_NUMBER]) == \
           sample_answers[PROBLEM_NUMBER], \
        "Incorrect answer to sample dataset"
    # ----------------------------------------------------------------------------------------------
    dataset_path = get_dataset_path()
    problem_dataset = load_dataset_naive(dataset_path, lines=False, binary=False)


    print(problem_functions[PROBLEM_NUMBER](problem_dataset))
