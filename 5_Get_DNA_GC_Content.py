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


def get_num_rabbits_after_n_generations(dataset):
    # n = months, k = new pairs per breeding pair per generation
    n, k = map(int, dataset.split(" "))
    # n,k = 33,4
    # INITIAL CONDITIONS
    f1 = 1
    f2 = 1
    f_cache = (1, f2, f1)
    for generation in range(n - 2):
        # UNPACK CACHE
        fn, fn_minus_1, fn_minus_2 = f_cache
        # ADD NEW POPULATION TO Fn
        fn_minus_2 = fn_minus_1  # total pop of ancestors of last generation
        fn_minus_1 = fn  # total pop as of last generation
        fn += fn_minus_2 * k  # new generation pop + total pop as of last generation
        # REBUILD CACHE
        f_cache = (fn, fn_minus_1,
                   fn_minus_2)  # new total pop, total pop as of last generation, total pop of ancestors of last generation

    (fn, _, _) = f_cache
    return fn


def get_dna_gc_content(dna_string):
    gc_count = 0.
    for nucleotide in dna_string:
        if nucleotide in ('G','C'):
            gc_count+=1

    return gc_count/len(dna_string)

def get_highest_dna_gc_content(dataset):
    dna_strings = dataset.split(">")
    dna_string_tuples = (dna_string.strip().split("\n") for dna_string in dna_strings if dna_string != '')
    dna_string_dict = {dna_string_tuple[0]: ''.join(dna_string_tuple[1:]) for dna_string_tuple in dna_string_tuples}
    gc_content_dict = {dna_string_name: get_dna_gc_content(dna_string) for dna_string_name, dna_string in
                       dna_string_dict.items()}
    max_gc_string = None
    max_gc = None
    for dna_string_name in gc_content_dict.keys():
        if max_gc_string is None:
            max_gc_string = dna_string_name
            max_gc = gc_content_dict[dna_string_name]
        else:
            if gc_content_dict[dna_string_name]>max_gc:
                max_gc = gc_content_dict[dna_string_name]
                max_gc_string = dna_string_name

    return '\n'.join(map(str,(max_gc_string,round(100*max_gc,5))))


# ----------------------------------------------------------------------------------------------


if __name__ == "__main__":
    PROBLEM_NUMBER = get_problem_number()

    problem_functions = {1: get_nucleotide_frequencies, 2: get_rna_transcription, 3: get_dna_reverse_complement,
                         4: get_num_rabbits_after_n_generations, 5: get_highest_dna_gc_content}

    sample_datasets = {1: 'AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC',
                       2: 'GATGGAACTTGACTACGTAAATT',
                       3: 'AAAACCCGGT', 4: '5 3', 5: """>Rosalind_6404
CCTGCGGAAGATCGGCACTAGAATAGCCAGAACCGTTTCTCTGAGGCTTCCGGCCTTCCC
TCCCACTAATAATTCTGAGG
>Rosalind_5959
CCATCGGTAGCGCATCCTTAGTCCAATTAAGTCCCTATCCAGGCGCTCCGCCGAAGGTCT
ATATCCATTTGTCAGCAGACACGC
>Rosalind_0808
CCACCCTCGTGGTATGGCTAGGCATTCAGGAACCGGAGAACGCTTCAGACCAGCCCGGAC
TGGGAACCTGCGGGCAGTAGGTGGAAT"""}
    sample_answers = {1: "20 12 17 21", 2: 'GAUGGAACUUGACUACGUAAAUU', 3: 'ACCGGGTTTT', 4: 19, 5: """Rosalind_0808
60.91954"""}

    sample_guess = problem_functions[PROBLEM_NUMBER](sample_datasets[PROBLEM_NUMBER])
    # print(sample_answers[PROBLEM_NUMBER])
    # print(sample_guess)
    assert sample_guess == \
           sample_answers[PROBLEM_NUMBER], \
        "Incorrect answer to sample dataset: " + str(sample_guess)

    # ----------------------------------------------------------------------------------------------
    dataset_path = get_dataset_path()
    problem_dataset = load_dataset_naive(dataset_path, lines=False, binary=False)

    print(problem_functions[PROBLEM_NUMBER](problem_dataset))
