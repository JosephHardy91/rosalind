import os, re
import numpy as np
import math

import requests
from tqdm import tqdm


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


def get_num_mortal_rabbits_after_n_generations(dataset):
    # n = months, k = new pairs per breeding pair per generation
    n, m = map(int, dataset.split(" "))
    immature_rabbits = 1
    # mature_rabbits = 0
    # dying_rabbits = 0
    generations = [0 for _ in range(m)]
    generations[-1] = immature_rabbits
    total_rabbits = sum(generations)
    for generation in range(n - 1):
        new_rabbits = sum(generations[:-1])  # everything but the immature rabbits can breed
        generations[:-1] = generations[1:]  # shift generations to the left (to remove dying generation)
        generations[-1] = new_rabbits  # add new rabbits as the latest generation
        total_rabbits = sum(generations)

    return total_rabbits


def get_dna_gc_content(dna_string):
    gc_count = 0.
    for nucleotide in dna_string:
        if nucleotide in ('G', 'C'):
            gc_count += 1

    return gc_count / len(dna_string)


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
            if gc_content_dict[dna_string_name] > max_gc:
                max_gc = gc_content_dict[dna_string_name]
                max_gc_string = dna_string_name

    return '\n'.join(map(str, (max_gc_string, round(100 * max_gc, 5))))


def get_hamming_distance(dataset):  # dH(s,t) - hamming distance to detect point mutations in genomes
    s, t = dataset.strip().split("\n")
    hamming_distance = 0
    for nucleotide1, nucleotide2 in zip(s, t):
        if nucleotide1 != nucleotide2:
            hamming_distance += 1

    return hamming_distance


def dom_allele_chance(org1_type, org2_type):
    if org1_type == 'rec' and org2_type == 'rec':
        return 0.
    elif org1_type == 'hetero' and org2_type == 'hetero':
        return 0.75
    elif (org1_type == 'rec' and org2_type == 'hetero') or (org1_type == 'hetero' and org2_type == 'rec'):
        return 0.5
    else:
        return 1.


def get_average_allele_dominance_chance_one_generation(dataset):
    k, m, n = map(int, dataset.strip().split(" "))
    total_pop = float(k + m + n)
    pops = {'dom': k, 'hetero': m, 'rec': n}
    dominant_allele_chances = {'dom': [k / total_pop], 'hetero': [m / total_pop], 'rec': [n / total_pop]}
    for k_ in dominant_allele_chances.keys():
        k_pop = pops[k_] - 1
        total_pop_minus_k = total_pop - 1.
        dominant_allele_chances[k_].append({k2: dominant_allele_chances[k_][0] *
                                                pops[k2] / total_pop_minus_k *
                                                dom_allele_chance(k_, k2) if k_ != k2 else dominant_allele_chances[k_][
                                                                                               0] *
                                                                                           k_pop / total_pop_minus_k *
                                                                                           dom_allele_chance(k_, k2) for
                                            k2 in dominant_allele_chances.keys()})
    all_chances = [dominant_allele_chances[k][-1][k2] for k in dominant_allele_chances.keys() for k2 in
                   dominant_allele_chances[k][-1]]
    return round(sum(all_chances), 5)


def get_codon_to_amino_acid_dict():
    codon_table = """UUU F      CUU L      AUU I      GUU V
    UUC F      CUC L      AUC I      GUC V
    UUA L      CUA L      AUA I      GUA V
    UUG L      CUG L      AUG M      GUG V
    UCU S      CCU P      ACU T      GCU A
    UCC S      CCC P      ACC T      GCC A
    UCA S      CCA P      ACA T      GCA A
    UCG S      CCG P      ACG T      GCG A
    UAU Y      CAU H      AAU N      GAU D
    UAC Y      CAC H      AAC N      GAC D
    UAA Stop   CAA Q      AAA K      GAA E
    UAG Stop   CAG Q      AAG K      GAG E
    UGU C      CGU R      AGU S      GGU G
    UGC C      CGC R      AGC S      GGC G
    UGA Stop   CGA R      AGA R      GGA G
    UGG W      CGG R      AGG R      GGG G """
    codon_amino_pair_tuples = (codon_amino_pair.strip().split(" ") for codon_table_line in codon_table.split("\n") for
                               codon_amino_pair in re.split(" \s+", codon_table_line) if codon_amino_pair.strip() != '')
    return dict(codon_amino_pair_tuples)


def get_amino_acids_from_mRNA(dataset):
    protein_string = dataset.strip()
    # translate codons (length 3 RNA strings [3 nucleotides]) to amino acids
    codons = [protein_string[i:i + 3] for i in range(0, len(protein_string), 3)]
    codon_to_amino_acid_dict = get_codon_to_amino_acid_dict()
    return ''.join(codon_to_amino_acid_dict[codon] for codon in codons if codon_to_amino_acid_dict[codon] != 'Stop')


def get_substring_positions(string, substring):
    substring_len = len(substring)
    positions = []
    for i in range(len(string) - substring_len + 1):
        string_substring = string[i:i + substring_len]
        if string_substring == substring:
            positions.append(i + 1)

    return positions


def get_dna_substring_positions(dataset):
    dna_string, substring = dataset.strip().split("\n")

    positions = get_substring_positions(dna_string, substring)
    return ' '.join(map(str, positions))


def get_profile_matrix_index_dict(inverse=False):
    profile_matrix_order = ('A', 'C', 'G', 'T')
    profile_matrix_indices = (0, 1, 2, 3)
    if not inverse:
        profile_matrix_index_dict = dict(zip(profile_matrix_order, profile_matrix_indices))
    else:
        profile_matrix_index_dict = dict(zip(profile_matrix_indices, profile_matrix_order))
    return profile_matrix_index_dict


def get_dna_profile_matrix(dataset):
    # print(dataset[:20])
    dna_strings_with_ids = dataset.strip().split(">")
    dna_strings_only = [dna_string_pair.split("\n", 1)[1].replace("\n", "") for dna_string_pair in dna_strings_with_ids
                        if
                        dna_string_pair != '']
    profile_matrix_index_dict = get_profile_matrix_index_dict()
    profile_matrix = np.zeros((4, len(dna_strings_only[0])), dtype=np.int8)
    for dna_string in dna_strings_only:
        indices = range(len(dna_string))
        chars = list(dna_string)
        for i, c in zip(indices, chars):
            profile_matrix[profile_matrix_index_dict[c], i] += 1
    return profile_matrix


def get_dna_profile_consensus(profile_matrix):
    profile_matrix_index_dict = get_profile_matrix_index_dict(inverse=True)
    consensus_indices = np.argmax(profile_matrix, axis=0)
    return ''.join(profile_matrix_index_dict[i] for i in consensus_indices)


def get_dna_profile_and_consensus(dataset):
    profile_matrix = get_dna_profile_matrix(dataset)
    consensus_string = get_dna_profile_consensus(profile_matrix)
    profile_matrix_order = ('A', 'C', 'G', 'T')
    profile_matrix_pretty_print = '\n'.join(
        profile_matrix_order[i] + ': ' + ' '.join(map(str, profile_matrix[i])) for i in range(4))
    return '\n'.join((consensus_string, profile_matrix_pretty_print))


def get_overlap_graph(dataset, k=3):
    dna_strings_with_ids = dataset.strip().split(">")
    dna_ids_only = [dna_string_pair.split("\n", 1)[0].replace("\n", "") for dna_string_pair in dna_strings_with_ids
                    if
                    dna_string_pair != '']
    dna_strings_only = [dna_string_pair.split("\n", 1)[1].replace("\n", "") for dna_string_pair in dna_strings_with_ids
                        if
                        dna_string_pair != '']
    adjacency_indices = []
    for i, dna_string in zip(dna_ids_only, dna_strings_only):
        for i2, dna_string2 in zip(dna_ids_only, dna_strings_only):
            if dna_string == dna_string2: continue
            string_suffix = dna_string[-k:]
            string2_prefix = dna_string2[:k]
            if string_suffix == string2_prefix:
                adjacency_indices.append((i, i2))
    return '\n'.join(' '.join((t[0], t[1])) for t in adjacency_indices)


def get_expected_dominant_phenotype_offspring(dataset):
    """AA-AA
        AA-Aa
        AA-aa
        Aa-Aa
        Aa-aa
        aa-aa; every couple has exactly two offspring"""
    # dataset has number of couples for the following combinations of phenotypes
    dom_dom, dom_hetero, dom_rec, hetero_hetero, hetero_rec, rec_rec = map(int, dataset.strip().split(" "))
    dd, dh, dr, hh, hr, rr = 1, 1, 1, 0.75, 0.5, 0
    return 2 * dom_dom * dd + 2 * dom_hetero * dh + 2 * dom_rec * dr + 2 * hetero_hetero * hh + 2 * hetero_rec * hr + 2 * rec_rec * rr


# SLOW, takes 90 seconds for Rosalind "Finding A Shared Motif" dataset, 1-1.15it/s for each string to check against
def get_longest_common_dna_substring(dataset):
    # GET DNA STRINGS FROM DATASET
    dna_ids_and_strings = dataset.strip().split(">")
    dna_strings = [d.split("\n", 1)[1].strip().replace("\n", "") for d in dna_ids_and_strings if d != '']

    # GET SUBSTRINGS OF SHORTEST STRING
    shortest_string = min(dna_strings, key=len)
    shortest_string_i = dna_strings.index(shortest_string)
    possible_common_substrings_and_lens = [(dna_strings[shortest_string_i][i:i + substr_len], substr_len) for substr_len
                                           in
                                           range(2, len(dna_strings[shortest_string_i])) for i in
                                           range(2, len(dna_strings[shortest_string_i]) - substr_len + 1)]
    possible_common_substrings, possible_common_substring_lens = zip(*possible_common_substrings_and_lens)

    # CHECK OTHER STRINGS FOR SUBSTRINGS FROM SHORTEST STRINGS
    substr_presences = np.zeros((len(dna_strings), len(possible_common_substrings)))
    substr_presences[shortest_string_i] = 1
    for s_i, dna_string in tqdm(enumerate(dna_strings), total=len(dna_strings)):
        if s_i == shortest_string_i: continue
        for i, substr in enumerate(possible_common_substrings):
            if substr in dna_string:
                substr_presences[s_i, i] = 1  # possible_common_substring_lens[i]

    # FILTER TO SUBSTRINGS THAT APPEAR IN ALL STRINGS
    substr_presence = np.count_nonzero(substr_presences, axis=0)
    substr_presence[substr_presence != len(dna_strings)] = 0

    # GET LONGEST COMMON SUBSTRINGS
    longest_common_substring_lens = np.array(possible_common_substring_lens)
    longest_common_substring_lens[substr_presence == 0] = 0
    lcs_indices = np.argwhere(
        longest_common_substring_lens == np.amax(longest_common_substring_lens)).flatten().tolist()
    # print(lcs_indices)
    lcs_instances = [possible_common_substrings[i] for i in lcs_indices]

    # RETURN ALL SHARED MOTIFS (LONGEST COMMON SUBSTRINGS)
    return lcs_instances

    # longest_common_substrings = []
    # longest_common_substring_len = 0
    # first_string = True
    # dna_string = dna_strings[0]
    # for dna_string2 in dna_strings[1:]:
    #     found_common_substring =
    #     if not first_string and len(longest_common_substrings) == 0:
    #         return ''
    #     if dna_string!=dna_string2:
    #         if len(longest_common_substrings)==0
    #         for substr_length in range(len(dna_string),longest_common_substring_len,-1):
    #             for dna_string_i in range(len(dna_string)-substr_length+1):
    #                 substr = dna_string[dna_string_i:dna_string_i+substr_length]
    #                 for dna_string2_i in range(len(dna_string2) - substr_length + 1):
    #                     if substr==dna_string2[dna_string2_i:dna_string2_i+substr_length]:
    #                         if longest_common_substring_len<len(substr):
    #                             longest_common_substring_len =
    #     first_string = False


def punnet_square_probs(p1, p2):
    prob_dict = {}
    for allele_part_s in p1:
        for allele_part_t in p2:
            new_allele = ''.join(sorted(allele_part_s + allele_part_t))
            if new_allele not in prob_dict:
                prob_dict[new_allele] = 0.
            prob_dict[new_allele] += 1
    for allele in prob_dict:
        prob_dict[allele] /= 4
    return prob_dict


def get_allele_probs(a1, a2):
    if a1 == ('AA', 'BB') and a2 == ('Aa', 'Bb'):
        print((a1, a2))
    # print((a1,a2))
    allele1_part_s, allele1_part_t = a1
    allele2_part_s, allele2_part_t = a2
    part_s_probs = punnet_square_probs(allele1_part_s, allele2_part_s)
    part_t_probs = punnet_square_probs(allele1_part_t, allele2_part_t)
    prob_dict = {}
    for part_s_allele in part_s_probs.keys():
        for part_t_allele in part_t_probs.keys():
            new_genotype = (part_s_allele, part_t_allele)
            if new_genotype not in prob_dict:
                prob_dict[new_genotype] = 0
            prob_dict[new_genotype] += part_s_probs[part_s_allele] * part_t_probs[part_t_allele]
    if a1 == ('AA', 'BB') and a2 == ('Aa', 'Bb'):
        print(prob_dict)
        # looks right, don't know what I'm doing wrong
    return prob_dict


def add_dict_values(list_of_dicts):
    new_dict = {}
    for dict in list_of_dicts:
        for k in dict:
            if k not in new_dict:
                new_dict[k] = 0
            new_dict[k] += dict[k]
    return new_dict


def scalar_multiply_dict(dict, scalar):
    for k in dict:
        dict[k] *= scalar

    return dict


def get_num_of_retained_alleles_at_kth_generation(dataset):
    k, N = map(int, dataset.strip().split(" "))
    print(k, N)
    # binomial probability from binomial distribution
    total_pop_at_k = 2 ** k
    bp = 0
    for n in range(N, total_pop_at_k + 1):
        nCx = math.factorial(total_pop_at_k) / (math.factorial(n) * math.factorial(total_pop_at_k - n))
        bp += nCx * 0.25 ** n * (1 - 0.25) ** (
                total_pop_at_k - n)  # 0.25 is the independent probability of producing an AaBb offspring from two AaBb offspring parents
    # start_alleles = ('Aa', 'Bb')
    # allele_probs = get_allele_probs(start_alleles, start_alleles)
    # print(list(sorted(allele_probs.items(),key=lambda kv:kv[0])))
    # for g in range(k):
    #     print(g)
    #     allele_probs = add_dict_values(
    #         scalar_multiply_dict(get_allele_probs(new_allele, start_alleles), allele_probs[new_allele]) for new_allele in allele_probs)
    #     print(list(sorted(allele_probs.items(),key=lambda kv:kv[0])))
    return bp

    # ----------------------------------------------------------------------------------------------


def get_protein_motifs(dataset_references):
    protein_sequences = {}
    # motif = "N{P}[ST]{P}"
    regex_motif = re.compile("(?=N[^P][S|T][^P])")
    sequence_order = []
    for UniProtID in dataset_references.split("\n"):
        clean_UniProtID = UniProtID.strip().split("_", 1)[0]
        url = 'http://www.uniprot.org/uniprot/%s.fasta' % clean_UniProtID
        res = requests.get(url).text.split("\n", 1)
        if res == ['']:
            parc_url = 'https://rest.uniprot.org/unisave/%s?format=fasta' % clean_UniProtID  # 'http://www.uniprot.org/uniparc/%s.fasta' % clean_UniProtID
            res = requests.get(parc_url).text.split("\n", 1)
        protein_sequences[UniProtID.strip()] = ''.join(res[1].split("\n"))
        sequence_order.append(UniProtID.strip())
        # if UniProtID.strip()=='P07204_TRBM_HUMAN':
        #     print(protein_sequences[UniProtID.strip()])
        # if needed-----
        # parc_url = 'http://www.uniprot.org/uniprotkb/%s.fasta' % clean_UniProtID
        # if motif:
        # result = ''.join(requests.get(url).text.split("\n", 1)[1].split("\n"))
        # if result=='':
        #     result = ''.join(requests.get(parc_url).text.split("\n", 1)[1].split("\n"))
        # else:
        #     motif = ''.join(requests.get(url).text.split("\n", 1)[1].split("\n"))
        # ------------

    motif_positions = {}
    for sequence in protein_sequences:
        motif_positions[sequence] = []
        for match in regex_motif.finditer(protein_sequences[sequence]):
            motif_positions[sequence].append(match.start() + 1)
        if motif_positions[sequence] == []:
            sequence_order.remove(sequence)

    return '\n'.join(
        ('\n'.join((sequence, ' '.join(map(str, motif_positions[sequence])))) for sequence in sequence_order))


# in modular arithmetic, if a mod n = b mod n and c mod n = d mod n,
# then (a+c) mod n = (b+d) mod n and (a*c) mod n = (b*d) mod n
def check_modular_arithmetic_assertions(a=29, b=73, c=10, d=32, n=11):
    assert a % n == b % n, "a mod n != b mod n"
    assert c % n == d % n, "c mod n != d mod n"
    assert (a + c) % n == (b + d) % n, "(a+c) mod n!= (b+d) mod n"
    assert (b + c) % n == (a + d) % n, "(b+c) mod n!= (a+d) mod n"
    assert (a * c) % n == (b * d) % n, "(a*c) mod n!= (b*d) mod n"
    assert (b * c) % n == (a * d) % n, "(b*c) mod n!= (a*d) mod n"


# check_modular_arithmetic_assertions()

def flip_amino_acid_dict():
    codon_to_amino_acid_dict = get_codon_to_amino_acid_dict()
    amino_acid_to_codon_dict = {}
    for codon in codon_to_amino_acid_dict:
        amino = codon_to_amino_acid_dict[codon]
        # print(aminos,codon)
        if amino not in amino_acid_to_codon_dict:
            amino_acid_to_codon_dict[amino] = []
        amino_acid_to_codon_dict[amino].append(codon)
    # print(amino_acid_to_codon_dict)
    return amino_acid_to_codon_dict


def get_num_possible_RNA_from_protein(dataset):
    protein_string = ''.join(dataset.split("\n"))
    amino_acid_to_codon_dict = flip_amino_acid_dict()
    amino_acid_to_codon_possibilities_dict = {amino_acid: len(codons) for amino_acid, codons in
                                              amino_acid_to_codon_dict.items()}
    possibilities = 1
    for amino_acid in protein_string:
        amino_possibilities = amino_acid_to_codon_possibilities_dict[amino_acid]
        possibilities *= amino_possibilities
    return (possibilities * amino_acid_to_codon_possibilities_dict['Stop']) % 1000000


def get_proteins_from_ORF(dataset):
    orfs_and_ids = (orf.strip().split("\n") for orf in dataset.split(">"))
    orfs_and_ids = [orf_and_id for orf_and_id in orfs_and_ids if orf_and_id != ['']]
    orfs_and_ids = [(orf_and_id[0], ''.join(orf_and_id[1:])) for orf_and_id in orfs_and_ids]
    orfs = [get_rna_transcription(orf) for id, orf in orfs_and_ids]
    reverse_complements = [get_rna_transcription(get_dna_reverse_complement(orf)) for id, orf in orfs_and_ids]
    orfs = reverse_complements + orfs
    codon_to_amino_acid_dict = get_codon_to_amino_acid_dict()
    stop_codons = [codon for codon in codon_to_amino_acid_dict if codon_to_amino_acid_dict[codon] == 'Stop']
    candidates = []
    for orf in orfs:
        starts = []
        stops = []
        for i in range(0, len(orf)):
            triplet = orf[i:i + 3]
            if triplet == 'AUG':
                starts.append(i)
            elif triplet in stop_codons:
                stops.append(i)
        for start in starts:
            for stop in stops:
                candidate_len = stop - start
                candidate = []
                if not candidate_len > 3 or candidate_len % 3 != 0:
                    continue
                for i in range(start, stop + 1, 3):
                    triplet = orf[i:i + 3]
                    amino = codon_to_amino_acid_dict[triplet]
                    if amino == 'Stop':
                        break
                    else:
                        candidate.append(amino)
                candidate_str = ''.join(candidate)
                if candidate_str not in candidates:
                    candidates.append(candidate_str)

    return '\n'.join(candidates)


# my solution - works but inefficient (with replacement so have to trash replacements)
def my_get_num_and_permutations(dataset):
    n = int(dataset.strip())
    permutations = []
    options = range(1, n + 1)
    for i in range(pow(n, n)):
        permutation = []
        for i2 in range(n):
            c = options[i % n]
            if c not in permutation:
                permutation.append(c)
            i //= n
            if permutation not in permutations and len(permutation) == n:
                permutations.append(permutation)

    permutations = [' '.join(map(str, permutation)) for permutation in sorted(permutations)]
    return '\n'.join([str(len(permutations)), '\n'.join(permutations)])


# Rayan 9/22/2012 - rosalind most upvoted (+30 as of 10/15/2022)
def get_permutations(l):
    permutations = []
    if len(l) > 1:
        for m in get_permutations(l[1:]):
            for i in range(len(m) + 1):
                permutations.append(m[:i] + [l[0]] + m[i:])
        return permutations
    else:
        return [l]


def get_num_and_permutations(dataset):
    n = int(dataset.strip())
    permutations = [' '.join(map(str, permutation)) for permutation in sorted(get_permutations(list(range(1, n + 1))))]
    return '\n'.join([str(len(permutations)), '\n'.join(permutations)])


def get_amino_acid_residue_monoisotopic_mass_table():
    return {residue.strip(): float(val.strip()) for residue, val in [p.split(" ", 1) for p in """A   71.03711
C   103.00919
D   115.02694
E   129.04259
F   147.06841
G   57.02146
H   137.05891
I   113.08406
K   128.09496
L   113.08406
M   131.04049
N   114.04293
P   97.05276
Q   128.05858
R   156.10111
S   87.03203
T   101.04768
V   99.06841
W   186.07931
Y   163.06333""".split("\n")]}


def get_protein_string_mass(dataset):
    residue_to_mass_dict = get_amino_acid_residue_monoisotopic_mass_table()
    mass = 0
    for char in dataset.strip():
        mass += residue_to_mass_dict[char]

    return round(mass, 3)


def find_reverse_palindromes(dataset):
    dna_string = ''.join(dataset.split("\n", 1)[1]).replace("\n", "")
    positions_and_lengths = []
    for length in range(4, min(len(dna_string) + 1, 12 + 1)):
        for candidate_start in range(0, len(dna_string) + 1 - length):
            dna_candidate = dna_string[candidate_start:candidate_start + length]
            reverse_complement_candidate = get_dna_reverse_complement(dna_candidate)
            if dna_candidate == reverse_complement_candidate:
                positions_and_lengths.append((candidate_start + 1, length))
    ans = '\n'.join(' '.join(map(str, pos_and_length)) for pos_and_length in sorted(positions_and_lengths))
    return ans


def splice_and_transcribe_pre_mRNA_made_from_DNA(dataset):
    dna_strings = [p.split("\n", 1)[1].replace("\n", "") for p in dataset.split(">") if p != '']
    dna_string = dna_strings[0]
    introns = sorted(dna_strings[1:], key=len)
    for intron in introns:
        dna_string = dna_string.replace(intron, "")

    mRNA_string = get_rna_transcription(dna_string)

    return get_amino_acids_from_mRNA(mRNA_string)


if __name__ == "__main__":
    PROBLEM_NUMBER = get_problem_number()

    problem_functions = {1: get_nucleotide_frequencies, 2: get_rna_transcription, 3: get_dna_reverse_complement,
                         4: get_num_rabbits_after_n_generations, 5: get_highest_dna_gc_content,
                         6: get_hamming_distance,
                         7: get_average_allele_dominance_chance_one_generation, 8: get_amino_acids_from_mRNA,
                         9: get_dna_substring_positions, 10: get_dna_profile_and_consensus,
                         11: get_num_mortal_rabbits_after_n_generations, 12: get_overlap_graph,
                         13: get_expected_dominant_phenotype_offspring, 14: get_longest_common_dna_substring,
                         15: get_num_of_retained_alleles_at_kth_generation, 16: get_protein_motifs,
                         17: get_num_possible_RNA_from_protein, 18: get_proteins_from_ORF, 19: get_num_and_permutations,
                         20: get_protein_string_mass, 21: find_reverse_palindromes,
                         22: splice_and_transcribe_pre_mRNA_made_from_DNA}

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
TGGGAACCTGCGGGCAGTAGGTGGAAT""", 6: """GAGCCTACTAACGGGAT
CATCGTAATGACGGCCT""",
                       7: '2 2 2', 8: 'AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA',
                       9: """GATATATGCATATACTT
ATAT""", 10: """>Rosalind_1
ATCCAGCT
>Rosalind_2
GGGCAACT
>Rosalind_3
ATGGATCT
>Rosalind_4
AAGCAACC
>Rosalind_5
TTGGAACT
>Rosalind_6
ATGCCATT
>Rosalind_7
ATGGCACT""", 11: '6 3', 12: """>Rosalind_0498
AAATAAA
>Rosalind_2391
AAATTTT
>Rosalind_2323
TTTTCCC
>Rosalind_0442
AAATCCC
>Rosalind_5013
GGGTGGG""", 13: '1 0 0 1 0 1', 14: """>Rosalind_1
GATTACA
>Rosalind_2
TAGACCA
>Rosalind_3
ATACA""", 15: '2 1',
                       16: """A2Z669
B5ZC00
P07204_TRBM_HUMAN
P20840_SAG1_YEAST""", 17: 'MA', 18: """>Rosalind_99
AGCCATGTAGCTAACTCAGGTTACATGGGGATGACCCCGCGACTTGGATTAGAGTCTCTTTTGGAATAAGCCTGAATGATCCGAGTAGCATCTCAG""", 19: '3',
                       20: 'SKADYEK', 21: """>Rosalind_24
TCAATGCATGCGGGTCTATATGCAT""", 22: """>Rosalind_10
ATGGTCTACATAGCTGACAAACAGCACGTAGCAATCGGTCGAATCTCGAGAGGCATATGGTCACATGATCGGTCGAGCGTGTTTCAAAGTTTGCGCCTAG
>Rosalind_12
ATCGGTCGAA
>Rosalind_15
ATCGGTCGAGCGTGT"""}
    sample_answers = {1: "20 12 17 21", 2: 'GAUGGAACUUGACUACGUAAAUU', 3: 'ACCGGGTTTT', 4: 19, 5: """Rosalind_0808
60.91954""", 6: 7, 7: 0.78333,
                      8: 'MAMAPRTEINSTRING', 9: '2 4 10', 10: """ATGCAACT
A: 5 1 0 0 5 5 0 0
C: 0 0 1 4 2 0 6 1
G: 1 1 6 3 0 1 0 0
T: 1 5 0 0 0 1 1 6""", 11: 4, 12: """Rosalind_0498 Rosalind_2391
Rosalind_0498 Rosalind_0442
Rosalind_2391 Rosalind_2323""",
                      13: 3.5, 14: ['AC', 'CA'], 15: 0.68359375, 16: """B5ZC00
85 118 142 306 395
P07204_TRBM_HUMAN
47 115 116 382 409
P20840_SAG1_YEAST
79 109 135 248 306 348 364 402 485 501 614""", 17: 12, 18: """MLLGSFRLIPKETLIQVAGSSPCNLS
M
MGMTPRLGLESLLE
MTPRLGLESLLE""", 19: """6
1 2 3
1 3 2
2 1 3
2 3 1
3 1 2
3 2 1""", 20: 821.392, 21: """4 6
5 4
6 6
7 4
17 4
18 4
20 6
21 4""",
                      22: "MVYIADKQHVASREAYGHMFKVCA"}

    sample_guess = problem_functions[PROBLEM_NUMBER](sample_datasets[PROBLEM_NUMBER])
    # print(sample_answers[PROBLEM_NUMBER])
    # print(sample_guess)
    assert sample_guess == \
           sample_answers[PROBLEM_NUMBER], \
        "Incorrect answer to sample dataset: " + str(sample_guess) + '\n--------\n' + str(
            sample_answers[PROBLEM_NUMBER])

    # ----------------------------------------------------------------------------------------------
    dataset_path = get_dataset_path()
    problem_dataset = load_dataset_naive(dataset_path, lines=False, binary=False)

    print(problem_functions[PROBLEM_NUMBER](problem_dataset))
