"""
Info:
    DNA sequences are made up of the characters A, C, G, and T,
    representing nucleotides. For example, a DNA string might be "ACCGTTTAAAG."

    The subsequence does not need to be continuous, and empty strings do not match anything.

Example:
    TAGTTCCGTCAAA
    GTGTTCCCGTCAAA

    Answer - TGTTCCGTCAAA = 12

Analysis:
    DNA sequences can be different lengths
    They don't need to be contiguous
    A subsequence of a sequence is just the given sequence with 0 or more elements left out
    
State:
    DNA1, DNA2
    i1, i2

Transitions:
    if dna1_n == dna2_m, then we check the subsequence length at i1 + 1 and i2 + 1
    if dna1_n != dna2_m, then 
        1. We check the subsequence length at i1 and i2 + 1
        or
        2. We check the subsequence length at i1 + 1 and i2
    
    lets start with 1 and then we can try 2

Base Case:
    if dna1_n == 0 or dna2_m == 0, return 0

-------------------BRUTE FORCE-----------------------------------------------
def dna_match_brute(DNA1: str, DNA2: str, i1: int, i2: int) -> int:
    # base case
    if i1 == len(DNA1) or i2 == len(DNA2):
        return 0 
    
    # transitions 
    if DNA1[i1] == DNA2[i2]:
        return 1 + dna_match_brute(DNA1, DNA2, (i1 + 1), (i2 + 1))
    else:
        lcs1 = dna_match_brute(DNA1, DNA2, i1, (i2 + 1))
        lcs2 = dna_match_brute(DNA1, DNA2, (i1 + 1), i2)
        return max(lcs1, lcs2)
-----------------------------------------------------------------------------
"""


def dna_match_topdown(DNA1, DNA2):
    cache = [[-1]*len(DNA2) for _n in range(len(DNA1))]
    return dna_match_topdown_helper(DNA1, DNA2, 0, 0, cache)

def dna_match_bottomup(DNA1, DNA2):
    cache = [[0]*(len(DNA2) + 1) for _n in range(len(DNA1) + 1)]
    return dna_match_bottomup_helper(DNA1, DNA2, cache)

def dna_match_topdown_helper(DNA1, DNA2, i1, i2, cache):
    """Running Time: O(len(DNA1)*len(DNA2))"""
    # Base Case
    if i1 == len(DNA1) or i2 == len(DNA2):
        return 0
    
    # We've already computed the length of this LCS 
    if cache[i1][i2] != -1:
        return cache[i1][i2]

    if DNA1[i1] == DNA2[i2]:
        cache[i1][i2] = 1 + dna_match_topdown_helper(DNA1, DNA2, (i1+1), (i2+1), cache)
    else:
        lcs1 = dna_match_topdown_helper(DNA1, DNA2, i1, (i2+1), cache)
        lcs2 = dna_match_topdown_helper(DNA1, DNA2, (i1+1), i2, cache)
        cache[i1][i2] = max(lcs1, lcs2)
    
    return cache[i1][i2]

"""
State:
    i1
    i2
Bounds (Base case and end point):
    i1: [0:len(DNA1)]
    i2: [0:len(DNA2)]
Transitions:
    if dna1_n == dna2_m then (i1 + 1, i2 + 1)
    if dna1_n != dna2_m then (i1, i2 + 1) or (i1 + 1, i2)

For each transition we only move in one direction for i1 and i2
(i1 + 1) > i1 = calculate big before small (iterate backwards)
(i2 + 1) > i2 = calculate big before small (iterate backwards)
"""
def dna_match_bottomup_helper(DNA1, DNA2, cache):
    """Running Time: O(len(DNA1)*len(DNA2))"""
    for n in range(len(DNA1)-1, -1, -1):
        for m in range(len(DNA2)-1, -1, -1):
            if DNA1[n] == DNA2[m]:
                cache[n][m] = 1 + cache[n+1][m+1]
            else:
                cache[n][m] = max(cache[n][m+1], cache[n+1][m])
     
    return cache[0][0]
