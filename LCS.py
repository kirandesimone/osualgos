from typing import List
"""
Info:
    A subsequence of a string is a new string generated from the original string with 
    some characters (can be none) deleted without changing the relative order of the remaining characters.

    A common subsequence of two strings is a subsequence that is common to both strings.

    "ace" is a subsequence of "abcde".

Goal:
    return the length of their longest (maximum) common subsequence. If there is no common subsequence, return 0

inputs:
    text1 -> str
    text2 -> str

    helpers:
        i1 for text1
        i2 for text2

    both can vary in length

state:
    i1, i2

transitions:
    if text1_i == text2_i, then i1 + 1, i2 + 1 
    if text1_i != text2_i, then two cases doesn't matter which one goes first
        i1 + 1, i2
        i1, i2 + 1  

base case:
    if text1_n == 0 or text2_n == 0 then we ran through one string but can still have more elements in the other

order:
    For it to be considered a subsequence, the letters must match AND
    0 <= i1 <= i2 < n for i1 + 1, i2 state
    0 <= i2 <= i1 < n for i1, i2 + 1 state

thoughts:
    since we have subsequences, we can cut down from or go up to n-1
"""

def bruteLCS(text1: str, text2: str, i1: int, i2: int) -> int:
    """Running Time:  """
    # Base Case
    if i1 == len(text1) or i2 == len(text2):
        return 0

    lcsl = 0
    # Decisions/Transitions
    if text1[i1] == text2[i2]:
        lcsl = 1 + bruteLCS(text1, text2, (i1 + 1), (i2 + 1))
    else:
        lcsl1 = bruteLCS(text1, text2, (i1 + 1), i2)
        lcsl2 = bruteLCS(text1, text2, i1, (i2 + 1))
        lcsl += max(lcsl1, lcsl2)
    
    return lcsl

def memoLCS(text1: str, text2: str, i1: int, i2: int, cache: List[List[int]]) -> int:
    """Running Time: """
    if i1 == len(text1) or i2 == len(text2):
        return 0
    
    # At text1_n and text2_m, if we calculated the subsequence length just return it
    if cache[i1][i2] > -1:
        return cache[i1][i2]
    
    if text1[i1] == text2[i2]:
        cache[i1][i2] = 1 + memoLCS(text1, text2, (i1 + 1), (i2 + 1), cache)
    else:
        lcsl1 = memoLCS(text1, text2, (i1 + 1), i2, cache)
        lcsl2 = memoLCS(text1, text2, i1, (i2 + 1), cache)
        cache[i1][i2] = max(lcsl1, lcsl2)
    
    return cache[i1][i2]

def iterLCS(text1: str, text2: str, cache: List[List[int]]) -> int:
    for n in range(len(text1)-1, -1, -1):
        for m in range(len(text2)-1, -1, -1):
            if text1[n] == text2[m]:
                cache[n][m] = 1 + cache[n+1][m+1]
            else:
                cache[n][m] = max(cache[n][m+1], cache[n+1][m])
    
    return cache[0][0]


class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # brute = bruteLCS(text1, text2, 0, 0)
        # memoCache = [[-1]*len(text2) for n in range(len(text1))]
        # memo = memoLCS(text1, text2, 0, 0, memoCache)
        iterCache = [[0]*(len(text2) + 1) for n in range(len(text1) + 1)]
        dp = iterLCS(text1, text2, iterCache)

        return dp