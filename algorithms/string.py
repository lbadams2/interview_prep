import collections

# return length of longest substring that doesn't repeat any characters
def longest_without_repeating(s):
    chars = [0] * 128 # 128 is number of lower case, upper case, digit, special and punctuation characters
    left = right = 0
    res = 0

    while right < len(s):
        r = s[right]
        chars[ord(r)] += 1 # ord returns unicode representation

        # this loop starts once you've seen 2 consecutive characters
        # then starts from the left pointer until it reaches the consecutive characters, decrementing the counter makes it break the loop
        while chars[ord(r)] > 1:
            l = s[left]
            chars[ord(l)] -= 1
            left += 1

        res = max(res, right - left + 1)
        right += 1
    
    return res  


# 2 strings s and t of length m and n respectively
# return the minimum substring of s called w such that every character in t (including duplicates) is included in w
# t doesn't need to be in w in order, there can be characters in w that aren't in t

# this solution finds the first window that contains all of t, then in subsequent iterations of for loop it expands the window to the right
# reduce left side if possible
def min_window_substring(s, t):
    need = collections.Counter(t)            #hash table to store char frequency, initialized with frequencies of chars in t, not zeroes
    missing = len(t)                         #total number of chars we care
    start, end = 0, 0
    i = 0 # left index
    for j, char in enumerate(s, 1):          #index j from 1, right index
        if need[char] > 0:
            missing -= 1
        need[char] -= 1 # when need[char] < 0 that means you have more instances of char than you need in the window
        if missing == 0:                     #match all chars
            while i < j and need[s[i]] < 0:  #remove chars to find the real start, reduce window size from left if possible
                need[s[i]] += 1
                i += 1
            need[s[i]] += 1                  #make sure the first appearing char satisfies need[char]>0
            missing += 1                     #we missed this first char, so add missing by 1
            if end == 0 or j-i < end-start:  #update window
                start, end = i, j
            i += 1                           #update i to start+1 for next window
    return s[start:end]


print(longest_without_repeating('geeksforgeeks'))