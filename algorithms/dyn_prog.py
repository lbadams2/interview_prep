
# given a string s, return the number of palindromic substrings in it
# minimum number is number of characters, a single character is considered a palindromic substring
def palindromic_substrings(s):
    n = len(s)
    # creates n x n matrix of zeroes, will be filled with boolean vals
    # each entry of the matrix represents whether the substring from i to j is a palindrome
    # matrix should be symmetric
    dp = [[0] * n for _ in range(n)]
    
    res = 0
    # i is right index
    for i in range(n-1, -1, -1): # start at n-1, stop at -1, subtract 1 at each iteration
        for j in range(i, n): # j is left index
            # 3 boolean expressions here, s[i] == s[j] checks if the ends of the string equal (could be single character)
            # (j-i+1) < 3 means its a short string that is automatically a palindrome if s[i] == s[j]
            # dp[i+1][j-1] is the previous value (this is the dp/memoization part) if the previous instance was a palindrome
            # and s[i] == s[j] then this is also a palindrome
            dp[i][j] = s[i] == s[j] and ((j-i+1) < 3 or dp[i+1][j-1])
            res += dp[i][j] # casts true to 1 and false to 0
    return res


# given integer array nums, find subarray with largest sum and return it
# could just remember prev_sum (dp[i-1]) like how max_subarray_product doesn't remember whole dp array
# remembering whole dp array would be necessary if you needed to know max subarray sum ending at each i
# only need to remember one val instead of 2 like in max_subarray_product
def max_subarray_sum(nums):
    n = len(nums)
    dp = [float('-inf')] * n
    dp[0] = nums[0]
    max_sum = dp[0]

    for i in range(1, n):
        # if dp[i-1] < 0 it effectively begins a new subarray starting at i
        # don't need two indices to keep track of start and end
        dp[i] = nums[i] + (dp[i-1] if dp[i - 1] > 0  else 0)
        max_sum = max(max_sum, dp[i])

    return max_sum


# robber maximizing profit and not robbing consecutive houses, values of nums are non negative
# dp but don't need array for history, just 2 variables
# recurrence relation rob(i) = max( rob(i - 2) + currentHouseValue, rob(i - 1) )
# dp more efficient than recursion here because of memoization (saving return values of rob(i)'s so rob(i) not called twice for same i)
def robber(nums):
    if len(nums) == 0:
        return 0
    
    # prev1 will be ahead of prev2
    prev1 = 0
    prev2 = 0
    
    for num in nums:
        tmp = prev1
        # prev1 and prev2 accumulate sum
        # at the start of an iteration prev1 is current sum and prev2 is 
        prev1 = max(prev2 + num, prev1) # either rob current and prev2 are rob previous
        prev2 = tmp # set prev2 to old prev1
    
    return prev1


# You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
def climb_stairs(n):
    if n <= 2: 
        return n
    
    dp = [0]*(n+1) # considering zero steps we need n+1 places
    dp[1] = 1 # only 1 way to climb 1 stair
    dp[2] = 2 # 2 ways to climb 2 stairs
    for i in range(3,n+1):
        dp[i] = dp[i-1] + dp[i-2] # unique ways to climb i stairs is sum of unique ways to climb i-1 and i-2 stairs
    return dp[n]


# given an array coins representing amounts coins can take, and an amount you need to reach, 
# reach amount with minimum number of coins, you can use infinite number of each coin, return number of coins used
# assume its possible to reach any number, so i - coin >= 0 is good enough, don't have to check for equality
def coin_change(coins, amount):
    # dp is array of size amount+1 with first element 0 and rest inf
    # dp[i] is minimum number of coins required to reach amount i
    dp = [0] + [float('inf') for i in range(amount)]

    # i represents an amount
    for i in range(1, amount+1):
        for coin in coins:
            if i - coin >= 0: # this means coin can be used for this amount i
                # dp[i - coin] is the minimum number of coins for amount i-coin, add 1 because we're using coin to reach amount i
                # dp[i] may already have a non inf value because all values of i are checked for each coin
                dp[i] = min(dp[i], dp[i-coin] + 1)

    # dp[-1] is at index amount, the answer we're looking for
    if dp[-1] == float('inf'):
        return -1
    return dp[-1]


# Given an integer array nums, return the length of the longest strictly increasing subsequence
def longest_increasing_sub(nums):
    n = len(nums)

    # 1 is the minimum answer
    # dp[i] is longest subsequence ending at nums[i]
    dp = [1] * n

    # these loops go through all subsequences that start at the beginning of the array
    for i in range(n):
        for j in range(i):
            # nums[i] has to be greater than nums[j] for the sequence to be increasing            
            if nums[i] > nums[j]:
                # dp[j] will have already been visited by the outer loop once you're at dp[i]
                # don't understand why adding 1 works if i and j are far apart
                # if i=7 and j=4, dp[i] has been visited already for j=0,1,2,3
                # so if there's a longer sequence at dp[4] that sequence builds on the sequences at prior j's
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


# Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
# subsequences here do not need to be contiguous, just in the same order. "ace" is a subsequence of "abcde"
def longest_common_sub(text1, text2):
    # 2 arrays so need matrix, text1 indices are rows, text2 indices are cols
    # need to use i+1 and j+1 as indices so dp dims 1 greater than string dims
    dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    
    # visit each cell in matrix once
    # compare all of text2 to every substring starting from first char in text1
    for i, c in enumerate(text1):
        for j, d in enumerate(text2):
            if c == d:
                dp[i + 1][j + 1] = 1 + dp[i][j]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    
    # answer will be max indices
    return dp[-1][-1]


# Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.
# can use each integer in nums as many times as you want for sum
# uses same strategy as coin change where size of dp array is the sum you need to reach
def combo_sum(nums, target):
    # dp[i] 
    dp = [0] * (target + 1)
    dp[0] = 1 # initialize to 1 otherwise sums wouldn't accumulate
    
    # same nested loop style as coin change, outer loop is over sum, inner loop is over nums that need to be added together
    for i in range(1, target + 1):
        for j in range(len(nums)):
            if (i - nums[j]) >= 0:
                # (i - nums[j]) is a number between 0 and target
                # each i-nums[j] value visited only once
                dp[i] += dp[i - nums[j]]
    
    return dp[target]


# There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). 
# The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
# The robot can only move either down or right at any point in time.
# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
def unique_paths(m, n):
    # dp has m rows and n cols to match the m x n grid
    # initialize to 1 otherwise sums wouldn't accumulate
    dp = [[1]*n for i in range(m)]
    
    # start at 1 in each loop to prevent index out of bounds
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[-1][-1]


print(combo_sum([1,2,3], 4))