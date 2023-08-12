# reverse_array
def reverse_array(arr):
    n = len(arr)
    i = 0
    # shouldn't need to return arr
    while i < n // 2:
        arr[i], arr[-1-i] = arr[-1-i], arr[i]

# given an array prices where prices[i] is the price of a given stock on the ith day.
# choose a single day i to buy and single day j to sell that maximizes profit, return profit
def stock_picker(prices):
    max_cost = 0
    min_price = prices[0]

    for i in range(len(prices)):
        if prices[i] - min_price > max_cost:
            max_cost = prices[i] - min_price
        if prices[i] < min_price:
            min_price = prices[i]
    return max_cost


# given array of integers and target sum, return minimum length of subarray whose sum is greater than or equal to target
def min_subarray_sum(nums, target):
    i, j, sum, min_length = 0, 0, 0, float('inf')
    for j in range(nums):
        sum += nums[j]

        while sum >= target:
            sum -= nums[i]
            min_length = min(j-i, min_length)
            i += 1
    return min_length

# temps is array of temps, return array ans such that ans[i] is the number of days you have to wait after the ith day
# to get a warmer temp. ans will be same size as temps
def daily_temperatures(temps):
    ans = [0] * len(temps)
    stack = []

    for i, t in enumerate(temps):
        while stack and stack[-1] < t:
            last_day = stack.pop()
            ans[last_day] = i - last_day
        stack.append(i)
    return ans

                

# heights is height of each person standing side by side in line
# the ith person can see the jth person if i < j and min(heights[i], heights[j]) > max(heights[i+1], heights[i+2], ..., heights[j-1])
# Return an array answer of length n where answer[i] is the number of people the ith person can see to their right in the queue
def num_visible_queue(heights):
    ans = [0] * len(heights)
    stack = []

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] < h:
            last_ind = stack.pop()
            ans[last_ind] += 1
        if stack:
            ans[stack[-1]] += 1
        stack.append(i)
    return ans



# given array of intervals (each element in array is a (start, end) pair), merge overlapping intervals and return non overlapping intervals
def merge_intervals(intervals):
    out = []
    sorted(intervals, lambda x: x[0]) # sorts ascending
    for i in intervals:
        if out and out[-1][1] >= i[0]:
            out[-1][1] = max(out[-1][1], i[0])
        else:
            out.append(i)
    return out


# given array of intervals, return minimum number of intervals to remove to make the rest of the intervals non overlapping
def non_overlapping_intervals(intervals):
    pass


# given integer array nums, return array ans such that ans[i] is product of all nums in array except nums[i]
def product_except_self(nums):
    ans = [1] * len(nums)
    pre, suf = 1, 1
    for i, n in enumerate(nums):
        ans[i] *= pre
        pre *= nums[i]

        ans[-1-i] *= suf
        suf *= nums[-1-i]
    return ans


# given array nums, find subarray with largest product and return product
# this is like a dp problem, similar to robber in that you don't need to remember an array of previous answers
# just need to remember max and min
def max_product_subarray(nums):
    max_product = nums[0]
    min_prod, max_prod = max_product, max_product
    for i in range(len(nums)):
        candidates = (nums[i], nums[i] * min_prod, nums[i] * max_prod)
        max_prod = max(candidates)
        min_prod = min(candidates)
        max_product = max(max_product, max_prod)
    return max_product


# given a list of heights representing vertical lines, find the 2 lines that can hold the most water if they formed the
# walls of a container, they can hold more water the farther apart they are in the array, basically find max area
def max_area(heights):
    pass


# given a string s, return the number of palindromic substrings in it
# minimum number is number of characters, a single character is considered a palindromic substring
def palindromic_substrings(s):
    pass


# given integer array nums, find subarray with largest sum and return it
# could just remember prev_sum (dp[i-1]) like how max_subarray_product doesn't remember whole dp array
# remembering whole dp array would be necessary if you needed to know max subarray sum ending at each i
# only need to remember one val instead of 2 like in max_subarray_product
def max_subarray_sum(nums):
    pass


# robber maximizing profit and not robbing consecutive houses, values of nums are non negative
# dp but don't need array for history, just 2 variables
# recurrence relation rob(i) = max( rob(i - 2) + currentHouseValue, rob(i - 1) )
# dp more efficient than recursion here because of memoization (saving return values of rob(i)'s so rob(i) not called twice for same i)
def robber(nums):
    pass


# You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
def climb_stairs(n):
    pass


# given an array coins representing amounts coins can take, and an amount you need to reach, 
# reach amount with minimum number of coins, you can use infinite number of each coin, return number of coins used
# assume its possible to reach any number, so i - coin >= 0 is good enough, don't have to check for equality
def coin_change(coins, amount):
    pass


# Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
# subsequences here do not need to be contiguous, just in the same order. "ace" is a subsequence of "abcde"
def longest_common_sub(text1, text2):
    pass


# Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.
# can use each integer in nums as many times as you want for sum
# uses same strategy as coin change where size of dp array is the sum you need to reach
def combo_sum(nums, target):
    pass


# There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). 
# The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
# The robot can only move either down or right at any point in time.
# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
def unique_paths(m, n):
    pass


# given an array of integers num and an integer target, return indices of the two numbers that add up to target
# can't use same element twice, there will be one solution
def two_sum(nums, target):
    pass


def reverse_polish(tokens):
    pass


def factorial(n):
    pass


def Fibonacci(n):
    pass


# given array of integers nums, return all possible subsets, the power set
def power_set(nums):
    pass


def binary_search(nums, target):
    pass


def insertionSort(arr):
    pass


def lychrel_numbers():
    pass