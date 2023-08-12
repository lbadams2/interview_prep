

def Fibonacci(n):
    if n <= 1:
        return n
 
    else:
        return Fibonacci(n-1) + Fibonacci(n-2)
    
def reverse_polish(tokens):
    stack = []
    for t in tokens:
        if t not in "+-*/":
            stack.append(int(t))
        else:
            r, l = stack.pop(), stack.pop()
            if t == "+":
                stack.append(l+r)
            elif t == "-":
                stack.append(l-r)
            elif t == "*":
                stack.append(l*r)
            else:
                stack.append(int(float(l)/r))
    return stack.pop()
    
def factorial(n):
    if n==1 or n==0:
        return 1
    else:
        return n * factorial(n - 1)
    

def robber(nums):
    def rob(nums, i):
        if i < 0:
            return 0
        return max(rob(nums, i - 2) + nums[i], rob(nums, i - 1))
    return rob(nums, len(nums) - 1)


# given n pairs of parenthesis, return list of all well formed groups of parenthesis using n pairs, order matters
def generate_parenthesis(n):
    # left is how many ( are in current string s, right is how many )
    # dfs is depth first search
    def dfs(left, right, s):
        if len(s) == n * 2: # for example, if there are 10 parens in s and n is 5 we're done
            res.append(s)
            return 

        if left < n:
            dfs(left + 1, right, s + '(')

        if right < left:
            dfs(left, right + 1, s + ')')

    res = []
    dfs(0, 0, '')
    return res


# Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n]
def generate_combinations(n, k):
    if k == 0:
        return [[]]
    
    combos = []
    for i in range(k, n+1): # start at k because can't find combo of k numbers for len(nums) < k, i is passed as n
        # generate_combinations(i-1, k-1) finds combos for smaller instance of problem, just need to add i in parent caller
        for pre_combos in generate_combinations(i-1, k-1):
            combos.append(pre_combos + [i]) # pre_combos will be a list
    return combos


# given array of integers nums, return all possible subsets, the power set
def power_set(nums):
    def dfs(nums, path, ret):
        ret.append(path) # path is holding current set
        for i in range(len(nums)):
            dfs(nums[i+1:], path+[nums[i]], ret) # keep slicing list by moving start index up, add nums[i] to current set path

    ret = []
    dfs(nums, [], ret)
    return ret


# given an array nums, return all possible permutations
def permutations(nums):
    def dfs(nums, path, res):
        # if all nums have been checked and sliced away, path should be a permutation
        if not nums:
            res.append(path)
        # nums[:i]+nums[i+1:] removes nums[i], which is added to current perm with path+[nums[i]]
        for i in range(len(nums)):
            dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)
    
    res = []
    dfs(nums, [], res)
    return res

print(permutations([1,2,3]))