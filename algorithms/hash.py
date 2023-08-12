

# given an array of integers num and an integer target, return indices of the two numbers that add up to target
# can't use same element twice, there will be one solution
def two_sum(nums, target):
    d = {} # maps num in nums to its index in nums
    for i, num in enumerate(nums):
        r = target - num # need r to be in nums
        if r in d: 
            return [d[r], i]
        d[num] = i


# given unsorted array of integers nums, return smallest missing postive integer
# trick to this problem is recognizing answer must be in range [1, len(nums)+1], this allows usage of hash
# example [6,7,1,2]
# after appending 0, array is [6,7,1,2,0] and n or len(nums) will be 5
def first_missing_positive(nums):
    nums.append(0) # add extra element for the case when first missing positive is len(nums) + 1
    n = len(nums)

    # remove negative integers, and numbers that can't be the answer nums[i] >= n
    # the answer will always be in the range (1, n+1)
    for i in range(len(nums)):
        if nums[i]<0 or nums[i]>=n:
            nums[i]=0

    # nums[i]%n will always be valid index into nums array
    for i in range(len(nums)): #use the index as the hash to record the frequency of each number
        # nums[i] and nums[nums[i]%n] reference to different locations in the array
        # nums[i]%n == (nums[i]+n)%n
        # nums[nums[i]%n] will never access the index i representing the smallest missing integer
        # if the smallest missing integer is 3, you will never get 3%5=3 so nums[3] will never have n added to it
        nums[nums[i]%n] += n
    
    # unless array contains all integers from range(1,len(nums)), answer will be in this range
    # don't need to start from 0 because nums[0] accessed when nums[i]%n = 0, meaning nums[i] is n
    for i in range(1,len(nums)): # endpoint of range non inclusive
        # if quotient is 0, this means n was never added to nums[i]
        if nums[i]/n == 0:
            return i
    
    return n # only when array contains all integers from range(1,len(nums))