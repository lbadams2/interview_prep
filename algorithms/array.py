
def reverse_array(a):
    i=0
    j = len(a) - 1
    while(i < j):
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp
        i += 1
        j -= 1
    return a
    # rev = a[::-1]

# given an array prices where prices[i] is the price of a given stock on the ith day.
# choose a single day i to buy and single day j to sell that maximizes profit, return profit
def stock_picker(prices):
	maxcost = 0
	
	mini = prices[0]
	for i in range(len(prices)):
		mini = min(mini, prices[i])
		cost = prices[i] - mini
		maxcost = max(maxcost, cost)
	return maxcost


# given array of integers and target sum, return minimum length of subarray whose sum is greater than or equal to target
def min_subarray_sum(nums, target):
    i, j, sum, min_length = 0, 0, 0, float('inf')

    # both i and j move to the right
    while j < len(nums):
        sum += nums[j]
        j += 1
        
        # will have j >= i, i is moving to right to catch up with j, min_length should decrease each iteration 
        # at the end of this loop i should equal j
        while sum >= target:
            min_length = min(min_length, j - i)
            sum -= nums[i] # will gradually subtract the nums[j] that have been added to sum
            i += 1

    if min_length == float('inf'):
        return 0
    else:
        return min_length


# nums1 and nums2 are arrays of numbers in non decreasing order of length m and n respectively
# nums1 has n+m size with the last m elements 0, merge nums1 and nums2 into nums1 in non decreasing order
def merge_sorted_arrays(nums1, m, nums2, n):
    while m > 0 and n > 0:
        if nums1[m-1] >= nums2[n-1]: # because nums1 and 2 are sorted, nums1[m-1] and nums2[n-1] are greatest elements in each array
            # add nums1[m-1] to end of merged array if its greater than last element if nums2
            # now nums1[m-1] is in nums1 twice, lowest LHS will go is nums[n-1] when m=0, it will eventually be overwritten
            nums1[m+n-1] = nums1[m-1]
            m -= 1
        else:
            nums1[m+n-1] = nums2[n-1] # add nums2[n-1] to end of merged array if its greater than last element if nums1
            n -= 1
    if n > 0:
        # nums2[:n] haven't been processed yet, they must be smaller than any element in nums1
        # elements added to nums1 from right so this won't overwrite anything
        nums1[:n] = nums2[:n]


# temps is array of temps, return array ans such that ans[i] is the number of days you have to wait after the ith day
# to get a warmer temp. ans will be same size as temps
def daily_temperatures(temps):
    ans = [0] * len(temps)
    
    # use stack to keep track of previous elements visited, 
    # and because answer for these previous elements depends on future elements
    # stack holds indices of temps array
    stack = []
    
    for i, t in enumerate(temps):
      # temps[stack[-1]] is most recent day that doesn't have an answer yet, t is temperature for current day
      # while current day temp is warmer than most recent day that doesn't have an answer yet, 
      # remove from stack and fill in answer
      while stack and temps[stack[-1]] < t:
        last_day = stack.pop() # removes and returns last item in list, this will be an index
        ans[last_day] = i - last_day
      stack.append(i) # add current day as it doesn't yet have an answer

    return ans


# heights is height of each person standing side by side in line
# the ith person can see the jth person if i < j and min(heights[i], heights[j]) > max(heights[i+1], heights[i+2], ..., heights[j-1])
# Return an array answer of length n where answer[i] is the number of people the ith person can see to their right in the queue
def num_visible_queue(heights):
    res = [0] * len(heights)

    # use stack to keep track of previous elements visited, 
    # and because answer for these previous elements depends on future elements
    # stack holds indices
    stack = []
    
    for i, v in enumerate(heights):
        # v is height of current person, moving right to left from perspective of people standing
        # heights[stack[-1]] is height of previous person, or person who is taller than everyone to his left so far
        while stack and heights[stack[-1]] <= v:
            res[stack.pop()] += 1 # remove last person on stack as they won't see beyond v, add 1 as they can see v
        if stack:
            res[stack[-1]] += 1 # add 1 as last person on stack is taller than current person v
        # add current person regardless of height as the next person may be shorter than him, even if person to his right is taller
        # there are many nested windows in which people can see each other
        stack.append(i)
    return res


# given array of intervals (each element in array is a (start, end) pair), merge overlapping intervals and return non overlapping intervals
def merge_intervals(intervals):
    out = [] # out is list of lists, allows out[][]
    for i in sorted(intervals, key=lambda i: i[0]): # sort by start
        if out and i[0] <= out[-1][1]: # if start of current interval i[0] is inside last merged interval out[-1][1]
            # update last merged interval if end of current interval i[1] greater than end of last merged interval
            out[-1][1] = max(out[-1][1], i[1])
        else:            
            #out += [i]
            # if no overlap add current interval to merged intervals, i is [start, end] list
            out.append(i)
    return out


# given array of intervals, return minimum number of intervals to remove to make the rest of the intervals non overlapping
def non_overlapping_intervals(intervals):
    end, cnt = float('-inf'), 0
    for s, e in sorted(intervals, key=lambda x: x[1]): # unpacks interval endpoints into s and e, sort by e
        # if s >= end, then current interval (s, e) does not overlap with most recent interval
        # end is the latest time reached so far in the sorted intervals
        if s >= end:
            end = e
        else: 
            cnt += 1 # if s < end, then (s, e) interval overlaps with another interval and it can be removed
    return cnt


# given integer array nums, return array ans such that ans[i] is product of all nums in array except nums[i]
def product_except_self(nums):
    ans, suf, pre = [1]*len(nums), 1, 1
    for i in range(len(nums)):
        # prefix product using left index
        # if i is an index in second half of array, this will be second time num at ans[i] is being multiplied due to ans[-1-i] below
        # the order of the below 2 lines ensures nums[i] (self) will not be multiplied in to ans[i]
        ans[i] *= pre
        pre *= nums[i] # pre gradually accumulates products of all numbers its seen
        
        # suffix product using right index, first iteration will be ans[-1] and then move to start of array
        # if i is index in first half of array, this will be second time num at ans[-1-i] is being multiplied due to ans[i] above
        # the order of the below 2 lines ensures nums[i] (self) will not be multiplied in to ans[i]
        ans[-1-i] *= suf
        suf *= nums[-1-i] # suf gradually accumulates products of all numbers its seen starting from end of array
    return ans


# given integer array nums, return true if any value appears at least twice, false otherwise
def contains_duplicate(nums):
    return len(nums) != len(set(nums))



# given array nums, find subarray with largest product and return product
# this is like a dp problem, similar to robber in that you don't need to remember an array of previous answers
# just need to remember max and min
def max_product_subarray(nums):
    max_product = nums[0]
    n = len(nums)
    # imax and imin remember previous max and min products, need to remember min product for negatives
    imax, imin = max_product, max_product

    for i in range(1, n):
        # candidates for max product are the current number, or current number times previous max or min
        candidates = (nums[i], imax * nums[i], imin * nums[i])
        imax = max(candidates)
        imin = min(candidates)

        max_product = max(max_product, imax)
    
    return max_product


# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, 
# and nums[i] + nums[j] + nums[k] == 0
# in below in each iteration nums[i] is held constant and it searches for 2 numbers that can add to nums[i] to get 0
# have to sort nums for this algorithm to work
def three_sum_zero(nums):
    res = []
    nums.sort()
    for i in range(len(nums)-2): # len(nums)-2 because 3 numbers are needed
        if i > 0 and nums[i] == nums[i-1]: # avoid triplets with duplicate values
            continue
        l, r = i+1, len(nums)-1 # both l and r are to the right of i
        while l < r:
            s = nums[i] + nums[l] + nums[r] # i will be to the left of l
            if s < 0: # since nums[l] is less than nums[r], if sum is too small need to increase l
                l +=1 
            elif s > 0: # since nums[r] is the largest number, if sum is too big need to decrease r
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                # below 2 while loops ensure no duplicate triplets are added as array was sorted
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res


# given a list of heights representing vertical lines, find the 2 lines that can hold the most water if they formed the
# walls of a container, they can hold more water the farther apart they are in the array, basically find max area
def max_area(heights):
    i, j = 0, len(heights) - 1
    water = 0
    while i < j: # i is left pointer, j is right pointer
        water = max(water, (j - i) * min(heights[i], heights[j])) # water is area of rectangle created by lines in array

        # keep the tallest line constant
        # there is a proof explaining why this greedy approach works, its intuitive to do the below but not clear its optimal
        if heights[i] < heights[j]:
            i += 1
        else:
            j -= 1
    return water


# merge n sorted lists
def merge_n_lists(lists):
    pass

# return the k most common items in a list
def k_most_common(arr, k):
    pass

print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))