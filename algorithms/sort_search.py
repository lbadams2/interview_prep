import random

# nums is already sorted
def binary_search(nums, target):
    left=0
    right=len(nums)-1
    while(left<= right):
        mid=(left+right) // 2
        if nums[mid]==target:
            return mid
        elif nums[mid] < target:
            left=mid+1 # target is in right half of array
        else:
            right=mid-1 # target is in left half of array
    return -1

# array consisting of 0, 1, and 2s. sort them in place so they are all adjacent with 0s first, then 1s then 2s
# problem may say 0s are red, 1s white, 2s blue
def sort_colors(nums):
    red, white, blue = 0, 0, len(nums)-1 # 3 pointers, one for each color
    
    # red, white then blue, loop ends once white reaches blue pointer and all colors should be sorted
    while white <= blue:
        if nums[white] == 0: # this means the element is red
            nums[red], nums[white] = nums[white], nums[red] # swap the elements, has no effect if pointers are equal
            white += 1 # advance both pointers
            red += 1
        elif nums[white] == 1: # the element is white
            white += 1
        else: # element is blue
            nums[white], nums[blue] = nums[blue], nums[white] # swap elements
            blue -= 1 # decrement blue pointer


# nums is originally sorted in ascending order, but then possibly pivoted about index k
# result of pivot about k is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]
# given such an array, return the index of target if it is in nums
def search_rotated_sorted_array(nums, target):
    lo, hi = 0, len(nums)

    while lo < hi:
        mid = (lo + hi) // 2 # floor division

        # since array was sorted, if nums[mid] < nums[0] then pivot index was in second half
        # if nums[mid] < nums[0] and target < nums[0] then ...
        # if nums[mid] > nums[0] then pivot index was in first half
        # if nums[mid] > nums[0] and target > nums[0] then ...
        if (nums[mid] < nums[0]) == (target < nums[0]):
            num = nums[mid]
        elif target < nums[0]:
            num = float('-inf')
        else:
            num = float('inf')

        # num here is either the midpoint, -inf, or inf
        if num < target:
            lo = mid + 1 # only search right half
        elif num > target:
            hi = mid # only search left half
        else: # num = nums[mid] must have been run above
            return mid
    
    return -1


# given an integer array nums and integer k, return the kth largest element in the array
def kth_largest(nums, k):
    if not nums: 
        return
    pivot = random.choice(nums)
    left =  [x for x in nums if x > pivot]    
    mid  =  [x for x in nums if x == pivot]
    right = [x for x in nums if x < pivot]    
    
    L, M = len(left), len(mid)
    
    # k originally has no relationship to pivot, pivot is chosen at random
    if k <= L: # if k is less than the number of elements that are greater than pivot
        return kth_largest(left, k)
    elif k > L + M: # if k is greater than the number of elements that are greater than pivot plus the number that equal pivot
        return kth_largest(right, k - L - M)
    else:
        return mid[0] # elements of mid will be the same, they equal pivot
    

def insertionSort(arr):
     
    # := operator does assignment within an expression
    if (n := len(arr)) <= 1:
        return
    
    # j is always less than i
    for i in range(1, n):
            
        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >=0 and key < arr[j]:
            arr[j+1] = arr[j] # arr[j+1] is initially arr[i] which is key, move arr[j] up one spot
            j -= 1
        arr[j+1] = key

print(kth_largest([3,2,3,1,2,4,5,5,6], 4))