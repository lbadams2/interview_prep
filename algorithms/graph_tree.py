from collections import deque
import math

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# given root of binary tree, return its max depth
def bt_max_depth(root):
    def dfs(root, depth):
        if not root: 
            return depth
        return max(dfs(root.left, depth + 1), dfs(root.right, depth + 1))
                    
    return dfs(root, 0)


# Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree 
# and inorder is the inorder traversal of the same tree, construct and return the binary tree
def build_tree(preorder, inorder):
    if inorder:
        ind = inorder.index(preorder.pop(0)) # pop(0) is same as popleft()
        root = TreeNode(inorder[ind])
        root.left = build_tree(preorder, inorder[0:ind])
        root.right = build_tree(preorder, inorder[ind+1:])
        return root
    

def bfs(matrix):
  # Check for an empty matrix/graph.
  if not matrix:
    return []

  rows, cols = len(matrix), len(matrix[0])
  visited = set()
  directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

  def traverse(i, j):
    queue = deque([(i, j)]) # popleft and append behave same as list
    while queue:
      curr_i, curr_j = queue.popleft()
      if (curr_i, curr_j) not in visited:
        visited.add((curr_i, curr_j))
        # Traverse neighbors.
        for direction in directions:
          next_i, next_j = curr_i + direction[0], curr_j + direction[1]
          if 0 <= next_i < rows and 0 <= next_j < cols:
            # Add in question-specific checks, where relevant.
            queue.append((next_i, next_j))

  for i in range(rows):
    for j in range(cols):
      traverse(i, j)


def dfs(matrix):
  # Check for an empty matrix/graph.
  if not matrix:
    return []

  rows, cols = len(matrix), len(matrix[0])
  visited = set()
  directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

  def traverse(i, j):
    if (i, j) in visited:
      return

    visited.add((i, j))
    # Traverse neighbors.
    for direction in directions:
      next_i, next_j = i + direction[0], j + direction[1]
      if 0 <= next_i < rows and 0 <= next_j < cols:
        # Add in question-specific checks, where relevant.
        traverse(next_i, next_j)

  for i in range(rows):
    for j in range(cols):
      traverse(i, j)


# Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), return the number of islands.
def num_islands(grid):
    def dfs(grid, i, j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
            return
        grid[i][j] = '#'
        # dfs in all 4 directions
        dfs(grid, i+1, j)
        dfs(grid, i-1, j)
        dfs(grid, i, j+1)
        dfs(grid, i, j-1)
    
    if not grid:
        return 0
        
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(grid, i, j) # this will mark all adjacent land to grid[i][j] with # instead of 1
                count += 1
    return count


# Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
# The distance between two adjacent cells is 1.
# this uses dp by saving top and left dists for the second nested for loop to reference
def matrix_dists(mat):
    m, n = len(mat), len(mat[0]) # m is rows, n is cols

    # 4 directions to cover, can cover 2 directions at a time with nested loop
    for r in range(m):
        for c in range(n):
            if mat[r][c] > 0: # if cell is 0 then dist is 0, don't need to do anything
                top = mat[r - 1][c] if r > 0 else math.inf
                left = mat[r][c - 1] if c > 0 else math.inf
                # if top and left are both 1 then distance to nearest zero is at least 2
                # if one of top or left is zero then dist is 1
                mat[r][c] = min(top, left) + 1 # mat[r][c] gets overwritten with answers to subproblems

    for r in range(m - 1, -1, -1): # traverse rows from bottom up
        for c in range(n - 1, -1, -1): # travese cols from right to left
            if mat[r][c] > 0:
                bottom = mat[r + 1][c] if r < m - 1 else math.inf
                right = mat[r][c + 1] if c < n - 1 else math.inf
                # mat[r][c] will be dist filled by previous loops
                # adding 1 to bottom and right for same reason as adding to top and left
                mat[r][c] = min(mat[r][c], bottom + 1, right + 1)

    return mat


print(num_islands([
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]))