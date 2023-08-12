def repeated_string(file):
    with open(file) as f:
        string = f.read()
    
    substrs = {}
    n = len(string)
    for i in range(n):
        for j in range(i):
            if string[i:j] in substrs:
                substrs[string[i:j]] += 1
    
    return max(substrs)


# if input is 5 return 6, if input is 6 return 5
def return_num(n):
    return 11 - n