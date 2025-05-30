def evaluateString(input):
    """
    Takes an arithmetic expression string and finds the sum
    recursively.
    """
    if len(input) <= 0:
        return 0

    i = 0
    while i < len(input):
        if input[i] == '+': # O(n)
            break
        i += 1

    num = int(input[0:i])
    addendum = evaluateString(input[i+1:]) # T(n-(i+1))

    return num + addendum 
