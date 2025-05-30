def ternary_search(input_list, target):
    """
    Finds the target element by splitting the list into thirds
    """
    left = 0
    right = len(input_list) - 1

    while right >= left:
        subset_size = (right - left) // 3
        m1 = left + subset_size 
        m2 = right - subset_size

        # Add 1 for 1-indexing
        if input_list[m1] == target:
            return (m1 + 1) 

        if input_list[m2] == target:
            return (m2 + 1)

        if target < input_list[m1]:
            right = m1 - 1
        elif target > input_list[m2]:
            left = m2 + 1
        else:
            left = m1 + 1
            right = m2 - 1
            
    return (-1)