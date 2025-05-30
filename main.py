import matplotlib.pyplot as plt
import time
import random

from typing import List, Callable


def insertion_sort(arr: List[int]) -> List[int]:
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
    
    return arr

def measure_time(func: Callable, arr: List[int]) -> float:
    start_time = time.time()
    _sorted = func(arr)
    end_time = time.time()

    return end_time - start_time

def main():
    sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    best_times = []
    worst_times = []
    average_times = []

    for size in sizes:
        sorted_arr = [n for n in range(size)]
        reversed_arr = [n for n in range(size-1, -1, -1)]
        random_arr = random.sample(range(size), size)

        best_time = measure_time(insertion_sort, sorted_arr)
        worst_time = measure_time(insertion_sort, reversed_arr)
        average_time = measure_time(insertion_sort, random_arr)

        best_times.append(best_time)
        worst_times.append(worst_time)
        average_times.append(average_time)

    plt.plot(sizes, best_times, label='Best Case')
    plt.plot(sizes, worst_times, label='Worst Case')
    plt.plot(sizes, average_times, label='Average Case')
    plt.xlabel('Array Size')
    plt.ylabel('Running Time (seconds)')
    plt.title('Insertion Sort Running Time Complexity')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()