"""
Info:
    we have n amount of pets
    each pet has a hunger level from 1 --> n where the ith dog has a hunger level of hl[i]
    we have m amount of dog biscuits 

Goal:
    to satify the maximum amount of hungry dogs

How:
    To satify a dog with a hunger level of i, we must provide a biscuit of size j that
    is greater than or equal to hunger level i

Method:
    optimization problem since we need to find the maximum
    DP? probably
    Greedy? yep, thats the point of the assignment
    div_and_conq? probably

Constraints:
    You cannot give the same biscuit to two different dogs.
    Each dog can receive only one biscuit.
    If no dogs can be satisfied, return 0.

State:
    amount_of_pets (n) --> int
    hunger_level --> array[n]
    biscuit_size --> array[m]
    
Theories:
    could we just sum up the hunger levels and the biscuit sizes, divide the two and then
    take the result and divide it by the amount of pets to find the amount of pets fed?

    If you run out of biscuits then you can't feed anymore dogs

    pets per biscuit -> if we have more biscuits than pets then we know a max of n pets
        could be fed. if we have more pets than biscuits then we know a max of m pets
        can be fed.
    
    We're gonna have to sort them I think...
    checking every combination will take too long (n^2)

"""

def feedDog(hunger_level, biscuit_size):
    hunger_level.sort()
    biscuit_size.sort()
    n = len(hunger_level)
    m = len(biscuit_size)
    i = 0
    j = 0
    pets_fed = 0
    curr_bs = 0

    while i < n and j < m: 
        curr_bs += biscuit_size[j]
        curr_hl = hunger_level[i]

        if curr_hl <= curr_bs:
            curr_bs = 0
            pets_fed += 1
            i += 1
            j += 1
        else:
            j += 1

    
    return pets_fed



def main():
    hunger_level = [3, 13, 5, 15, 12, 28]
    biscuit_size = [30, 1, 5, 3, 14, 10, 2]
    pets_fed = feedDog(hunger_level, biscuit_size)
    print(pets_fed)

if __name__ == "__main__":
    main()