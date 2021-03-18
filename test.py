def recursive_fact(number):
    if number == 0:
        return 1
    if number == 1:
        return number
    else:
        return number * recursive_fact(number - 1)


def iterative_fact(number):
    if number == 0:
        return 1
    res = number
    for i in range(number - 1, 1, -1):
        res *= i
    return res
