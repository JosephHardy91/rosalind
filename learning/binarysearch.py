# https://www.khanacademy.org/computing/computer-science/algorithms/binary-search/a/binary-search
# efficient algorithm for finding an item from a sorted list of items
# binary search rules out unreasonable guesses and keeps track of where reasonable guesses may lie as it
# progresses towards the answer
import math


def guessing_game(n, answer=26):
    min_guess = 1
    max_guess = n
    guess = -1
    steps = 0
    while guess != answer:
        if steps != 0:
            if guess < answer:
                min_guess = guess + 1
            elif guess > answer:
                max_guess = guess - 1

        guess = math.floor((min_guess + max_guess) / 2.)

        steps += 1

    return steps, guess


if __name__ == "__main__":
    steps_taken = {}
    steps_taken_counts = {}
    for answer in range(1, 100):
        steps, _ = guessing_game(100, answer)
        k = answer % 100  # str(answer)[-1]
        if k in steps_taken:
            steps_taken[k] = (steps_taken[k] * steps_taken_counts[k] + steps) / (steps_taken_counts[k] + 1)
            steps_taken_counts[k] = steps_taken_counts[k] + 1
        else:
            steps_taken[k] = steps
            steps_taken_counts[k] = 1

    #steps_taken_inv = {v: k for k, v in steps_taken.items()}



    import matplotlib.pyplot as plt

    x = range(min(steps_taken.keys()), max(steps_taken.keys()) + 1)
    y = list(map(lambda d: steps_taken[d], x))
    plt.scatter(x, y)
    plt.show()
