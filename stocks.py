import numpy as np
import matplotlib.pyplot as pyplot
import random
import csv

def calculateEmpiricalEpsilonMonte(arr, trials, h, consecutive):
    step = 0.01
    epsilons = np.arange(step, 0.2, step)
    best_epsilon = -1
    best_payoff = -1
    random_trials = []
    for i in range(trials):
        cur_trial = random.randint(0, len(arr) - 1)
        end = cur_trial + consecutive
        while cur_trial < end:
            random_trials.append(arr[cur_trial])
            cur_trial += 1

    for epsilon in epsilons:
        weights = [[0 for i in range(len(arr[0]))]]
        weights_idx = 1
        for i in range(len(random_trials)): #number of monte carlo trials
            payoffs = random_trials[i]
            payoffs = np.add(payoffs, weights[weights_idx-1])
            weights.append(payoffs)
            weights_idx += 1

        cur_payoff = calculateExpectedPayoff(random_trials, weights, epsilon, h)

        if cur_payoff > best_payoff:
            best_payoff = cur_payoff
            best_epsilon = epsilon

    return best_epsilon, best_payoff


def calculateExpectedPayoff(arr, weights, epsilon, h): #calculates by previously produced weights
    payoff = 0
    i = 0
    trials = 100
    avg = 0
    for j in range(trials):
        while i < len(arr):
            cur_weight = exponentialWeights(weights[i], epsilon, h)
            cur_arr = arr[i]
            payoff += np.random.choice(cur_arr, 1, p=cur_weight)
            i+=1
        avg += payoff
    return avg/trials

def exponentialWeights(arr, epsilon, h):
    out = []
    sum = 0
    for ele in arr:
        numerator = np.power((1+epsilon), ele/h)
        sum += numerator
        out.append(numerator)

    return np.divide(out, sum)

data = []
with open('all_stocks_5yr.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data.append(row)

data.pop(0)
new_data = []
row = []
print(data[:10])
for i in range(1260*20):
    row.append(data[i][7])
    if(i%1260 == 0 and i!=0):
        new_data.append(row)
        row = []
print(new_data[:1])

