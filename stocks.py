import numpy as np
import matplotlib.pyplot as pyplot
import random
import csv

def calculateWeights(arr):
    weights = [[0 for i in range(len(arr[0]))]]
    weights_idx = 1
    for i in range(len(arr)):  # number of monte carlo trials
        payoffs = arr[i]
        payoffs = np.add(payoffs, weights[weights_idx - 1])
        weights.append(payoffs)
        weights_idx += 1
    return weights

def calculateEmpiricalEpsilonExact(arr, weights, h):
    epsilons = np.arange(0, 0.2, 0.01)
    best_payoff = -1
    best_epsilon = -1
    for epsilon in epsilons:
        payoff = calculateExpectedPayoff(arr, weights, epsilon, h)
        if payoff > best_payoff:
            best_payoff = payoff
            best_epsilon = epsilon

    return best_epsilon

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

# reaarange data for 20 stocks
new_data = []
stock_name_arr = []
row = []
for i in range(1258*20):
    row.append(data[i][7])
    if(i%1258 == 0 and i!=0):
        stock_name_arr.append(data[i][6])
        new_data.append(row)
        row = []
# print(stock_name_arr)
new_data_weights = calculateWeights(new_data)
# print(new_data_weights[0:1])
print("calculateEmpiricalEpsilonExact = ",calculateEmpiricalEpsilonExact(new_data, new_data_weights, 1.117198))
