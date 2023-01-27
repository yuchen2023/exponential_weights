import numpy as np
import matplotlib.pyplot as pyplot
import random
import csv

def calculateTheoreticalEpsilon(n, k):
    return np.sqrt(np.log(k)/n)

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

def calculateExpectedPayoff(arr, weights, epsilon, h, type="exp"): #calculates by previously produced weights
    payoff = 0
    i = 0
    avg = 0
    while i < len(arr):
        if type == "exp":
            cur_weight = exponentialWeights(weights[i], epsilon, h)
        elif type == "linear":
            cur_weight = linearWeights(weights[i], epsilon, h)
        cur_arr = arr[i]
        payoff = np.multiply(cur_weight, cur_arr)
        i+=1
        avg += np.sum(payoff)

    return avg

def exponentialWeights(arr, epsilon, h):
    out = []
    sum = 0
    for ele in arr:
        numerator = np.power((1+epsilon), ele/h)
        sum += numerator
        out.append(numerator)

    return np.divide(out, sum)
def linearWeights(arr, epsilon, h): #for comparison with linear weights
    sum = 0
    out = []
    for ele in arr:
        numerator = ((1+epsilon) * ele /h)
        sum += numerator
        out.append(numerator)
    if sum==0:
        return out
    return np.divide(out,np.sum(out))

def followTheLeader(arr):
    weights = [[0 for i in range(len(arr[0]))]]
    expected_payoff = 0
    args = []
    for idx, row in enumerate(arr):
        maxarg = np.argmax(weights[idx])
        args.append(maxarg)
        expected_payoff += row[maxarg]
        weights.append(np.add(weights[idx], row))

    return expected_payoff, args

def optimal(arr):
    best_payoff = -1
    best_action = -1
    for k in range(len(arr[0])):
        cur_payoff = 0
        for n in range(len(arr)):
            cur_payoff += arr[n][k]

        if cur_payoff > best_payoff:
            best_payoff = cur_payoff
            best_action = k

    return best_action, best_payoff

def rearrangeData(arr, days):
    data = []
    for i in range(10):
        data.append(arr[i][:days])
    return data


data = []
with open('all_stocks_5yr.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data.append(row)

data.pop(0)

# reaarange data for 20, 50, 100 stocks
# stock_amount = [10, 20, 50, 100]
# empirical_epsilon_monte = []
# theoretical_epsilon = []
# trials = 100
# for stocks in stock_amount:
#     new_data = []
#     stock_name_arr = []
#     row = []
#     k = 0
#     for i in range(1259*stocks):
#         k+=1
#         d = float(data[i][7]) if(data[i][7]!='#DIV/0!') else 0
#         row.append(d)
#         if(k%1259 == 0 and k!=0):
#             k = 0
#             stock_name_arr.append(data[i][6])
#             new_data.append(row)
#             row = []
#     # print(stock_name_arr)
#     new_data_weights = calculateWeights(new_data)
#     # print("calculateEmpiricalEpsilonExact = ",calculateEmpiricalEpsilonExact(new_data, new_data_weights, 1.117198))
#     e = 0
#     for i in range(trials):
#         e+=calculateEmpiricalEpsilonMonte(new_data, 1259,1.117198, consecutive=1)[0]/10
#     empirical_epsilon_monte.append(e)
#     theoretical_epsilon.append(calculateTheoreticalEpsilon(1259, stocks))
    

# pyplot.title("Stocks - Empirical Epsilon over 100 trials")
# pyplot.xlabel("k Stocks")
# pyplot.ylabel("Epsilon")
# pyplot.plot(stock_amount, empirical_epsilon_monte, label = "Empirical Best Epsilon")
# pyplot.plot(stock_amount, theoretical_epsilon,label = "Theoretical Best Epsilon")
# pyplot.legend()
# pyplot.show()


#linear and exponential comparison
new_data = []
stock_name_arr = []
row = []
k = 0
for i in range(1259*20):
    k+=1
    d = float(data[i][7]) if(data[i][7]!='#DIV/0!') else 0
    row.append(d)
    if(k%1259 == 0 and k!=0):
        k = 0
        stock_name_arr.append(data[i][6])
        new_data.append(row)
        row = []

# days = [5, 10, 100, 1000]
# trials = 100
# linear = []
# exp = []
# for days_len in days:
#     linear_avg = 0
#     exp_avg = 0
#     data = rearrangeData(new_data, days_len)
#     new_data_weights = calculateWeights(data)
#     for i in range(trials):
#         theoretical_epsilon = calculateTheoreticalEpsilon(1259, 10)
#         linear_avg += calculateExpectedPayoff(data, new_data_weights, theoretical_epsilon, 1.117198, "linear")
#         exp_avg += calculateExpectedPayoff(data, new_data_weights, theoretical_epsilon, 1.117198, "exp")
#     linear.append(np.divide(linear_avg, trials))
#     exp.append(np.divide(exp_avg, trials))

# pyplot.plot(days, np.divide(linear, days), label = "Linear Weights")
# pyplot.plot(days, np.divide(exp, days), label = "Exponential Weights")
# pyplot.title(f"Average Payoff/Days over 100 Trials for Stocks dataset")
# pyplot.ylabel("Payoff/Days")
# pyplot.xlabel("Days")
# pyplot.legend()
# print(linear, exp)
# pyplot.show()


rounds = [5, 10, 100, 1000]
trials = 20
ftl= []
theoretical_best = []
random = []
opt = []
for days_len in rounds:
    ftl_avg = 0
    tb_avg = 0
    random_avg = 0
    opt_avg = 0
    tb = calculateTheoreticalEpsilon(days_len, 10)
    data = rearrangeData(new_data, days_len)
    for i in range(trials):
        weights = calculateWeights(data)
        ftl_avg += followTheLeader(data)[0]
        tb_avg += calculateExpectedPayoff(data, weights, tb, 1.117198)
        random_avg += calculateExpectedPayoff(data, weights, 0, 1.117198)
        opt_avg += optimal(data)[1]
    opt.append(opt_avg/trials)
    ftl.append(ftl_avg/trials)
    theoretical_best.append(tb_avg/trials)
    random.append(random_avg/trials)

opt = np.divide(opt, rounds)
ftl = np.divide(ftl, rounds)
theoretical_best = np.divide(theoretical_best, rounds)
random = np.divide(random, rounds)

pyplot.plot(rounds, opt-ftl, label="FTL")
pyplot.plot(rounds, opt-theoretical_best, label = "Theoretical Best Epsilon")
pyplot.plot(rounds, opt-random, label = "No Epsilon")
pyplot.title("Averaged Per Day Regret over 20 Trials for Stocks")
pyplot.ylabel("Regret/Day")
pyplot.xlabel("Days")
pyplot.xlim(xmin = 10)
pyplot.legend()
pyplot.show()
