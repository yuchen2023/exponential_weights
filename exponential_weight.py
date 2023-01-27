import numpy as np
import matplotlib.pyplot as plt
import random
import heapq



def calculateTheoreticalEpsilon(n, k):
    return np.sqrt(np.log(k)/n)

def calculateEmpiricalEpsilonMonte(arr, trials, h, consecutive=1):
    if len(arr) > 1000:
        epsilons = np.arange(0, 0.3, 0.05)
    else:
        epsilons = np.arange(0, 0.5, 0.01)
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

def calculateEmpiricalEpsilonExact(arr, weights, h, type="exp"):
    if len(arr) > 1000:
        epsilons = np.arange(0, 0.3, 0.05)
    else:
        epsilons = np.arange(0, 0.5, 0.01)
    best_payoff = -1
    best_epsilon = -1
    for epsilon in epsilons:
        payoff = calculateExpectedPayoff(arr, weights, epsilon, h, type)
        if payoff > best_payoff:
            best_payoff = payoff
            best_epsilon = epsilon

    return best_epsilon, best_payoff

def generateAdversarial(round_len, action_len):
    type_a = []
    cur = action_len
    i = 0
    while cur > 0:
        copy = 0
        while copy < cur:
            new = [0 for i in range(action_len)]
            new[i] = 1
            type_a.append(new)
            copy += 1
        i += 1
        cur -= 1

    type_b = []
    cur = action_len
    i = action_len - 1
    while cur > 0:
        copy = 0
        while copy < cur:
            new = [0 for i in range(action_len)]
            new[i] = 1
            type_b.append(new)
            copy += 1
        i -= 1
        cur -= 1

    arr = []
    i = 0
    a = True
    while len(arr) < round_len:
        if a:
            if i == len(type_a):
                i = 0
                a = False
                continue
            arr.append(type_a[i])
        else:
            if i == len(type_b):
                i = 0
                a = True
                continue
            arr.append(type_b[i])

        i+=1
    return arr

def generateAdversarial2(round_len, action_len):

    type_a = []
    val = 1/round_len
    count = -1
    i = int(action_len/2)
    while len(type_a) < round_len:

        if i == action_len:
            i = int(action_len/2)
            continue
        count += 1
        new = [0 for j in range(action_len)]
        new[i] = val * count
        i += 1
        type_a.append(new)
    return type_a


def calculateWeights(arr):
    weights = [[0 for i in range(len(arr[0]))]]
    weights_idx = 1
    for i in range(len(arr)):
        payoffs = arr[i]
        payoffs = np.add(payoffs, weights[weights_idx - 1])
        weights.append(payoffs)
        weights_idx += 1
    return weights

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

def generateLuckyStreak(round_len, action_len):
    last_chance = 0
    last_index = -1
    lucky_matrix = []
    used = []
    for i in range(round_len):
        arr = [0 for i in range(action_len)]
        chance = random.random()
        if chance < last_chance:

            last_chance -= 0.1
            arr[last_index] = 1
        else:
            last_index = random.randint(0, action_len-1)
            while last_index in used:
                last_index = random.randint(0, action_len - 1)
            used.append(last_index)
            if len(used) == action_len:
                used = []
            last_chance = 0.8
            arr[last_index] = 1
        lucky_matrix.append(arr)

    return lucky_matrix

def generateStrictLuckyStreak(round_len, action_len):
    lucky_matrix = []
    n = action_len
    n_copy = action_len
    sign = -1
    count = 0
    for i in range(round_len):
        arr = [0 for i in range(action_len)]
        if n_copy == 0:
            n += sign * 1
            if n == 0 and sign == -1:
                n = 1
                sign = sign * -1
                count += 1
            n_copy = n
        arr[n-1] = 1
        n_copy += sign * 1
        lucky_matrix.append(arr)
    return lucky_matrix

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


#conclusions: monte carlo sampling for lucky streaks is less accurate for smaller amounts due to removing patterns.
#for large samples, it is mostly the same however
#for smaller samples, increasing epsilon is better since it will learn patterns from the previous one
#best epsilon is around 1

#why for larger samples, the epsilon doesn't really matter?


round_len = 1000 #n
action_len = 10 #k

def generateAdversarialFair(round_len, action_len):
    actions_total_payoff = [[0, i] for i in range(action_len)]
    actions_total_payoff_stable = [0 for i in range(action_len)]


    heapq.heapify(actions_total_payoff)


    random_payoffs = np.random.rand(round_len, 1).squeeze()
    # print(random_payoffs)

    round_payoff_matrix = np.zeros((round_len, action_len))
    weight_matrix = []
    exp_weight_matrix = []
    weight_matrix.append(np.copy(actions_total_payoff_stable))
    for i in range(round_payoff_matrix.shape[0]):
        new_payoff = random_payoffs[i]
        min_payoff = heapq.heappop(actions_total_payoff)
        idx = min_payoff[1]
        actions_total_payoff_stable[idx] += new_payoff
        min_payoff[0] += new_payoff
        heapq.heappush(actions_total_payoff, min_payoff)
        # print(actions_total_payoff)
        exp_weight_matrix.append(exponentialWeights(actions_total_payoff_stable, 0.05, 1))
        weight_matrix.append(np.copy(actions_total_payoff_stable))
        round_payoff_matrix[i][idx] = new_payoff

    return round_payoff_matrix, weight_matrix
# print(followTheLeaderRegularized(round_payoff_matrix))
# print(calculateExpectedPayoff(round_payoff_matrix, weight_matrix, 100, 1))
# print(optimal(round_payoff_matrix))
# total = 0
# monte_total = 0
# for i in range(20):
#     lucky_streak = generateLuckyStreak(round_len, action_len)
#     weights = calculateWeights(lucky_streak)
#     monte_total += calculateEmpiricalEpsilonMonte(lucky_streak, 100, 1, 1)[0]
#     total += calculateEmpiricalEpsilonExact(lucky_streak, weights, 1)
#
# print(calculateTheoreticalEpsilon(round_len, action_len))
# print(total/20)
# print(monte_total/20)



#print(bernoulli_matrix)
# print(bernoulli_exp_weight_matrix)
bernoulli_matrix = np.random.rand(round_len, action_len)
bernoulli_matrix = np.divide(bernoulli_matrix, 2)
bernoulli_weights = calculateWeights(bernoulli_matrix)
def compareLinearExp(data="fair"): #fair, bernoulli, lucky
    rounds = [5, 10, 100, 1000, 2000]
    trials = 100
    linear = []
    exp = []
    for round_len in rounds:
        linear_avg = 0
        exp_avg = 0
        for i in range(trials):
            arr = []
            weights = []
            h = -1
            if data == "fair":
                arr, weights = generateAdversarialFair(round_len, action_len)
                h = 1
            elif data == "bernoulli":
                arr = np.random.rand(round_len, action_len)
                arr = np.divide(arr, 2)
                weights = calculateWeights(arr)
                h = 0.5
            elif data == "lucky":
                arr = generateAdversarial2(round_len, action_len)
                weights = calculateWeights(arr)
                h = 1

            theoretical_epsilon = calculateTheoreticalEpsilon(round_len, action_len)
            linear_avg += calculateExpectedPayoff(arr, weights, theoretical_epsilon, h, "linear")
            exp_avg += calculateExpectedPayoff(arr, weights, theoretical_epsilon, h, "exp")

        linear.append(np.divide(linear_avg, trials))
        exp.append(np.divide(exp_avg, trials))

    plt.plot(rounds, np.divide(linear, rounds), label = "Linear Weights")
    plt.plot(rounds, np.divide(exp, rounds), label = "Exponential Weights")
    plt.title(f"Average Payoff/Rounds over 100 Trials for Gen Adv. dataset")
    plt.ylabel("Payoff/Round")
    plt.xlabel("Rounds")
    plt.legend()
    print(linear, exp)
    plt.show()

# compareLinearExp("lucky")
# compareLinearExp("bernoulli")
# compareLinearExp("lucky")
# print(theoretical_epsilon)
# print(calculateEmpiricalEpsilonExact(bernoulli_matrix, bernoulli_weights, 0.5))

# print(calculateExpectedPayoff(bernoulli_matrix, bernoulli_weights, 0, 0.5))
# print(optimal(bernoulli_matrix))

#
# summed = np.sum(bernoulli_matrix, axis=0)

# print(bernoulli_weights)
# total = 0
# for i in range(20):
#     print(calculateEmpiricalEpsilonExact(bernoulli_matrix, bernoulli_weights, 0.5))

# print(calculateExpectedPayoff(bernoulli_matrix, bernoulli_weights, 2, 0.5))
# print(total)
# print(total/20)

lucky_streak = generateStrictLuckyStreak(round_len, action_len)
# print(lucky_streak)
weights = calculateWeights(lucky_streak)
# print(lucky_streak)
# print(weights)
# print(calculateExpectedPayoff(lucky_streak, weights, 1, 1))
trials = 100
# print(optimal(lucky_streak))
# print(f"Theoretical Epsilon: {calculateTheoreticalEpsilon(round_len, action_len)}")
# epsilons = np.arange(0.01, 0.2, 0.005)
# best_epsilon = -1
# best_payoff = -1
#
# for epsilon in epsilons:
#     weights = calculateWeights(lucky_streak)
#     payoff = calculateExpectedPayoff(lucky_streak, weights, epsilon, 1)
#     if payoff > best_payoff:
#         best_payoff = payoff
#         best_epsilon = epsilon
#
# print(best_epsilon, best_payoff)
# print(optimal(lucky_streak))
# print(followTheLeader(lucky_streak)[0])
# print(f"Empirical Epsilon with {trials} trials: {calculateEmpiricalEpsilonExact(lucky_streak, weights, 1)}")

# rounds = [5, 10, 100, 1000, 2000]
# trials = 20
# ftl= []
# theoretical_best = []
# random = []
# for round_len in rounds:
#     ftl_avg = 0
#     tb_avg = 0
#     random_avg = 0
#     tb = calculateTheoreticalEpsilon(round_len, action_len)
#     for i in range(trials):
#         arr = np.random.rand(round_len, action_len)
#         arr = np.divide(arr, 2)
#         weights = calculateWeights(arr)
#         ftl_avg += followTheLeader(arr)[0]
#         tb_avg += calculateExpectedPayoff(arr, weights, tb, 0.5)
#         random_avg += calculateExpectedPayoff(arr, weights, 0, 0.5)
#     ftl.append(ftl_avg/trials)
#     theoretical_best.append(tb_avg/trials)
#     random.append(random_avg/trials)
#
# plt.plot(rounds, np.divide(ftl, rounds), label="FTL")
# plt.plot(rounds, np.divide(theoretical_best, rounds), label = "Theoretical Best Epsilon")
# plt.plot(rounds, np.divide(random, rounds), label = "No Epsilon")
# plt.title("Averaged Payoff/Rounds over 100 Trials for Bernoulli distribution")
# plt.ylabel("Payoff/Round")
# plt.xlabel("Rounds")
# plt.xlim(xmin = 10)
# plt.legend()
# plt.show()
# rounds = [5, 10, 100, 1000, 2000]
# trials = 20
# ftl= []
# theoretical_best = []
# random = []
# opt = []
# for round_len in rounds:
#     ftl_avg = 0
#     tb_avg = 0
#     random_avg = 0
#     opt_avg = 0
#     tb = calculateTheoreticalEpsilon(round_len, action_len)
#     for i in range(trials):
#         arr = generateAdversarial2(round_len, action_len)
#         weights = calculateWeights(arr)
#         ftl_avg += followTheLeader(arr)[0]
#         tb_avg += calculateExpectedPayoff(arr, weights, tb, 0.5)
#         random_avg += calculateExpectedPayoff(arr, weights, 0, 0.5)
#         opt_avg += optimal(arr)[1]
#     opt.append(opt_avg/trials)
#     ftl.append(ftl_avg/trials)
#     theoretical_best.append(tb_avg/trials)
#     random.append(random_avg/trials)
#
# opt = np.divide(opt, rounds)
# ftl = np.divide(ftl, rounds)
# theoretical_best = np.divide(theoretical_best, rounds)
# random = np.divide(random, rounds)
#
# plt.plot(rounds, opt-ftl, label="FTL")
# plt.plot(rounds, opt-theoretical_best, label = "Theoretical Best Epsilon")
# plt.plot(rounds, opt-random, label = "No Epsilon")
# plt.title("Averaged Per Round Regret over 20 Trials for Generated Adv.")
# plt.ylabel("Regret/Round")
# plt.xlabel("Rounds")
# plt.xlim(xmin = 10)
# plt.legend()
# plt.show()

actions = [3, 5, 10, 20]
tb = []
eb = []
eb_p = []
tb_p = []
opt = []
trials = 10
for action_len in actions:
    e = 0
    tb_e = calculateTheoreticalEpsilon(round_len, action_len)
    tb.append(tb_e)
    for i in range(trials):
        arr = generateAdversarial2(round_len, action_len)
        weights = calculateWeights(arr)
        coupled = calculateEmpiricalEpsilonMonte(arr, round_len, 1)
        e += coupled[0]
        eb_p.append(coupled[1])
        tb_p.append(calculateExpectedPayoff(arr, weights, tb_e, 1))
        opt.append(optimal(arr)[1])
    eb.append(e/trials)

print(opt)
print(eb_p)
print(tb_p)
plt.plot(actions, eb)
plt.plot(actions, tb)
plt.title("Empirical Best Epsilon vs Theoretical Best at k=10")
plt.show()
