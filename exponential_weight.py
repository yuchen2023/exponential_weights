import numpy as np
import matplotlib.pyplot as pyplot
import random
import heapq



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

def calculateWeights(arr):
    weights = [[0 for i in range(len(arr[0]))]]
    weights_idx = 1
    for i in range(len(arr)):  # number of monte carlo trials
        payoffs = arr[i]
        payoffs = np.add(payoffs, weights[weights_idx - 1])
        weights.append(payoffs)
        weights_idx += 1
    return weights

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


def linearWeights(arr, epsilon, h): #for comparison with linear weights
    sum = 0
    out = []
    for ele in arr:
        numerator = (1+epsilon) * ele /h
        sum += numerator
        out.append(numerator)

    return np.divide(out,sum)

def generateLuckyStreak(trials, round_len):
    last_chance = 0
    last_index = -1
    lucky_matrix = []
    for i in range(trials):
        arr = [0 for i in range(round_len)]
        chance = random.random()

        if chance < last_chance:

            last_chance -= 0.1
            arr[last_index] = 1
        else:
            last_index = random.randint(0, round_len-1)
            last_chance = 0.8
            arr[last_index] = 1
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
# round_len = 10 #n
# action_len = 4 #k

#conclusions: monte carlo sampling for lucky streaks is less accurate for smaller amounts due to removing patterns.
#for large samples, it is mostly the same however
#for smaller samples, increasing epsilon is better since it will learn patterns from the previous one
#best epsilon is around 1

#why for larger samples, the epsilon doesn't really matter?


round_len_arr = [10, 100,500, 1000, 2000, 4000] #n
action_len_arr = [5, 10, 100, 500] #k

for action_len in action_len_arr:
    empirical_epsilon =[]
    for round_len in round_len_arr:
        empirical_epsilon_exact_adversial = 0
        for i in range(100):

            actions_total_payoff = [[0, i] for i in range(action_len)]
            actions_total_payoff_stable = [0 for i in range(action_len)]


            heapq.heapify(actions_total_payoff)


            random_payoffs = np.random.rand(round_len, 1).squeeze()
            print('random payoffs\n',random_payoffs)


            round_payoff_matrix = np.zeros((round_len, action_len))
            exp_weight_matrix = []
            for i in range(round_payoff_matrix.shape[0]):
                new_payoff = random_payoffs[i]
                min_payoff = heapq.heappop(actions_total_payoff)
                idx = min_payoff[1]
                actions_total_payoff_stable[idx] += new_payoff
                min_payoff[0] += new_payoff
                heapq.heappush(actions_total_payoff, min_payoff)
                # print(actions_total_payoff)
                exp_weight_matrix.append(exponentialWeights(actions_total_payoff_stable, 0.05, 1))
                round_payoff_matrix[i][idx] = new_payoff
                empirical_epsilon_exact_adversial+= calculateEmpiricalEpsilonExact(round_payoff_matrix, exp_weight_matrix, 0.5)/100
            empirical_epsilon.append(empirical_epsilon_exact_adversial)
        pyplot.plot(round_len_arr, empirical_epsilon, label=str(action_len) + "actions")

print("empirical_epsilon\n",empirical_epsilon)
pyplot.title("Adversial Fair Method - Empirical Epsilon over 100 trials")
pyplot.xlabel("Trial count")
pyplot.ylabel("epsilon")
pyplot.legend()
pyplot.show()



#Bernoulli payoffs




for action_len in action_len_arr:
    empirical_epsilon =[]
    for round_len in round_len_arr:
        empirical_epsilon_exact_bernoulli = 0
        empirical_epsilon_exact_luckystreak = 0
        for i in range(100):
            bernoulli_matrix = np.random.rand(round_len, action_len)
            bernoulli_matrix = np.divide(bernoulli_matrix, 2)

            print("round = ",round_len, "\taction = ", action_len)
            # print(bernoulli_matrix)


            bernoulli_weights = calculateWeights(bernoulli_matrix)
            print("\ntheoretical epsilon = ",calculateTheoreticalEpsilon(round_len, action_len))
            # pyplot.plot(range(1, 50),calculateTheoreticalEpsilon(round_len, action_len))

            summed = np.sum(bernoulli_matrix, axis=0)
            # print(summed)
            # print(np.argmax(summed))
            # print("calculateEmpiricalEpsilonExact =",calculateEmpiricalEpsilonExact(bernoulli_matrix, bernoulli_weights, 0.5))
            # empirical_epsilon_exact_bernoulli+= calculateEmpiricalEpsilonExact(bernoulli_matrix, bernoulli_weights, 0.5)/100

            lucky_streak = generateLuckyStreak(round_len, action_len)
            weights = calculateWeights(lucky_streak)
            # print("lucky_streak\n",lucky_streak)
            # print("weights = ",weights)
            print("calculateExpectedPayoff \n",calculateExpectedPayoff(lucky_streak, weights, 1, 1))
            empirical_epsilon_exact_luckystreak += calculateEmpiricalEpsilonExact(lucky_streak, weights, 0.5)/100
            trials = 100
            # print(optimal(lucky_streak))
            print(f"Theoretical Epsilon: {calculateTheoreticalEpsilon(round_len, action_len)}")
            epsilons = np.arange(0.01, 0.2, 0.01)
            best_epsilon = -1
            best_payoff = -1

            for epsilon in epsilons:
                weights = calculateWeights(lucky_streak)
                payoff = calculateExpectedPayoff(lucky_streak, weights, epsilon, 1)
                if payoff > best_payoff:
                    best_payoff = payoff
                    best_epsilon = epsilon

            print("\n best epsilon and best payoff =",best_epsilon, best_payoff)
            print(f"\nEmpirical Epsilon with {trials} trials: {calculateEmpiricalEpsilonMonte(lucky_streak, trials, 1, consecutive=1)}")
            # emp_monte_carlo_result = calculateEmpiricalEpsilonMonte(lucky_streak, trials, 1, consecutive=1)[1]
            # print("emp_monte_carlo_result = ",emp_monte_carlo_result)
        # empirical_epsilon.append(empirical_epsilon_exact_bernoulli)
        empirical_epsilon.append(empirical_epsilon_exact_luckystreak)
    pyplot.plot(round_len_arr, empirical_epsilon, label=str(action_len) + "actions")

print("empirical_epsilon\n",empirical_epsilon)
pyplot.title("Lucky Streak - Empirical Epsilon over 100 trials")
pyplot.xlabel("Trial count")
pyplot.ylabel("epsilon")
pyplot.legend()
pyplot.show()


# pyplot.title("Empirical Epsilon with 100 trials")
# pyplot.xlabel("Trial Number")
# pyplot.ylabel("Expected Probability")
# pyplot.plot(emp_monte_carlo_result[1])
# pyplot.legend()
# pyplot.show()
