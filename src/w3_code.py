# CSCE633 Written Assignment 3 Code
# Mary Julian (622004026)
# Sara Callahan (731008655)

from asyncio.windows_events import NULL
from audioop import avg
from copy import deepcopy
import mnist_loader
from network import Network
import numpy as np
import random

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

## dimensions [size][x/y][784][1]
epochs = 30
n = 9 # (731008655 + 622004026) mod 24 + 4 = 9 + 4 = 13
net = Network([784, n, 10])
# net = network.SGD([784, n, 10])
# net.SGD(training_data, epochs, 10, 3.0, test_data=test_data)
 # Question 1: Find test accuracy (correct/total classifications) after 30 epochs.

 # Question 1b: Tune minibatch and learning rate parameters over validation data.
# mini_size_array = [2, 4, 8, 16, 32]
# learn_rate_array = [0.1, 1.0, 1.5, 2.0, 3.0]
# result_arr = np.zeros([len(mini_size_array)*len(learn_rate_array), 3])
# # Uses telescoping search?
# i = 0
# for batch_size in mini_size_array:
#     for lr in learn_rate_array:
#         net = Network([784, n, 10])
#         training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#         results = net.SGD(training_data, epochs, batch_size, lr, test_data=validation_data)
#         result_arr[i, 0] = lr
#         result_arr[i, 1] = batch_size
#         result_arr[i, 2] = results["test_acc"].pop()
#         print("Learning Rate: {0}, Batch Size: {1}, Validation Accuracy: {2}".format(lr, batch_size, result_arr[i, 2]))
#         i += 1

# max_index = result_arr[np.argmax(result_arr[:, 2]), :]
# opt_accuracy = max_index[2]
opt_lr = 1.5 #max_index[0]
opt_bs = 8 #max_index[1]
# print("Tuned Parameters: Learning Rate: {0}, Batch Size: {1}, Validation Accuracy: {2}".format(opt_lr, opt_bs, opt_accuracy))

# Question 2: Use Bagging - m from 1-20
m = 20
sample_size = 50000 # Still using the original sample size
learners = []
training_data, validation_data, test_data = mnist_loader.load_sample_set_wrapper()
full_dataset = list(training_data)+list(validation_data) # list(test_data)
test_acc_sum = 0
train_acc_sum = 0
all_learner_res = []
ensemble_testset = list(deepcopy(test_data))
y_list = [(y) for (x, y) in ensemble_testset]

for j in range(m):
    full_copy = deepcopy(full_dataset)
    ensemble_testset = list(deepcopy(test_data))
    random.shuffle(full_copy)
    sample_set = full_copy[0:sample_size] # Using sample w/ replacement
    out_of_bag = full_copy[sample_size:]
    
    # Reshape the training zip list from the sample set. Probably an easier way to do this?
    sample_inputs = [np.reshape(list(x)[0], (784, 1)) for x in sample_set]
    sample_results = [mnist_loader.vectorized_result(list(x)[1]) for x in sample_set]
    training_set = zip(sample_inputs, sample_results)
    
    # Reshape the testing zip list from the out of bag set
    test_inputs = [np.reshape(list(x)[0], (784, 1)) for x in out_of_bag]
    test_results = [list(x)[1] for x in out_of_bag]
    testing_set = zip(test_inputs, test_results)
    
    # Create and train the learner
    net = Network([784, n, 10])
    results = net.SGD(training_set, epochs, opt_bs, opt_lr, test_data=testing_set) # Use out of bag error
    learners.append(net)

    # Perform the testing on the full ensemble to find total test error
    learner_results = [net.feedforward(x) for (x, y) in ensemble_testset]
    # Add the guess of the current ensemble
    if len(all_learner_res) == 0:
        all_learner_res = learner_results
    else:
        for k in range(len(learner_results)):
            all_learner_res[k] += learner_results[k]

    ensemble_results = [np.argmax(y) for (y) in all_learner_res] # Calculate the full ensemble's guess
    ensemble_tuple = zip(ensemble_results, y_list)

    # Find total test error
    ens_test_results = list(ensemble_tuple)
    acc_val = sum(int(x == y) for (x, y) in ens_test_results)
    ens_acc = acc_val / len(y_list)
    print("Full Ensemble, m={0}, test accuracy: {1}".format(j+1, ens_acc))

    # Report the results for the learners' average
    test_acc_sum += results["test_acc"].pop()
    train_acc_sum += results["train_acc"].pop()
    test_current_avg = (test_acc_sum / (j+1)) / len(test_results)
    train_current_avg = (train_acc_sum / (j+1)) / len(sample_results)
    print("One learner, m={0}, Test Accuracy is {1}, Train Accuracy is {2}".format(
        j+1, test_current_avg, train_current_avg))

# Prompt so we can see the output after done.
input("Press Enter to continue...")