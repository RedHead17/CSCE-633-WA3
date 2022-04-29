# CSCE633 Written Assignment 3 Code
# Mary Julian (622004026)
# Sara Callahan (731008655)

from copy import deepcopy
import mnist_loader
from network import Network
import numpy as np
import random

def Question1a(_epochs, n, batch_size, lr):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    num_test_points = 10000 # len(list(test_data))
    net = Network([784, n, 10])
    results = net.SGD(training_data, _epochs, batch_size, lr, test_data=test_data)
    test_acc_perc = results["test_acc"].pop() / num_test_points
    return test_acc_perc

def Question1b(_epochs, n, batch_arr, lr_arr):
    result_arr = np.zeros([len(batch_arr)*len(lr_arr), 3])
    # Uses grid search over all combos of lr and batch size
    i = 0
    for batch_size in batch_arr:
        for lr in lr_arr:
            net = Network([784, n, 10])
            training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
            results = net.SGD(training_data, _epochs, batch_size, lr, test_data=validation_data)
            result_arr[i, 0] = lr
            result_arr[i, 1] = batch_size
            result_arr[i, 2] = results["test_acc"].pop()
            print("Learning Rate: {0}, Batch Size: {1}, Validation Accuracy: {2}".format(lr, batch_size, result_arr[i, 2]))
            i += 1

    max_index = result_arr[np.argmax(result_arr[:, 2]), :]
    opt_accuracy = max_index[2]
    opt_lr = max_index[0]
    opt_bs = max_index[1]
    return {'opt_acc': opt_accuracy, 'opt_lr': opt_lr, 'opt_bs': opt_bs}

def Question2(_epochs, n, batch_size, lr, m, sample_size):
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
        results = net.SGD(training_set, _epochs, batch_size, lr, test_data=testing_set) # Use out of bag error
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
        print("Full Ensemble: m={0}, Test Accuracy: {1}".format(j+1, ens_acc))

        # Report the results for the learners' running average
        test_acc_sum += results["test_acc"].pop()
        train_acc_sum += results["train_acc"].pop()
        test_current_avg = (test_acc_sum / (j+1)) / len(test_results)
        train_current_avg = (train_acc_sum / (j+1)) / len(sample_results)
        print("Average of Learners: m={0}, Test Accuracy is {1}, Train Accuracy is {2}".format(
            j+1, test_current_avg, train_current_avg))
    return 0

## Main, running the question code functions
## dimensions [size][x/y][784][1]
q_epochs = 30
n_val = 9 # (731008655 + 622004026) mod 24 + 4 = 9 + 4 = 13
# Question 1a: Find test accuracy (correct/total classifications) after 30 epochs.
question_1a_res = Question1a(q_epochs, n_val, 10, 3.0)
print("Question 1a: Test Accuracy after {0} epochs: {1}".format(q_epochs, question_1a_res))

# Default tuned lr and batch size if 1b is commented out.
opt_bs_val = 8
opt_lr_val = 1.5

# Question 1b: Tune minibatch and learning rate parameters over validation data.
mini_size_array = [2, 4, 8, 16, 32]
learn_rate_array = [0.1, 1.0, 1.5, 2.0, 3.0]
question_1b_res = Question1b(q_epochs, n_val, mini_size_array, learn_rate_array)
opt_accuracy = question_1b_res["opt_acc"]
opt_bs_val = question_1b_res["opt_bs"]
opt_lr_val = question_1b_res["opt_lr"]
print("Question 1b (Tuned Parameters): Learning Rate: {0}, Batch Size: {1}, Validation Accuracy: {2}".format(opt_lr_val, opt_bs_val, opt_accuracy))

# Question 2: Use Bagging - m from 1-20
m_val = 20
samples = 50000 # Still using the original sample size
Question2(q_epochs, n_val, opt_bs_val, opt_lr_val, m_val, samples)

# Prompt so we can see the output after done.
input("Press Enter to continue...")