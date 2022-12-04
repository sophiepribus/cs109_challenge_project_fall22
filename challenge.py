import numpy as np
from scipy.special import softmax
import csv

'''
Steps for multiclass logistic regression

1. Import dataset and format as a input matrix (X) and class outcome matrix (Y)
2. Generate an initial randomized weight matrix (W) of appropriate size
3. Multiply X and W to get Z
4. Apply softmax to Z to get P
5. Use previous components to calculate gradient descent of W and update W
6. Repeat steps 1-5 many, many times.
'''

# number of times to run the gradient descent step
NUM_ITERATIONS = 1000
# number of output classes for simplicity throughout document
CLASSES = 11

# one-hot encoding function
def encode_class_matrix(Y, n):
    encoded = np.zeros((n, CLASSES)).astype(int)
    for i in range(n):
        encoded[i, Y[i]] = 1
    return encoded

# calculate the gradient given X (input matrix), Y (output matrix), W (current weight matrix), and n (number of data points)
def gradient(X, Y, W, n):
    Z = np.matmul(X, W)
    P = softmax(Z, axis=1)
    grad = (1/n)*np.matmul(X.transpose(), (encode_class_matrix(Y, n)) - P)
    return grad

# calculate gradient descent given X (input matrix), Y (output matrix), W (original weight matrix),
# n (number of data points), and step size
def gradient_descent_train(X, W, Y, n, step):
    for i in range(NUM_ITERATIONS):
        # update W
        W += (step*gradient(X, Y, W, n))
    return W

# convert provided input csv file to a matrix of float entries
def file_to_matrix_X(file):
    # import X file
    file_list = []
    n = 0
    with open(file) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            file_list.append(row)
            n += 1
        m = len(row)
    # convert to matrix for later use
    array = np.array(file_list)
    float_array = array.astype(float)
    matrix = np.asmatrix(float_array)
    return matrix, n, m

# convert provided output csv file to a matrix of integer entries
# note that entries must be converted to integers for the later one-hot encoding function
def file_to_matrix_Y(file):
    # import Y file
    file_list = []
    with open(file) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            file_list.append(row)
    # convert to matrix for later use
    array = np.array(file_list)
    float_array = array.astype(int)
    matrix = np.asmatrix(float_array)
    return matrix

# test accuracy and top-k accuracy of the algorithm given X (input matrix), Y (output matrix), W (optimized weight
# matrix), n (number of data points), and k (top-k value)
def test(X, W, Y, n, k):
    Z = np.matmul(X, W)
    P = softmax(Z)
    correct_count = 0
    top_k_count = 0
    for i in range(n):
        predicted = np.argmax(P[i])
        top_k_idx = np.argpartition(P[i], -k)[-k:]
        if predicted == Y[i]:
            correct_count += 1
        if Y[i] in top_k_idx:
            top_k_count += 1
    return correct_count/n, top_k_count/n

def main():
    # convert input file to X matrix
    X, n, m = file_to_matrix_X("training_input.csv")
    # convert output file Y matrix
    Y = file_to_matrix_Y("training_output.csv")
    # generate initial W matrix
    w_array = np.zeros((m, CLASSES))
    W = np.asmatrix(w_array)
    # assign step size
    step = 0.0001
    # train algorithm to calculate W
    updated_W = gradient_descent_train(X, W, Y, n, step)
    # use W to predict on new data
    test_X, n, m = file_to_matrix_X("testing_input.csv")
    test_Y = file_to_matrix_Y("testing_output.csv")
    # set top-k value
    k = 5
    # calculate overall accuracy and top-K accuracy
    accuracy, top_k_accuracy = test(test_X, updated_W, test_Y, n, k)
    print("Accuracy: " + str(accuracy))
    print("Top-K Accuracy: " + str(top_k_accuracy))
    print('\n\n')

if __name__ == '__main__':
    main()