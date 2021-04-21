import numpy as np
from random import seed
import random
from random import choice
from random import random
from keras.datasets import mnist
import matplotlib.pyplot as plt
from math import exp
from numpy.random import rand
import imageio
import glob
from math import sqrt
import pandas as pd
import time

np.random.seed(10)
"""
train_data = open('mnist_train.csv', 'r')

df = pd.DataFrame(train_data)
tra = df[0:10]
"""



(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

print('train data=', len(x_train_image))
print('test data=', len(x_test_image))
print('x_train_image :', x_train_image.shape)
print('y_train_label :', y_train_label.shape)


""""
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()


def plot_images_labels(images, labels, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
        idx += 1
    plt.show()

"""

x_Train = x_train_image.reshape(60000, 28*28).astype('float32')
x_Test = x_test_image.reshape(10000, 28*28).astype('float32')
x_Train = x_Train/255
x_Test = x_Test/255

data = [[0] * 785] * 60000
test_data = [[0] * 785] * 10000

for i in range(60000):
    data[i] = np.append(x_Train[i], y_train_label[i]/10)

for i in range(10000):
    test_data[i] = np.append(x_Test[i], y_test_label[i]/10)


def initialize_network(n_inputs, n_hidden, n_hidden1, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    hidden_layer1 = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_hidden1)]
    network.append(hidden_layer1)
    output_layer = [{'weights': [random() for i in range(n_hidden1 + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def initialize_network_1(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    if activation >= 0:
        return activation/50
    else:
        return 0


def forward_pro(net, rows):
    inputs = rows
    sum = 0
    for layer in net:
        new_inputs = []
        if len(layer) != 10:
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
        else:
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['exp'] = exp(activation) #math range error
                sum += neuron['exp']
            for neuron in layer:
                neuron['output'] = neuron['exp'] / sum
                new_inputs.append(neuron['output'])
        inputs = new_inputs
        sum = 0
    return inputs


def derivative(out):
    if out > 0:
        return out
    else:
        return 0
    #return out * (1.0 - out)


def ini_delta(network):
    for i in range(len(network)):
        layer = network[i]
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['last_delta'] = 0
            neuron['G'] = 0


def bp(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = (errors[j] * derivative(neuron['output']))  ## delta == g(t) #+ neuron['last_delta'])  #aj(1-aj)

            neuron['G'] = neuron['G'] * 0.95 + 0.05 * neuron['delta'] ** 2  # adadelta

            #neuron['G'] += neuron['delta'] ** 2   # adagrad


def update_weights(network, row, rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += (rate / (sqrt(neuron['G'] + 1e-8))) * neuron['delta'] * inputs[j]
                #neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]

            neuron['weights'][-1] += rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, rate, n_outputs):
    error = 100
    epoch = 0
    error_data = []
    y = []
    while error > 5:
        sum_error = 0
        for row in train:
            outputs = forward_pro(network, row)
            expected = [0 for i in range(n_outputs)]
            buffer = int(row[-1] * 10)
            expected[buffer] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            #sum_error += 0 - sum([expected[i] * np.log(outputs[i]) for i in range(len(expected))])
            bp(network, expected)
            update_weights(network, row, rate)
        epoch += 1
        error = sum_error
        error_data.append(error)
        print('>epoch->%d,  err->%.3f' % (epoch, sum_error))

    for i in range(len(error_data)):
        y.append(i)
    plt.plot(y, error_data, color='green', linestyle='dashed', linewidth=3, marker='.', markerfacecolor='blue', markersize=3)
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.title('error chart')


"""
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    i = 0
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_pro(network, row)
            expected = [0 for i in range(n_outputs)]
            buffer = int(row[-1] * 10)
            expected[buffer] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            #sum_error += 0 - sum([expected[i] * np.log(outputs[i]) for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

"""


def predict(network, row):
    outputs = forward_pro(network, row)
    return outputs.index(max(outputs))


def train_data(net, test_case):
    train = np.array([])
    buffer = []
    choi = [2, 3, 4, 5, 7]
    for i in range(0, test_case, 1):
        str1 = str('{0:04}'.format(i + 1))
        num = choice(choi)
        str1 = str(num) + "/" + str1 + '.png'
        im = imageio.imread(glob.glob(str1)[0])
        x_Train = im.reshape(2352).astype('float32')
        x_Train = x_Train / 255
        for j in range(0, len(x_Train), 3):
            y = 0.2126 * x_Train[j] + 0.7125 * x_Train[j + 1] + 0.0722 * x_Train[j + 2]
            buffer.append(y)
        buffer.append(num / 10)
        train = np.concatenate((train, buffer), axis=0)
        buffer.clear()
    train = np.reshape(train, (test_case, 785))
    train_network(net, train, 0.01, 10)


def test(net):
    answer = open('answer.txt', 'w+')
    buffer = []
    for i in range(1, 10, 1):
        str1 = str('{0:04}'.format(i+1))
        index = str1
        #str1 = str(i)
        str1 = str1 + '.png'
        im = imageio.imread(glob.glob(str1)[0])
        x_Train = im.reshape(2352).astype('float32')
        x_Train = x_Train / 255
        for j in range(0, len(x_Train), 3):
            y = 0.2126 * x_Train[j] + 0.7125 * x_Train[j+1] + 0.0722 * x_Train[j+2]
            buffer.append(y)
        answer.write(index + ' ' + str(predict(net, buffer)) + '\n')
        buffer.clear()


def graphing_data(error_data):
    for i in range(len(error_data)):
        y.append(i)
    plt.plot(y, error_data, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
             markersize=3)
    plt.xlabel('error')
    plt.ylabel('epoch')
    plt.title('error chart')
    plt.show()


X1 = data[0:30]
n_inputs = len(X1[0]) - 1
n_outputs = len(set([row[-1] for row in X1]))
network = initialize_network_1(n_inputs, 32, n_outputs)
ini_delta(network)

#train_data(network, 100)
#print(network)
#test(network)


index = int(rand() * 60000)
x = data[index: index + 200]
start_time = time.time()
train_network(network, x, 0.01, n_outputs)
print(time.time() - start_time)

correct = 0
y = test_data[0:1000]
for row in y:
    prediction = predict(network, row)
    if int(row[-1] * 10) == prediction: correct += 1
print(correct / 1000)

plt.show()

