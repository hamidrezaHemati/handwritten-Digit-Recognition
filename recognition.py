import numpy as np
import matplotlib.pyplot as plt
import random as random


#global variables
e = 2.71


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading The Train Set
train_set = []
def train_set_reader(manual_number = False):
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    if not manual_number:
        num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    else:
        num_of_train_images = manual_number
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))

# Reading The Test Set
test_set = []
def test_set_reader():
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)


    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))


# Plotting an image
def plot_image():
    manual_number = 100
    counter = 0;
    train_set_reader(manual_number)
    # for i in range(manual_number):
    #     show_image(train_set[i][0])
    #     plt.show()
    #     label = np.where(train_set[i][1] == np.amax(train_set[i][1]))
    #     print(int(label[0]))
    #     counter += 1
    # print("counter: ", counter)


def sigmoid_deriv(x):
    return 1 / (1 + (pow(e, (-1 * x))))


def calculus(neuron, weight, bias):
    weight_neuron_multipy = weight @ neuron
    zigma = weight_neuron_multipy + bias
    zigma = sigmoid_deriv(zigma)
    # print("zigma after sigmoed: ", zigma)
    return zigma



def forward_chaining():
    correctness = []
    for i in range(100):
        # weight matrix's
        weight_W = np.random.normal(size=(16, 784))
        weight_V = np.random.normal(size=(16, 16))
        weight_Q = np.random.normal(size=(10, 16))

        # bias vectors
        bias_b0 = np.random.uniform(0, 0, (16, 1))
        bias_b1 = np.random.uniform(0, 0, (16, 1))
        bias_b2 = np.random.uniform(0, 0, (10, 1))

        correct_counter = 0
        for img in range(100):
            image = train_set[img][0]
            neuron_A0 = np.array(image)
            neuron_A1 = calculus(neuron_A0, weight_W, bias_b0)
            neuron_A2 = calculus(neuron_A1, weight_V, bias_b1)
            neuron_A3 = calculus(neuron_A2, weight_Q, bias_b2)
            print(neuron_A3)
            max_number = neuron_A3.max()
            print("max number: ", max_number)
            max_index = 0
            for i in range(10):
                if max_number == neuron_A3[i][0]:
                    max_index = i
                    break
            print("index of the max number: ", max_index)

            print(np.where(train_set[img][1] == np.amax(train_set[img][1])))
            label = np.where(train_set[img][1] == np.amax(train_set[img][1]))
            label = label[0]
            if label == max_index:
                correct_counter += 1
            print("lable: ", int(label[0]))
            print()

        print(correct_counter, "%")
        correctness.append(correct_counter)

    plt.plot([x for x in range(100)], correctness)
    plt.show()


def main():
    plot_image()
    forward_chaining()



main()