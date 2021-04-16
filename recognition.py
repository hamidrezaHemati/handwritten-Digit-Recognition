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
    for i in range(manual_number):
        show_image(train_set[i][0])
        plt.show()
        label = np.where(train_set[i][1] == np.amax(train_set[i][1]))
        print(int(label[0]))
        counter += 1
    print("counter: ", counter)


def sigmoed_function(x):
    return 1 / (1 + (pow(e, (-1 * x))))


weight_W = np.random.uniform(0,1,(16,784))
weight_V = np.random.uniform(0, 1, (16, 16))
weight_Q = np.random.uniform(0, 1, (10, 16))
bias_b0 = np.random.uniform(0,0, (784, 1))
bias_b1 = np.random.uniform(0, 0, (15, 1))
bias_b2 = np.random.uniform(0, 0, (10, 1))

def main():
    print(weight_Q)



main()