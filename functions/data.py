'''
A variety of functions for managing the loading of MNIST data.
'''

import os
import pickle
import numpy as np
from struct import unpack
import tqdm

from numba import jit

def get_labeled_data(picklename, bTrain = True, MNIST_data_path='./mnist'):
    """ Read input-vector (image) and target class (label, 0-9) and return
        it as list of tuples.
        picklename: Path to the output pickle file.
        bTrain: True if training data, else False for test data.
        MNIST_data_path: Directory containing the MNIST files.
    """
    if os.path.isfile('{}.pickle'.format(picklename)):
        data = pickle.load(open('{}.pickle'.format(picklename), mode='rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(os.path.join(MNIST_data_path, 'train-images.idx3-ubyte'), mode='rb')
            labels = open(os.path.join(MNIST_data_path, 'train-labels.idx1-ubyte'), mode='rb')
        else:
            images = open(os.path.join(MNIST_data_path, 't10k-images.idx3-ubyte'), mode='rb')
            labels = open(os.path.join(MNIST_data_path, 't10k-labels.idx1-ubyte'), mode='rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise ValueError('The number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        print('Unpacking {} images...'.format('training' if bTrain else 'test'))
        for i in tqdm.tqdm(range(N)):
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("{}.pickle".format(picklename), "wb"))
    return data


def get_matrix_from_file(fileName, ending, n_input, number_exc_neurons, number_inh_neurons):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = number_exc_neurons
        else:
            n_src = number_inh_neurons
    if fileName[-1-offset]=='e':
        n_tgt = number_exc_neurons
    else:
        n_tgt = number_inh_neurons
    readout = np.load(fileName)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))

    print('get_matrix_from_file')
    print(readout.shape)
    print(value_arr.shape)
    print(n_src, n_tgt)
    # print('readout 0', readout[:, 0])
    # print('readout 1',readout[:, 1])
    # print('readout 2',readout[:, 2])

    if not readout.shape == (0,):
        for i in range(len(readout)):
            row = np.int32(readout[i,0])
            col = np.int32(readout[i,1])
            if (row >= n_src): continue
            if (col >= n_tgt): continue

            value_arr[row, col] = readout[i,2]


        # value_arr[np.int32(readout[0:n_src,0]), np.int32(readout[0:n_tgt,1])] = readout[:,2]
    return value_arr

def save_connections(save_conns, connections, data_path, ending = ''):
    print('save connections')
    for connName in save_conns:
        print('Save:', connName)
        print(type(connections[connName]))
        print(type(connections[connName].i))
        conn = connections[connName]
        connListSparse = zip(conn.i[:], conn.j[:], conn.w[:]) # conn = connections[connName]
        print(type(connListSparse)) 

        np.save(data_path + '\\output\\' + connName + ending, connListSparse)


def save_theta(population_names, neuron_groups, data_path, ending = ''):
    print('save theta')
    for pop_name in population_names:
        np.save(data_path + '\\output\\theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)


def normalize_weights(connections, weight, number_exc_neurons):
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            len_source = len(connections[connName].source)
            len_target = len(connections[connName].target)
            connection = np.zeros((len_source, len_target))
            connection[connections[connName].i, connections[connName].j] = connections[connName].w
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in range(number_exc_neurons):#
                temp_conn[:,j] *= colFactors[j]
            connections[connName].w = temp_conn[connections[connName].i, connections[connName].j]
    
    return connections


def get_2d_input_weights(connections, n_input, number_exc_neurons):
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, number_exc_neurons))
    n_e_sqrt = int(np.sqrt(number_exc_neurons))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = np.zeros((n_input, number_exc_neurons))
    connMatrix[connections[name].i, connections[name].j] = connections[name].w
    weight_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers, number_exc_neurons):
    assignments = np.zeros(number_exc_neurons)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * number_exc_neurons
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(number_exc_neurons):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments
