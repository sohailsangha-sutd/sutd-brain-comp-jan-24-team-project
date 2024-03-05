from math import ceil
import brian2 as b2
import numpy as np
import matplotlib.cm as cmap

from . import data

def plot_2d_input_weights(connections, n_input, number_exc_neurons, fig_num, wmax_ee):
    name = 'XeAe'
    weights = data.get_2d_input_weights(connections, n_input, number_exc_neurons)
    fig = b2.figure(fig_num, figsize = (8, 8))
    im2 = b2.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b2.colorbar(im2)
    b2.title('weights of connection' + name)
    fig.show()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return im2, fig


def update_2d_input_weights(connections, n_input, number_exc_neurons, im, fig):
    weights = data.get_2d_input_weights(connections, n_input, number_exc_neurons)
    im.set_array(weights)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return im, fig


def get_current_performance(performance, current_example_num, update_interval, input_numbers, outputNumbers):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def get_aggegrate_performance(performance_aggregate, current_example_num, update_interval, input_numbers, outputNumbers):
    current_evaluation = int(current_example_num/update_interval)
    start_num = 0
    end_num = current_example_num
    if current_example_num > update_interval:
        end_num = current_example_num - update_interval
    
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance_aggregate[current_evaluation] = correct / float(current_example_num) * 100
    return performance_aggregate

def plot_performance(fig_num, num_examples, update_interval):
    num_evaluations = int(ceil(num_examples/update_interval))
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    performance_aggregate = np.zeros(num_evaluations)
    fig = b2.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(211)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b2.ylim(ymax = 100)
    b2.title('Classification Performance over Update Intervals')

    ax2 = fig.add_subplot(212)
    im3, = ax2.plot(time_steps, performance) #my_cmap
    b2.ylim(ymax = 100)
    b2.title('Aggregate Classification Performance')

    fig.show()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return im2, im3, performance, performance_aggregate, fig_num, fig

def update_performance_plot(im, im3, performance, performance_aggregate, current_example_num, fig, update_interval, input_numbers, outputNumbers):
    performance = get_current_performance(performance, current_example_num, update_interval, input_numbers, outputNumbers)
    im.set_ydata(performance)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return im, im3, performance, performance_aggregate

def update_aggregate_performance_plot(im, im3, performance, performance_aggregate, current_example_num, fig, update_interval, input_numbers, outputNumbers):
    performance_aggregate = get_aggegrate_performance(performance_aggregate, current_example_num, update_interval, input_numbers, outputNumbers)
    im3.set_ydata(performance_aggregate)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return im, im3, performance, performance_aggregate