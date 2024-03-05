'''
Created on 15.12.2014

@author: Peter U. Diehl
'''


import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
from brian2 import *
import os
import brian2 as b2
from brian2tools import *
prefs.codegen.target = 'cython'


from functions import data
from functions import plots



# specify the location of the MNIST data
MNIST_data_path = '..\\Brian2STDPMNIST\\mnist\\'
data_path = '.\\' # TODO: This should be a parameter

#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = data.get_labeled_data(MNIST_data_path + 'training', MNIST_data_path=MNIST_data_path)
end = time.time()
print('time needed to load training set:', end - start)

start = time.time()
testing = data.get_labeled_data(MNIST_data_path + 'testing', bTrain = False, MNIST_data_path=MNIST_data_path)
end = time.time()
print('time needed to load test set:', end - start)


#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = False # Change this to False to retrain the network

np.random.seed(0)

if test_mode:
    weight_path = data_path + 'weights\\'
    num_examples = 10 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = False
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random\\'
    num_examples = 1 * 1
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 30:
        record_spikes = True
    else:
        record_spikes = False
    ee_STDP_on = True

# encoding_type = 'poisson'
# encoding_type = 'ttfs'
# encoding_type = 'phase'
encoding_type = 'burst'

intensity_scalar = 1
intensity_time_offset = 0 * b2.second
intensity_decay_rate = 6 * b2.ms

ending = ''
n_input = 784
n_e = 100
n_i = n_e
single_example_time =   0.035 * b2.second #
resting_time = 0.015 * b2.second
if encoding_type == 'poisson':
    single_example_time = 0.100 * b2.second
    resting_time = 0.080 * b2.second
if encoding_type == 'ttfs':
    single_example_time = 0.020 * b2.second
    resting_time = 0.020 * b2.second
if encoding_type == 'phase':    
    single_example_time = 0.100 * b2.second
    resting_time = 0.030 * b2.second
if encoding_type == 'burst':
    single_example_time = 0.020 * b2.second
    resting_time = 0.020 * b2.second

runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000


# update_interval = 30

v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV # going to focus on this value as the threshold for LIF
v_thresh_i = -40. * b2.mV
# if encoding_type == 'poisson':
#     v_thresh_e = -60. * b2.mV
#     v_thresh_i = 0.6 * b2.mV
# elif encoding_type == 'ttfs':
#     v_thresh_e = 0.5 * b2.mV
#     v_thresh_i = 0.5 * b2.mV
# elif encoding_type == 'phase':
#     v_thresh_e = 0.8 * b2.mV
#     v_thresh_i = 0.8 * b2.mV
# elif encoding_type == 'burst':
#     v_thresh_e = 0.4 * b2.mV
#     v_thresh_i = 0.4 * b2.mV

refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b2.ms,10*b2.ms)
delay['ei_input'] = (0*b2.ms,5*b2.ms)
input_intensity = 1.

start_input_intensity = 1.
input_intensity_increment = 1.
if encoding_type == 'phase':
    start_input_intensity = 0.1
    input_intensity_increment = 0.05
elif encoding_type == 'burst':
    start_input_intensity = 1
    input_intensity_increment = 0.1
# Synaptic Conductance Model Parameters

tc_pre_ee = 30*b2.ms                                # time constants for STDP, leave post 2 as is, it is an enhancement
tc_post_1_ee = 30*b2.ms
if encoding_type == 'ttfs':
    tc_pre_ee = 10*b2.ms
    tc_post_1_ee = 10*b2.ms


tc_post_2_ee = 40*b2.ms

nu_ee_pre =  0.0001                                 # learning rate in STDP Model
nu_ee_post = 0.01                                   # learning rate
if encoding_type == 'poisson':
    nu_ee_pre =  0.002
    nu_ee_post = 0.02    
elif encoding_type == 'ttfs':
    nu_ee_pre =  0.0004
    nu_ee_post = 0.09  
elif encoding_type == 'phase':
    nu_ee_pre =  0.0006
    nu_ee_post = 0.004  
elif encoding_type == 'burst':
    nu_ee_pre =  0.00001
    nu_ee_post = 0.0007  

wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

scr_e = ''
tc_theta = 1e7 * b2.ms
theta_plus_e = 0.05 * b2.mV

def set_source_e(test_mode):
    global scr_e, tc_theta, theta_plus_e

    if test_mode:
        scr_e = 'v = v_reset_e; timer = 0*ms'
    else:
        tc_theta = 1e7 * b2.ms

        theta_plus_e = 0.05 * b2.mV                     # Firing Threshold Adaptation Constant (ThetaPlus)
        if encoding_type == 'poisson':
            theta_plus_e = 0.008 * b2.mV
        elif encoding_type == 'ttfs':
            theta_plus_e = 0.04 * b2.mV
        elif encoding_type == 'phase':
            theta_plus_e = 0.008 * b2.mV
        elif encoding_type == 'burst':
            theta_plus_e = 0.005 * b2.mV
        
        scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

set_source_e(test_mode)

offset = 20.0*b2.mV
v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'

# Time constant for LiF possibly hard coded in the equation below (Original author uses 100ms, but mentions biologically this will be 10ms)
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

def set_neuron_e_test_mode(test_mode):
    global neuron_eqs_e
    if test_mode:
        neuron_eqs_e += '\n  theta      :volt'
    else:
        neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'

set_neuron_e_test_mode(test_mode)

neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

b2.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

neuron_groups['e'] = b2.NeuronGroup(n_e*len(population_names), neuron_eqs_e, 
                                    threshold= v_thresh_e_str, refractory= refrac_e, 
                                    reset= scr_e, method='euler')

neuron_groups['i'] = b2.NeuronGroup(n_i*len(population_names), neuron_eqs_i, 
                                    threshold= v_thresh_i_str, refractory= refrac_i, 
                                    reset= v_reset_i_str, method='euler')


#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
for subgroup_n, name in enumerate(population_names):
    print('create neuron group', name)

    neuron_groups[name+'e'] = neuron_groups['e'][subgroup_n*n_e:(subgroup_n+1)*n_e]
    neuron_groups[name+'i'] = neuron_groups['i'][subgroup_n*n_i:(subgroup_n+1)*n_e]

    neuron_groups[name+'e'].v = v_rest_e - 40. * b2.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b2.mV
    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy') * b2.volt
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b2.mV

    print('create recurrent connections')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = data.get_matrix_from_file(weight_path + '../random/' + connName + ending + '.npy',
                                                 ending, n_input, n_e, n_i)
        model = 'w : 1'
        pre = 'g%s_post += w' % conn_type[0]
        post = ''
        if ee_STDP_on:
            if 'ee' in recurrent_conn_names:
                model += eqs_stdp_ee
                pre += '; ' + eqs_stdp_pre_ee
                post = eqs_stdp_post_ee
        connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        connections[connName].connect(True) # all-to-all connection
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

    print('create monitors for', name)
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(neuron_groups[name+'e'])
    rate_monitors[name+'i'] = b2.PopulationRateMonitor(neuron_groups[name+'i'])
    spike_counters[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])

    if record_spikes:
        spike_monitors[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b2.SpikeMonitor(neuron_groups[name+'i'])


#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------


pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    if encoding_type == 'poisson':
        input_groups[name+'e'] = b2.PoissonGroup(n_input, 0*Hz)
    else:
        input_groups[name+'e'] = b2.SpikeGeneratorGroup(n_input, [], [] * b2.second)
    
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(input_groups[name+'e'])

    if record_spikes:
        spike_monitors[name+'e'] = b2.SpikeMonitor(input_groups[name+'e'])

for name in input_connection_names:
    print('create connections between', name[0], 'and', name[1])
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = data.get_matrix_from_file(weight_path + connName + ending + '.npy',
                                                 ending, n_input, n_e, n_i)
        model = 'w : 1'
        if encoding_type == 'ttfs':
            pre = 'g%s_post += intensity_scalar * exp(-(t - intensity_time_offset)/intensity_decay_rate) * w' % connType[0]
        elif encoding_type == 'phase':
            
            pre = 'g%s_post += intensity_scalar * (2 ** -(1 + (ceil((t - intensity_time_offset)/(1*ms)) - 1) %% 8)) * w' % connType[0]
        else:
            pre = 'g%s_post += intensity_scalar * w' % connType[0]
        post = ''
        if ee_STDP_on:
            print('create STDP for connection', name[0]+'e'+name[1]+'e')
            model += eqs_stdp_ee
            pre += '; ' + eqs_stdp_pre_ee
            post = eqs_stdp_post_ee

        connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        minDelay = delay[connType][0]
        maxDelay = delay[connType][1]
        deltaDelay = maxDelay - minDelay
        # TODO: test this
        connections[connName].connect(True) # all-to-all connection
        connections[connName].delay = 'minDelay + rand() * deltaDelay'
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

# input_indices = np.concatenate((np.linspace(0, n_input-1, n_input), np.ones(100)*500))
    # input_times = np.concatenate(((np.linspace(0, n_input-1, n_input) / n_input), (np.linspace(0, 99, 100) / n_input)))
    # print(len(input_indices), len(input_times))

    # input_groups[name+'e'] = SpikeGeneratorGroup(n_input, input_indices, input_times * single_example_time)



#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

net = Network()
for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
        spike_monitors, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
outputAggregateNumbers = np.zeros((num_examples, 10))


if not test_mode:
    input_weight_monitor, fig_weights = plots.plot_2d_input_weights(connections, n_input, n_e, fig_num, wmax_ee)
    fig_num += 1
if do_plot_performance:
    performance_monitor, aggregate_performane_monitor, performance, performance_aggregate, fig_num, fig_performance = plots.plot_performance(fig_num, num_examples, update_interval)

b2.show()

coding_ttfs_min_delay = 5
coding_ttfs_max_delay = 20
coding_ttfs_diff = coding_ttfs_max_delay - coding_ttfs_min_delay
coding_ttfs_spike_decay_rate = 15

coding_phase_start_time = 0*b2.ms
coding_phase_spike_delay = 1

coding_burst_min_time = 2
coding_burst_max_time = 10
coding_burst_N_max = 5

def get_phase_array(value):
    arr = np.zeros(8)
    for i in range(8):
        arr[7 - i] = value & 1
        value = value >> 1
    return arr

track_aggregate_performance = False

def set_input_example(example, input_intensity):
    global input_groups, intensity_scalar, intensity_time_offset

    if encoding_type == 'poisson':
        input_groups['Xe'].rates = example / 8. * input_intensity * Hz

    elif encoding_type == 'ttfs': # TODO: needs weight adjustment
        # print('net time: ', net.t)
        print('example: ', (coding_ttfs_max_delay - ((255 / 255.) * (coding_ttfs_diff))))
        print('example: ', (coding_ttfs_max_delay - ((0 / 255.) * (coding_ttfs_diff))))

        input_times = np.array([])
        input_indices = np.array([])
        for i in range(n_input):
            if example[i] == 0: continue
            input_t = -log(example[i] / 255) * coding_ttfs_spike_decay_rate
            if input_t > coding_ttfs_max_delay: continue
            input_times = np.append(input_times, input_t)
            input_indices = np.append(input_indices, i)

        input_times = net.t + (input_times * b2.ms)
        input_groups['Xe'].set_spikes(input_indices, input_times)

        intensity_scalar = input_intensity
        intensity_time_offset = net.t

    elif encoding_type == 'phase': # TODO: needs weight adjustment
        print('phase coding')
        input_times = np.array([])
        input_indices = np.array([])
        for i in range(n_input):
            phase_array = get_phase_array(int(example[i]))
            # print('phase array: ', phase_array, ' for example: ', example[i])
            for p in range(len(phase_array)):
                if phase_array[p] == 1:
                    input_times = np.append(input_times, (p * coding_phase_spike_delay))
                    input_indices = np.append(input_indices, i)

        input_times = net.t + (coding_phase_start_time + (input_times * b2.ms))
        all_input_times = np.array([])
        all_input_indices = np.array([])

        for i in range(12):
            all_input_times = np.append(all_input_times, i*8*b2.ms + input_times)
            all_input_indices = np.append(all_input_indices, input_indices)

        all_input_ts = all_input_times * b2.second

        # print(input_times)
        input_groups['Xe'].set_spikes(all_input_indices, all_input_ts)

        intensity_scalar = input_intensity
        intensity_time_offset = net.t

    elif encoding_type == 'burst':
        Ns_p = np.ceil(example / 255. * coding_burst_N_max * input_intensity)
        isi_p = np.ceil(coding_burst_max_time - (coding_burst_max_time - coding_burst_min_time) * (example / 255.))
        input_times = np.array([])
        input_indices = np.array([])
        for i in range(n_input):
            if Ns_p[i] == 0: continue
            input_t = np.arange(0, int(Ns_p[i])) * isi_p[i]    
            input_times = np.append(input_times, input_t)
            input_indices = np.append(input_indices, np.ones(len(input_t)) * i)

        input_times = net.t + (input_times * b2.ms)
        input_times = input_times + np.random.rand(len(input_times)) * 0.1 * b2.ms
        input_groups['Xe'].set_spikes(input_indices, input_times)

        # intensity_scalar = input_intensity
        
def set_input_resting():
    global input_groups

    if encoding_type == 'poisson':
        input_groups['Xe'].rates = 0 * Hz
    # elif encoding_type == 'ttfs':
        # input_groups['Xe'].set_spikes(np.arange(0, n_input), -1.0 * second)


set_input_resting()
net.run(0*second)



j = 0
time_sim_average = np.array([])
effective_latency = np.array([])

effective_latency_start_time = net.t

training_start_time = time.time()

test_aggregate_performance = False
test_aggregate_performance_till_example = 0

while j < (int(num_examples)):
    if test_mode:
        if use_testing_set:
            spike_rates = testing['x'][j%10000,:,:].reshape((n_input))
        else:
            spike_rates = training['x'][j%60000,:,:].reshape((n_input))
    else:
        if not test_aggregate_performance:
            data.normalize_weights(connections, weight, n_e)
        spike_rates = training['x'][j%60000,:,:].reshape((n_input))
    
    set_input_example(spike_rates, input_intensity)
    
    

    time_sim_start = time.time()
    net.run(single_example_time) #, report='text')
    time_sim_end = time.time()
    time_sim = time_sim_end - time_sim_start
    time_sim_average = np.append(time_sim_average, time_sim)

    if j % update_interval == 0 and j > 0:
        assignments = data.get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j], n_e)
    if j % weight_update_interval == 0 and not (test_mode or test_aggregate_performance):
        plots.update_2d_input_weights(connections, n_input, n_e, input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not (test_mode or test_aggregate_performance):
        data.save_connections(save_conns, connections, data_path, str(j) + '_' + encoding_type)
        data.save_theta(population_names, neuron_groups, data_path, str(j) + '_' + encoding_type)

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    print_end = '\r'
    if j % update_interval == 0 and np.sum(current_spike_count) > 5:
        print_end = '\n'
    print('run number: ', j+1, ' of ', int(num_examples), ' with intensity:', round(input_intensity, 1), 'and test', test_aggregate_performance, ' ' * 5, end=print_end)

    if np.sum(current_spike_count) < 5:
        input_intensity += input_intensity_increment
        set_input_resting()
        net.run(resting_time)
    else:
        effective_latency = np.append(effective_latency, net.t - effective_latency_start_time)

        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        if test_aggregate_performance:
            outputAggregateNumbers[j,:] = data.get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        else:
            outputNumbers[j,:] = data.get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])

        # if j % 100 == 0 and j > 0:
        #     print('runs done:', j, 'of', int(num_examples))
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                
                if test_aggregate_performance and j == test_aggregate_performance_till_example:
                    unused, unused, performance, performance_aggregate = plots.update_aggregate_performance_plot(performance_monitor, aggregate_performane_monitor, performance, performance_aggregate, j, fig_performance, update_interval, input_numbers, outputNumbers)
                    print('Aggregate classification performance', performance_aggregate[:int(j/float(update_interval))+1])
                    test_aggregate_performance = False
                    j += update_interval
                elif not test_aggregate_performance:
                    unused, unused, performance, performance_aggregate = plots.update_performance_plot(performance_monitor, aggregate_performane_monitor, performance, performance_aggregate, j, fig_performance, update_interval, input_numbers, outputNumbers)
                    print('Classification performance', performance[:int(j/float(update_interval))+1])
                    if j > update_interval and track_aggregate_performance:
                        test_aggregate_performance_till_example = j - update_interval
                        test_aggregate_performance = True
                        j = -1

                set_source_e(test_aggregate_performance)
                set_neuron_e_test_mode(test_aggregate_performance)
        
        set_input_resting()
        net.run(resting_time)
        input_intensity = start_input_intensity

        effective_latency_start_time = net.t
        j += 1


    # print('_'*30)


training_end_time = time.time()

#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print('save results')
if not test_mode:
    data.save_theta(population_names, neuron_groups, data_path, '_' + encoding_type)
if not test_mode:
    data.save_connections(save_conns, connections, data_path, '_' + encoding_type)

np.save(data_path + 'activity/resultPopVecs' + str(num_examples)  + '_' + encoding_type, result_monitor)
np.save(data_path + 'activity/inputNumbers' + str(num_examples) + '_' + encoding_type, input_numbers)
np.save(data_path + 'activity/simTime' + str(num_examples) + '_' + encoding_type, time_sim_average)
np.save(data_path + 'activity/effectiveLatency' + str(num_examples) + '_' + encoding_type, effective_latency)

# print(effective_latency)
# print(result_monitor)

print('average effective latency: ', np.average(effective_latency))
print('average simulation time: ', np.average(time_sim_average))

training_time_seconds = (training_end_time - training_start_time) * (60000 / num_examples)
print('Training Time: ', (training_end_time - training_start_time), ' seconds')
print('Estimated Training Time:', int(( training_time_seconds ) / (60*60)), 'hours',  int((training_time_seconds / 60) % (60)), 'mins', int(training_time_seconds % 60), 'seconds')

#------------------------------------------------------------------------------
# plot results
#------------------------------------------------------------------------------
if rate_monitors:
    b2.figure(fig_num)
    fig_num += 1
    ax1 = ''
    for i, name in enumerate(rate_monitors):
        if ax1 == '':
            ax1 = b2.subplot(len(rate_monitors), 1, 1+i)
        else:
            b2.subplot(len(rate_monitors), 1, 1+i, sharex=ax1)
        
        b2.plot(rate_monitors[name].t/b2.second, rate_monitors[name].rate, '.')
        b2.title('Rates of population ' + name)

if record_spikes:
    if spike_monitors:
        b2.figure(fig_num)
        fig_num += 1
        ax1 = ''
        for i, name in enumerate(spike_monitors):
            if ax1 == '':
                ax1 = b2.subplot(len(spike_monitors), 1, 1+i)
            else:
                b2.subplot(len(spike_monitors), 1, 1+i, sharex=ax1)
            
            b2.plot(spike_monitors[name].t/b2.ms, spike_monitors[name].i, '.')
            b2.title('Spikes of population ' + name)

    total_simulation_time = net.t
    print('Total simulation time: ', total_simulation_time)
    spike_mat = np.zeros((n_input, int(total_simulation_time*1000)))
    for i in range(spike_monitors['Xe'].i.shape[0]):
        spike_mat[spike_monitors['Xe'].i[i], int(spike_monitors['Xe'].t[i]*1000)] = 1

    if 'Xe' in spike_monitors:
        b2.figure(fig_num)
        fig_num += 1
        b2.plot(spike_monitors['Xe'].count[:])
        b2.title('Spike count of population Xe')
        # print('Spike Monitor Xe:', spike_monitors['Xe'].count[:].reshape((28,28)))

        fig = b2.figure(fig_num)
        b2.matshow(spike_mat, fignum=fig_num, aspect='auto')
        b2.title('Spikes of population Xe, Matshow')
        b2.xlabel('Input Number: ' + str(training['y'][0][0]))
        fig_num += 1

        b2.figure(fig_num)
        b2.matshow(spike_monitors['Xe'].count[:].reshape((28,28)), fignum=fig_num)
        b2.title('Spike count of population Xe, Matshow')
        b2.xlabel('Input Number: ' + str(training['y'][0][0]))
        fig_num += 1

    if spike_counters:
        b2.figure(fig_num)
        fig_num += 1
        b2.plot(spike_monitors['Ae'].count[:])
        b2.title('Spike count of population Ae')









plots.plot_2d_input_weights(connections, n_input, n_e, fig_num, wmax_ee)

plt.figure(fig_num)
fig_num += 1

subplot(3,1,1)

# The code seems to fail at the following step (NotImplementedError: Do not know how to plot object of type <class 'brian2.core.variables.VariableView'>)
brian_plot(connections['XeAe'].w)
subplot(3,1,2)

brian_plot(connections['AeAi'].w)

subplot(3,1,3)

brian_plot(connections['AiAe'].w)


plt.figure(fig_num)
fig_num += 1

subplot(3,1,1)

brian_plot(connections['XeAe'].delay)
subplot(3,1,2)

brian_plot(connections['AeAi'].delay)

subplot(3,1,3)

brian_plot(connections['AiAe'].delay)


b2.ioff()
b2.show()



