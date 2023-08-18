import torch

def design_input(shape, path, stimulus_times, interval):
    
    inp = torch.zeros(shape[0],2)
    direction = path%4
    
    stimulus_times = torch.tensor(stimulus_times)
    stimulus_times = torch.div(stimulus_times, interval, rounding_mode='floor')
    
    if direction == 0: #left-up
        inp[stimulus_times[1]:,0] = -1
        inp[stimulus_times[2]:,1] = 1
    elif direction == 1: #left-down
        inp[stimulus_times[1]:,0] = -1
        inp[stimulus_times[2]:,1] = -1
    elif direction == 2: #right-up
        inp[stimulus_times[1]:,0] = 1
        inp[stimulus_times[2]:,1] = 1
    elif direction == 3: #right-down
        inp[stimulus_times[1]:,0] = 1
        inp[stimulus_times[2]:,1] = -1
    else:
        NotImplemented
        
    return inp 