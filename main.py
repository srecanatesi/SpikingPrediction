# I want something with event driven channels and continuous channels both in input and output
# The event driven channels should propagate as a CNF while the continuous ones shouldn't but could be decoded by the CNF
# This should be a mesh between spatiotemporalcnf and torchcde

#%% Write the data loader with behavioral data for neural recordings as well and should have two possible forms
# - continuous and discrete
# - discrete variables should have forward or symmetric smoothing
# - continuous variables should have multiple interpolation schemes

#%% Write a couple of toy datasets
# - maybe a spiking SLDS dataset (pass the state switch as a discrete action)
# - maybe a spiking HMM dataset
# - a sequence of HMM states
# - a grid world with HMM states (or Hawkes processes)
# - maybe a balanced RNN with two dimensional drive
# - Hawkes process
# - maybe lorentz attractor
# - two dimensional Ornstein Uhlembeck process
# - textual prediction (each word an HMM state and link them via text prediction)

#%% Write Neuroscience tasks
# - write two dimensional example of egocentric navigation with godot
# - write decision making 2AFC pseudo task
# - write center-out reaching task
# - consider genralize to "Cognitive Mapping and Planning for Visual Navigation"

#%% Models ideas
# - two output categories: probabilistic output for spikes (requires cnf) or continuous interpolation
# - two initial value categories: continuous integration (requires cde) versus autonoumous systems
# - write prediction model and show that it performs better in generalization than others

#%% Actual models
# - write (predictive) autoencoder that can actually generalize

#%% Write manifold analysis tools of ODE continuous solution


#%% Scientific results
# - predictive coding does better at generalizing spatially and temporally in neural data extracting dynamics
# - analyze differences between predictive and non-predictive models of neural activity
# - analyze predictive representations in grid world to see how actions build and navigate latent space
# - analyze predictive representations of stereotypical sequences
# - visual temporal prediction model

# - navigation + RL with manifold modified by the rewards
# - modeling 2ACF task and analyze information flow
# - perform cross-recordings (opto to neuropixels) prediction
# - link stochastic ODE models to general Feynman expansion theory
# - use neural actvity of micron dataset and check if adding connectivity helps
# - analyze navigation problem with muscolar movement to understand how all the variables are tensored in the representation
# - create neurons which are RL agents and perform some kind of operations, for example minimize activity and maximize influence

## Project 1 ("Prediction algorithms extract latent dynamics of neural data")
# verify spatial prediction of multiple models (predictive vs non-predictive)
# - datasets:
#       - Hawkes
#       - SLDS
#       - MC_maze
#       - Allen movie
# - models:
#       - Poisson regression
#       - RNN nonautonomous (predictive and not)
#       - CDE (predictive and not)
#       - STPP CNF (predictive and not)
#       - CDE encoder (IVP) CNF decoder
# - metrics:
#       - co-bps (predictive lag-dependent)
#       - MSE (predictive lag-dependent)







def get_data(dataset_name='mc_maze', bin_width=20):
    # dataset_name = 'mc_maze_small'
    # datapath = '000140/sub-Jenkins/'
    dataset_name = 'mc_maze'
    datapath = '000128/sub-Jenkins/'
    dataset = NWBDataset(datapath)
    # dataset = NWBDataset(datapath, skip_fields=['joint_ang', 'joint_vel', 'muscle_len', 'muscle_vel'])
    # bin_width = 5
    dataset.resample(bin_width)

    train_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split='train', save_file=False)
    train_spikes_heldin = train_dict['train_spikes_heldin']
    train_spikes_heldout = train_dict['train_spikes_heldout']
    timepoints = np.arange(train_spikes_heldin.shape[1])*dataset.bin_width /1000.
    eval_dict = make_eval_input_tensors(dataset, dataset_name=dataset_name, trial_split='val', save_file=False)
    eval_spikes_heldin = eval_dict['eval_spikes_heldin']
    eval_spikes_heldout = eval_dict['eval_spikes_heldout']
    input_dict = {'train_spikes_heldin': train_spikes_heldin,
                  'train_spikes_heldout': train_spikes_heldout,
                  'eval_spikes_heldin': eval_spikes_heldin,
                  'eval_spikes_heldout': eval_spikes_heldout,
                  'timepoints': timepoints}
    return input_dict, dataset

def smooth_spikes(data, t_scale=15, bin_width=5, normalize=False, forward=True):
    kern_sd_ms = t_scale
    kern_sd = int(round(kern_sd_ms / bin_width))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    if forward:
        window[:len(window)//2] = 0
    if normalize:
        window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    data = np.apply_along_axis(filt, 1, data)
    return data


def cast(tensor, device=device):
    if device=='cpu':
        if torch.is_tensor(tensor):
            return tensor.clone().detach().cpu().numpy().astype(np.double)
        else:
            return np.array(tensor).astype(np.double)
    if torch.is_tensor(tensor):
        if tensor.device.type == device.type:
            return tensor
        else:
            return tensor.float().to(device)
    else:
        return torch.tensor(tensor).float().to(device)
    print('Tensor cannnot be cast on device')
    return 0


#%%
input_dict, dataset = get_data()
ker_forward = False
Delta_forward = 0
N_hidden = 128
N_tangent = 64
train_X, train_Y, timepoints = input_dict['train_spikes_heldin'], input_dict['train_spikes_heldout'], input_dict['timepoints']
eval_X, eval_Y, timepoints = input_dict['eval_spikes_heldin'], input_dict['eval_spikes_heldout'], input_dict['timepoints']
Ny = eval_Y.shape[2]
Nx = eval_X.shape[2]
Nt = eval_Y.shape[1]
Ntrials = eval_X.shape[0]
