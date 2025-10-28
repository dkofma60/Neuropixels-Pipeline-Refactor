from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
import concurrent.futures
import torch
import json
import time
import gc
import psutil
import logging



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")



def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")



class NeuronDataset(Dataset):
    def __init__(self, spike_times, image_start_times, image_end_times, total_bins, bins_per_image):
        self.args_list = [(times, image_start_times, image_end_times, total_bins, bins_per_image) for times in spike_times.values()]

    def __len__(self):
        return len(self.args_list)

    def __getitem__(self, idx):
        return process_neuron_wrapper(self.args_list[idx])
        



def create_directory_and_manifest(directory_name='output'):
    output_dir = os.path.join(os.getcwd(), directory_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        print("Using existing manifest.json file.")
    else:
        print("Creating a new manifest.json file.")
    
    return output_dir, manifest_path



def create_cache_get_session_table(manifest_path):
    cache = EcephysProjectCache(manifest=manifest_path)
    session_table = cache.get_session_table()
    return cache, session_table



def pick_session_and_pull_data(cache, session_number, timeout=300):
    try:
        session = cache.get_session_data(session_number,
                                         isi_violations_maximum=np.inf,
                                         amplitude_cutoff_maximum=np.inf,
                                         presence_ratio_minimum=-np.inf,
                                         timeout=timeout)
        spike_times = session.spike_times
        stimulus_table = session.get_stimulus_table("natural_scenes")
        
        # Display objects in session
        print("Session objects")
        print([attr_or_method for attr_or_method in dir(session) if attr_or_method[0] != '_'])
        
        return session, spike_times, stimulus_table
    
    except TimeoutError:
        print(f"Timeout occurred while fetching session data for session {session_number}")
        return None, None, None



def filter_valid_spike_times(spike_times, session):
    # Access the invalid_times DataFrame
    invalid_times = session.invalid_times

    # Function to check if a spike time is valid
    def is_valid_time(spike_times, invalid_intervals):
        invalid = np.zeros_like(spike_times, dtype=bool)
        for _, row in invalid_intervals.iterrows():
            start, end = row['start_time'], row['stop_time']
            invalid |= (spike_times >= start) & (spike_times <= end)
        return ~invalid
    
    # Filter the valid spike times
    valid_spike_times = {}
    with tqdm(total=len(spike_times), desc='Filtering valid spike times') as pbar:
        for neuron, times in spike_times.items():
            valid_mask = is_valid_time(times, session.invalid_times)
            valid_spike_times[neuron] = times[valid_mask]
            pbar.update(1)

    return valid_spike_times



def get_stimulus_table(session, stimulus_type="natural_scenes"):
    return session.get_stimulus_table(stimulus_type)



def calculate_bins(stimulus_table, timesteps_per_frame=1):
    image_start_times = torch.tensor(stimulus_table.start_time.values)
    image_end_times = torch.tensor(stimulus_table.stop_time.values)
    bins_per_image = timesteps_per_frame
    total_bins = bins_per_image * len(image_start_times)
    return image_start_times, image_end_times, total_bins, bins_per_image



def process_neuron(times, image_start_times, image_end_times, total_bins, bins_per_image, device):
    start_bin = 0
    neuron_spike_bins = torch.zeros(total_bins, dtype=torch.int32, device=device)
    times_tensor = torch.as_tensor(times, device=device, dtype=torch.float32)
    for image_idx, (start_time, end_time) in enumerate(zip(image_start_times, image_end_times)):
        binned_spike_times = torch.histc(
            times_tensor,
            bins=bins_per_image,
            min=float(start_time),
            max=float(end_time),
        )
        end_bin = start_bin + bins_per_image
        if len(binned_spike_times) == len(neuron_spike_bins[start_bin:end_bin]):
            neuron_spike_bins[start_bin:end_bin] = binned_spike_times
        start_bin = end_bin
    return neuron_spike_bins



def process_neuron_wrapper(args):
    return process_neuron(*args, device=device)



def process_all_neurons(spike_times, image_start_times, image_end_times, total_bins, bins_per_image, batch_size=1000):
    image_start_times = image_start_times.to(device)
    image_end_times = image_end_times.to(device)
    args_list = [(times, image_start_times, image_end_times, total_bins, bins_per_image) for times in spike_times.values()]
    spike_matrix = []
    
    # Process neurons in smaller batches to prevent memory overload
    for i in range(0, len(args_list), batch_size):
        batch = args_list[i:i+batch_size]
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                batch_result = list(tqdm(executor.map(process_neuron_wrapper, batch), total=len(batch), desc=f'Processing neurons batch {i//batch_size + 1}'))
            batch_result_cpu = [t.cpu() for t in batch_result]
            spike_matrix.extend(batch_result_cpu)
            
            print_memory_usage()
            print_gpu_memory()
            del batch_result
            torch.cuda.empty_cache() # Clear GPU cache
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            break
        
        gc.collect()
        
    print(f'Spike matrix size -> {np.shape(spike_matrix)}')
    return spike_matrix



def create_and_prepare_spike_dataframe(spike_matrix, spike_times, stimulus_table, timesteps_per_frame):
    spike_matrix = torch.stack(spike_matrix, dim=0)
    
    # Create the DataFrame
    spike_df = pd.DataFrame(spike_matrix.cpu().numpy(), index=spike_times.keys())
    
    # Transpose the DataFrame
    spike_df = spike_df.T
    
    # Add and populate the frame column
    spike_df["frame"] = np.repeat(np.asarray(stimulus_table["frame"]), timesteps_per_frame)

    return spike_df



def save_and_count_spike_dataframe(spike_df, session_number, output_dir, timesteps_per_frame):
    nan_count = spike_df.isna().sum().sum()
    print(f"Number of NaN values in the DataFrame: {nan_count}")
    
    file_name = f'spike_trains_with_stimulus_session_{session_number}_{timesteps_per_frame}.pkl'
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(spike_df, f)
        
    return file_name



def normalize_firing_rates(df):
    df_copy = df.drop('frame', axis=1)
    normalized_firing_rates = (df_copy - df_copy.mean()) / df_copy.std()
    normalized_firing_rates.insert(0, 'frame', df['frame'])
    return normalized_firing_rates



def filter_and_save_neurons(normalized_firing_rates, highest_value=100, lowest_value=0, session_number=None, output_dir=None): #TODO: Don't use mask??
    # Filter neurons
    selected_neurons_mask = (normalized_firing_rates > highest_value).any() | (normalized_firing_rates < lowest_value).any()
    filtered_normalized_firing_rates = normalized_firing_rates.loc[:, selected_neurons_mask]
    
    # Check for NaN values
    nan_present = filtered_normalized_firing_rates.isna().any().any()
    print(f"There {'are' if nan_present else 'are no'} NaN values in the DataFrame")
    
    # Save the filtered data, if session_number and output_dir are provided
    if session_number is not None and output_dir is not None:
        file_name = f'filtered_normalized_pickle_{session_number}.pkl'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(filtered_normalized_firing_rates, f)
        print(f"Filtered normalized firing rates saved to {file_path}")
        
    return filtered_normalized_firing_rates



def get_session_ids(cache):
    session_table = cache.get_session_table()
    return session_table.index.tolist()



def master_function(session_number, output_dir="/proj/STOR/pipiras/Neuropixel/Neuropixels-Pipeline-Refactor/src/output", timesteps_per_frame=10, timeout=300):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    start_time = time.time()

    print("Updated version 3!")
    print("Initializing workflow...")
    
    # Paths for input files
    spike_trains_file_path = os.path.join(output_dir, f'spike_trains_with_stimulus_session_{session_number}_{timesteps_per_frame}.pkl')

    # Check if the files already exist
    if os.path.exists(spike_trains_file_path):
        print("Loading existing raw datasets...")
        
        # Load the spike trains 
        with open(spike_trains_file_path, 'rb') as f:
            spike_df_raw = pickle.load(f)
        print(f"Loaded spike trains dataset: {type(spike_df_raw)}")

        print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
        return spike_df_raw

    else:
        try:
            # Create data if non-existent
            print("Creating directory and manifest path...")
            output_dir, manifest_path = create_directory_and_manifest(directory_name=output_dir)

            print("Initializing EcephysProjectCache and session table...")
            cache, session_table = create_cache_get_session_table(manifest_path)

            print("Fetching session data and spike times...")
            session, spike_times, stimulus_table = pick_session_and_pull_data(cache, session_number, timeout)

            print("Filtering valid spike times...")
            valid_spike_times = filter_valid_spike_times(spike_times, session)
            gc.collect()
            torch.cuda.empty_cache()

            print("Calculating bins and processing neurons...")
            image_start_times, image_end_times, total_bins, bins_per_image = calculate_bins(stimulus_table, timesteps_per_frame)
            spike_matrix = process_all_neurons(valid_spike_times, image_start_times, image_end_times, total_bins, bins_per_image)
            gc.collect()
            torch.cuda.empty_cache()

            print("Preparing spike DataFrame...")
            spike_df = create_and_prepare_spike_dataframe(spike_matrix, valid_spike_times, stimulus_table, timesteps_per_frame)
            gc.collect()
            torch.cuda.empty_cache()

            print("Save raw rate dataframes as a pickle file.")
            with open(spike_trains_file_path, 'wb') as f:
                pickle.dump(spike_df, f)

            print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error: An unexpected error occurred - {str(e)}")
            return None

    return spike_df