#import numpy as np
#import glob

#def merge_npz_files(file_list, output_file):
    # Initialize a dictionary to hold all arrays
#    combined_arrays = {}
    
#    for file in file_list:
#        with np.load(file) as data:
#            # Iterate through each array in the file
#            for key in data:
#                # Append or update arrays in the combined dictionary
#                if key in combined_arrays:
#                    combined_arrays[key] = np.concatenate((combined_arrays[key], data[key]), axis=0)
#                else:
#                    combined_arrays[key] = data[key]
    
    # Save the combined arrays to a new .npz file:
#    with np.savez_compressed(output_file, **{key: np.concatenate(combined_arrays[key], axis=0) for key in combined_arrays}):
#        pass

# List of .npz files you want to merge
#file_list = glob.glob('graphs/*?.npz')

# Output file name
#output_file = 'merged_file.npz'

# Call the function to merge files
#merge_npz_files(file_list, output_file)


import numpy as np
import glob
import os

def merge_npz_files_with_temp(file_list, output_file):
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    temp_files = {}

    for file in file_list:
        with np.load(file) as data:
            for key in data:
                if key not in temp_files:
                    temp_files[key] = []
                temp_file = os.path.join(temp_dir, f"{key}.npz")
                
                if os.path.exists(temp_file):
                    # Load existing temporary data
                    with np.load(temp_file) as temp_data:
                        merged_array = np.concatenate((temp_data[key], data[key]), axis=0)
                else:
                    merged_array = data[key]
                
                # Save updated data to the temporary file
                np.savez_compressed(temp_file, **{key: merged_array})

    # Load final merged arrays from temp files and save them to the output file
    final_data = {key: np.load(os.path.join(temp_dir, f"{key}.npz"))[key] for key in temp_files}
    np.savez_compressed(output_file, **final_data)

    # Clean up temporary files
    for temp_file in temp_files.values():
        os.remove(temp_file)

# List of .npz files
file_list = glob.glob('graphs/*?.npz')
output_file = 'merged_file.npz'

# Merge files using temporary files
merge_npz_files_with_temp(file_list, output_file)