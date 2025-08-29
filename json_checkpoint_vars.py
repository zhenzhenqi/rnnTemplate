# json_checkpoint_vars.py

import tensorflow as tf
import numpy as np
import json
import os
import pickle

def dump_checkpoints(model, save_checkpoints_path, save_model_path, model_name):
    """
    Dumps the model weights to a JSON manifest and binary files
    compatible with ml5.js and tensorflow.js.
    """
    print(f"Exporting model '{model_name}' to {save_model_path} for use with ml5.js")

    # Create the target directory for the JavaScript model
    output_dir = os.path.join(save_model_path, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Create the manifest.json file and save weights ---
    manifest = {}
    all_vars_manifest = []
    
    # Iterate through all learned variables in the trained model.
    for var in model.variables:
        # Clean up the variable name for consistency.
        name_parts = var.name.split('/')
        name = '/'.join(name_parts[-2:]).split(':')[0]

        # Get the variable's value as a NumPy array
        data = var.numpy()
        
        # Create an entry for the manifest
        var_entry = {
            'filename': f'{name.replace("/", "_")}.bin',
            'name': name,
            'shape': data.shape
        }
        all_vars_manifest.append(var_entry)

        # Save the variable's data to a binary file
        data.astype(np.float32).tofile(os.path.join(output_dir, var_entry['filename']))

    manifest['vars'] = all_vars_manifest

    # Save the final manifest.json file
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)

    # --- 2. Copy vocabulary files for JS ---
    vocab_file = os.path.join(save_checkpoints_path, model_name, 'chars_vocab.pkl')
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as f:
            chars, vocab = pickle.load(f)
        
        with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
            json.dump(vocab, f)
        
        with open(os.path.join(output_dir, 'chars.json'), 'w') as f:
            json.dump(chars, f)

    print(f"Model successfully exported to {output_dir}")