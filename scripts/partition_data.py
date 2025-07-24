import h5py
import numpy as np
import time

base_path = '/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/data_files/DeepSTARR_data.h5'
fractions = [0.1, 0.25, 0.5, 0.75]
seeds = [0, 42, 123]

with h5py.File(base_path, 'r') as f:
    dataset_names = list(f.keys())
    print("Datasets and shapes:")
    for name in dataset_names:
        print(f"  {name}: {f[name].shape}")

    for fraction in fractions:
        for seed in seeds:
            start_time = time.time()
            out_path = f'{base_path}_subset_{fraction}_{seed}.h5'
            print(f"\nProcessing fraction={fraction}, seed={seed} -> {out_path}")
            with h5py.File(out_path, 'w') as out_f:
                for name in dataset_names:
                    num_samples = f[name].shape[0]
                    subset_size = int(num_samples * fraction)
                    rng = np.random.default_rng(seed)
                    indices = rng.choice(num_samples, size=subset_size, replace=False)
                    indices.sort()
                    print(f"  Subsetting {name}: {subset_size}/{num_samples} samples")
                    # Time the data extraction
                    t0 = time.time()
                    data = f[name][indices]
                    t1 = time.time()
                    print(f"    Data extraction took {t1-t0:.2f} seconds")
                    out_f.create_dataset(name, data=data)
            elapsed = time.time() - start_time
            print(f"Saved: {out_path} (total time: {elapsed:.2f} seconds)")