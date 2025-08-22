import os 
import numpy as np 
import pandas as pd
from scipy.io import arff

DATA_DIR = "../NATOPS-data/NATOPS"
OUTPUT_CSV = "../Results/phase1_output.csv"

# Load a single .arff file and return as pandas DataFrame
def load_arff_data(path):
    data, _ = arff.loadarff(path)
    return pd.DataFrame(data)

# Process all 24 feature files for either 'train' or 'test' set
def process_all_features(split, sid_offset=0):
    feature_dfs = []  # To collect each reshaped feature column
    sids = []         # Will store sample IDs (one per time step)
    labels = None     # Will store gesture class labels (one per time step)

    for i in range(1, 25):  # Loop through 24 features
        fname = f"NATOPSDimension{i}_{split.upper()}.arff"
        path = os.path.join(DATA_DIR, fname) 
        df = load_arff_data(path)

        # Separate time-series data and labels
        X = df.iloc[:, :-1].astype(float)  # All but last column = time step values
        y = df.iloc[:, -1].apply(lambda x: float(x.decode()) if isinstance(x, bytes) else float(x))  # Last column = gesture label

        # Reshape feature into long format: one value per time step per sample
        X_reshaped = X.to_numpy().reshape(-1, 1)
        feature_dfs.append(pd.DataFrame(X_reshaped, columns=[f"fea{i}"]))

        # On the first feature, build sid and label columns (same for all features)
        if i == 1:
            n_samples, n_timesteps = X.shape
            sids = np.repeat(np.arange(n_samples) + sid_offset, n_timesteps)
            labels = np.repeat(y.to_numpy(), n_timesteps)

    # Combine all 24 features into a single DataFrame
    full_features = pd.concat(feature_dfs, axis=1)
    full_features['sid'] = sids  # Add sample ID column
    full_features['class'] = labels  # Add gesture label column
    full_features['isTest'] = 1 if split == 'test' else 0  # Add test/train indicator

    return full_features

# Process both train and test sets
def main():
    # Process each split independently
    df_train = process_all_features('train', sid_offset=1)    # Train data, sample IDs 1–180
    df_test = process_all_features('test', sid_offset=181)    # Test data, sample IDs 181+

    # Combine both into one unified DataFrame
    final_df = pd.concat([df_train, df_test], ignore_index=True)

    cols = ['isTest'] + [col for col in final_df.columns if col != 'isTest']
    final_df = final_df[cols]

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Output written to {OUTPUT_CSV}")
    print(f"Final shape: {final_df.shape}")
    print(final_df.head())

if __name__ == "__main__":
    main()
