import os
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

def load_real_epilepsy_dataset(data_dir="sim/data"):
    """
    Loads the Epilepsy dataset from Dimension-specific ARFF files.
    Combines TRAIN and TEST sets for Cross-Validation.
    Returns:
        X_magnitude (list of np.arrays): The 1D magnitude signal of each sequence.
        y (np.array): Encoded labels.
    """
    print(f"Loading Epilepsy dataset from {data_dir}...")

    # Helper to load specific dimensions
    def load_arff_dimension(dim_name, split):
        filename = f"Epilepsy{dim_name}_{split}.arff"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find {filepath}")
            
        data, meta = arff.loadarff(filepath)
        # Convert structured array to regular numpy array, excluding the class label (last col)
        # The ARFF format usually has 'att1', 'att2', ... 'target'
        # We assume the last column is the label
        df_data = np.array(data.tolist())
        
        # Separate features and labels
        X_dim = df_data[:, :-1].astype(float)
        y_raw = df_data[:, -1]
        
        # Decode byte strings if necessary
        if isinstance(y_raw[0], (bytes, np.bytes_)):
            y_raw = [label.decode('utf-8') for label in y_raw]
            
        return X_dim, np.array(y_raw)

    try:
        # Load all 3 dimensions (Tri-axial accelerometer: X, Y, Z)
        # We load both TRAIN and TEST to merge them for 5-Fold CV
        X_train_d1, y_train = load_arff_dimension("Dimension1", "TRAIN")
        X_train_d2, _       = load_arff_dimension("Dimension2", "TRAIN")
        X_train_d3, _       = load_arff_dimension("Dimension3", "TRAIN")
        
        X_test_d1, y_test   = load_arff_dimension("Dimension1", "TEST")
        X_test_d2, _        = load_arff_dimension("Dimension2", "TEST")
        X_test_d3, _        = load_arff_dimension("Dimension3", "TEST")
        
        # Stack dimensions: Shape becomes (N_samples, Seq_Len, 3)
        X_train_3d = np.stack([X_train_d1, X_train_d2, X_train_d3], axis=2)
        X_test_3d  = np.stack([X_test_d1, X_test_d2, X_test_d3], axis=2)
        
        # Merge Train and Test
        X_all_3d = np.concatenate([X_train_3d, X_test_3d], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        # Calculate Magnitude Vector: sqrt(x^2 + y^2 + z^2)
        # This mimics the "Acceleration" signal used in the paper (Figure 1)
        # Result is Shape (N_samples, Seq_Len)
        X_magnitude = np.linalg.norm(X_all_3d, axis=2)
        
        # Encode Labels (Seizure, Walking, etc. -> 0, 1, 2, 3)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_all)
        
        print(f"Dataset Loaded. Total Samples: {len(y_encoded)}")
        print(f"Classes Found: {le.classes_}")
        print(f"Sequence Length: {X_magnitude.shape[1]}")
        
        return list(X_magnitude), y_encoded

    except Exception as e:
        print(f"Error loading ARFF files: {e}")
        print("Ensure files like 'EpilepsyDimension1_TRAIN.arff' are in the data/ directory.")
        raise e

if __name__ == "__main__":
    # Simple test
    X, y = load_real_epilepsy_dataset()
    print("First sequence shape:", X[0].shape)
    print("First label:", y[0])