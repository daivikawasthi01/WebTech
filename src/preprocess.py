import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_dataset(input_csv, output_csv):
    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Original shape: {df.shape}")

    # 1. Drop missing values (NaNs)
    df_clean = df.dropna()
    print(f"Shape after dropping NaNs: {df_clean.shape}")

    # Separate identifier, features (X), and target (y)
    # 'file_name' is just a string identifier, we don't scale it.
    # 'target_bug_proneness' is our Ground Truth (Y), we typically don't scale the target for classification/regression unless necessary.
    identifiers = df_clean['file_name'].reset_index(drop=True)
    target = df_clean['target_bug_proneness'].reset_index(drop=True)
    features = df_clean.drop(columns=['file_name', 'target_bug_proneness'])

    # 2. Normalize features
    # Note: You mentioned "StandardScaler to normalize between 0 and 1". 
    # StandardScaler transforms data to have a mean of 0 and standard deviation of 1 (Z-score). 
    # MinMaxScaler specifically scales data strictly between 0 and 1. 
    # Using MinMaxScaler here since Neural Networks love 0-to-1 bounds!
    scaler = MinMaxScaler()
    
    scaled_feature_array = scaler.fit_transform(features)
    
    # Reconstruct the DataFrame
    scaled_features_df = pd.DataFrame(scaled_feature_array, columns=features.columns)
    
    # Combine back together
    processed_df = pd.concat([identifiers, scaled_features_df, target], axis=1)
    
    processed_df.to_csv(output_csv, index=False)
    print(f"Processed dataset successfully saved to: {output_csv}")
    
    return processed_df

if __name__ == "__main__":
    input_file = "flask_dataset.csv"
    output_file = "flask_dataset_processed.csv"
    
    processed_data = preprocess_dataset(input_file, output_file)
    print("\nSample of normalized features (first 3 rows):")
    print(processed_data.head(3))
