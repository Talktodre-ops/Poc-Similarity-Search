import pandas as pd
import numpy as np

def clean_realtor_data(input_path, output_path):
    """
    Clean and optimize the realtor dataset
    """
    # Read the dataset
    print("Loading dataset...")
    realtor = pd.read_csv(input_path)
    
    # Display initial statistics
    print("\nInitial missing values:")
    print(realtor.isnull().sum())
    
    print("\nCleaning data...")
    # Drop unnecessary columns
    realtor = realtor.drop('street', axis=1)
    
    # Handle missing values in numerical columns
    numeric_columns = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
    for col in numeric_columns:
        median_value = realtor[col].median()
        realtor[col] = realtor[col].fillna(median_value)
    
    # Handle missing values in categorical columns
    categorical_columns = ['city', 'state']
    for col in categorical_columns:
        realtor[col] = realtor[col].fillna('unknown')
    
    # Handle special cases
    realtor['brokered_by'] = realtor['brokered_by'].fillna(-1)
    realtor['zip_code'] = realtor['zip_code'].fillna(realtor['zip_code'].mode()[0])
    realtor['prev_sold_date'] = realtor['prev_sold_date'].fillna('Never')
    
    # Optimize memory usage
    print("\nOptimizing memory usage...")
    # Convert numeric columns to smaller types
    for col in numeric_columns:
        if realtor[col].dtype == 'float64':
            if realtor[col].isnull().sum() == 0:
                if realtor[col].max() < 32768 and realtor[col].min() > -32768:
                    realtor[col] = realtor[col].astype('int16')
                else:
                    realtor[col] = realtor[col].astype('int32')
            else:
                realtor[col] = realtor[col].astype('float32')
    
    # Convert categorical columns
    categorical_columns = ['status', 'city', 'state']
    for col in categorical_columns:
        realtor[col] = realtor[col].astype('category')
    
    # Save cleaned dataset
    print("\nSaving cleaned dataset...")
    realtor.to_csv(output_path, index=False)
    
    # Final statistics
    print("\nFinal missing values:")
    print(realtor.isnull().sum())
    print(f"\nCleaned dataset saved to: {output_path}")
    
    return realtor

if __name__ == "__main__":
    input_path = r'C:\Users\VICTUS\Documents\GITHUB\Poc-Similarity-Search\realtor-data.csv'
    output_path = r'C:\Users\VICTUS\Documents\GITHUB\Poc-Similarity-Search\realtor_cleaned_final.csv'
    
    cleaned_data = clean_realtor_data(input_path, output_path)  