import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import argparse

def generate_balanced_diabetes_splits(
    dataset_path,
    num_clients=3, 
    overlap_percentage=30,
    visualization=False,
    output_dir="."
):
    """
    Generate balanced splits of the Kaggle Diabetes dataset with equal class distribution across clients.
    
    Args:
        dataset_path: Path to the Kaggle Diabetes dataset CSV file
        num_clients: Number of clients to split data for
        overlap_percentage: Percentage of overlap between clients
        visualization: Whether to create visualization of the data distribution
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with client data splits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please download the dataset from Kaggle and provide the correct path.")
        return None
    
    # Check if this is the correct dataset
    expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    if not all(col in df.columns for col in expected_columns):
        print("This doesn't appear to be the Kaggle Diabetes dataset.")
        print(f"Expected columns: {expected_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        return None
    
    # Feature and target separation
    feature_names = df.columns[:-1].tolist()  # All columns except the last one (Outcome)
    X = df[feature_names]
    y = df['Outcome']
    
    # Add the target back to the dataframe for easier handling
    df['target'] = y  # Rename for consistency with other code
    
    # Print total dataset stats
    total_samples = len(df)
    total_positive = df['target'].sum()
    total_negative = total_samples - total_positive
    
    print(f"Total dataset: {total_samples} samples")
    print(f"Class distribution: {total_negative} negative (non-diabetic), {total_positive} positive (diabetic)")
    print(f"Diabetes rate: {total_positive/total_samples:.2%}")
    
    # Standardize features
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    
    # Split the data by class
    positive_samples = df[df['target'] == 1]
    negative_samples = df[df['target'] == 0]
    
    # Shuffle both sets
    positive_samples = positive_samples.sample(frac=1, random_state=42)
    negative_samples = negative_samples.sample(frac=1, random_state=42)
    
    # Calculate how many samples per class per client (with overlap)
    pos_per_client_base = len(positive_samples) // num_clients
    neg_per_client_base = len(negative_samples) // num_clients
    
    # Calculate overlap samples
    pos_overlap = int(pos_per_client_base * (overlap_percentage / 100))
    neg_overlap = int(neg_per_client_base * (overlap_percentage / 100))
    
    # Prepare client splits dictionary
    client_splits = {}
    
    # For visualization
    client_stats = []
    
    # Create client datasets with overlapping data
    for i in range(num_clients):
        # Calculate start and end indices for positive samples
        if i == 0:
            pos_start = 0
            pos_end = pos_per_client_base + pos_overlap
        elif i == num_clients - 1:
            pos_start = i * pos_per_client_base - pos_overlap
            pos_end = len(positive_samples)
        else:
            pos_start = i * pos_per_client_base - (pos_overlap // 2)
            pos_end = (i + 1) * pos_per_client_base + (pos_overlap // 2)
        
        # Calculate start and end indices for negative samples
        if i == 0:
            neg_start = 0
            neg_end = neg_per_client_base + neg_overlap
        elif i == num_clients - 1:
            neg_start = i * neg_per_client_base - neg_overlap
            neg_end = len(negative_samples)
        else:
            neg_start = i * neg_per_client_base - (neg_overlap // 2)
            neg_end = (i + 1) * neg_per_client_base + (neg_overlap // 2)
        
        # Ensure indices are within bounds
        pos_start = max(0, pos_start)
        pos_end = min(len(positive_samples), pos_end)
        neg_start = max(0, neg_start)
        neg_end = min(len(negative_samples), neg_end)
        
        # Get data for this client
        client_pos = positive_samples.iloc[pos_start:pos_end].copy()
        client_neg = negative_samples.iloc[neg_start:neg_end].copy()
        
        # Combine positive and negative samples
        client_df = pd.concat([client_pos, client_neg])
        
        # Shuffle the data
        client_df = client_df.sample(frac=1, random_state=42 + i)
        
        # Split into train and test sets
        train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42, stratify=client_df['target'])
        
        # Calculate statistics
        pos_count = client_df['target'].sum()
        neg_count = len(client_df) - pos_count
        pos_rate = pos_count / len(client_df)
        
        # Calculate glucose range
        glucose_min = float(client_df['Glucose'].min())
        glucose_max = float(client_df['Glucose'].max())
        
        # Store client data
        client_splits[f"client-{i+1}"] = {
            "train": train_df,
            "test": test_df,
            "split_type": "balanced-overlapping",
            "overlap_percentage": overlap_percentage,
            "glucose_range": [glucose_min, glucose_max],
            "positive_samples": int(pos_count),
            "negative_samples": int(neg_count),
            "diabetes_rate": pos_rate,
            "client_id": f"client-{i+1}"
        }
        
        # Store stats for visualization
        client_stats.append({
            'client_id': f"Client {i+1}",
            'pos_count': pos_count,
            'neg_count': neg_count,
            'total': len(client_df),
            'diabetes_rate': pos_rate,
            'glucose_min': glucose_min,
            'glucose_max': glucose_max
        })
        
        print(f"Client {i+1}: {len(client_df)} samples ({pos_count} positive, {neg_count} negative, {pos_rate:.2%} diabetes rate)")
    
    # Create visualizations
    if visualization:
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot class distribution
        client_ids = [stat['client_id'] for stat in client_stats]
        pos_counts = [stat['pos_count'] for stat in client_stats]
        neg_counts = [stat['neg_count'] for stat in client_stats]
        
        # Plot stacked bar chart
        ax1.bar(client_ids, neg_counts, label='Non-diabetic (0)', color='skyblue')
        ax1.bar(client_ids, pos_counts, bottom=neg_counts, label='Diabetic (1)', color='salmon')
        
        # Add total count labels on top of bars
        for i, stat in enumerate(client_stats):
            ax1.text(i, stat['total'] + 5, f"{stat['total']}", ha='center')
            
        # Add percentage labels inside bars for diabetic class
        for i, stat in enumerate(client_stats):
            if stat['pos_count'] > 10:  # Only add label if bar is big enough
                # Position label in the middle of the positive bar
                y_pos = stat['neg_count'] + stat['pos_count']/2
                ax1.text(i, y_pos, f"{stat['diabetes_rate']:.1%}", ha='center', va='center',
                         fontweight='bold', color='white')
        
        ax1.set_title('Class Distribution by Client')
        ax1.set_ylabel('Number of Samples')
        ax1.set_ylim(0, max([stat['total'] for stat in client_stats]) * 1.1)
        ax1.legend()
        
        # Plot glucose ranges
        client_colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Get overall glucose range
        overall_min = df['Glucose'].min()
        overall_max = df['Glucose'].max()
        
        # Plot histogram of all data
        ax2.hist(df['Glucose'], bins=30, alpha=0.3, color='gray', label='Full Dataset')
        
        # Plot histogram for each client
        for i, client_id in enumerate(client_splits.keys()):
            client_data = client_splits[client_id]['train'].append(client_splits[client_id]['test'])
            ax2.hist(client_data['Glucose'], bins=20, alpha=0.5, 
                     color=client_colors[i % len(client_colors)], 
                     label=f"Client {i+1} ({client_stats[i]['glucose_min']:.0f}-{client_stats[i]['glucose_max']:.0f})")
        
        ax2.set_title('Glucose Distribution by Client')
        ax2.set_xlabel('Glucose Level')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/balanced_diabetes_split.png")
        plt.close()
        
        print(f"Visualization saved to {output_dir}/balanced_diabetes_split.png")
    
    # Save each client's data to CSV files
    for client_id, data in client_splits.items():
        # Create header with metadata
        header = [
            f"# Dataset: Kaggle Diabetes",
            f"# Split type: balanced-overlapping",
            f"# Client ID: {client_id}",
            f"# Overlap percentage: {overlap_percentage}%",
            f"# Features: {', '.join(feature_names)}",
            f"# Target: target (0 = No diabetes, 1 = Diabetes)",
            f"# Glucose range: {data['glucose_range'][0]:.2f} to {data['glucose_range'][1]:.2f}",
            f"# Class distribution: {data['negative_samples']} non-diabetic, {data['positive_samples']} diabetic",
            f"# Diabetes rate: {data['diabetes_rate']:.2%}",
            f"# Train samples: {len(data['train'])}",
            f"# Test samples: {len(data['test'])}",
            ""  # Empty line before data
        ]
        
        # Save train and test files with header
        train_file = f"{output_dir}/{client_id}_train.txt"
        test_file = f"{output_dir}/{client_id}_test.txt"
        
        # Save train data
        with open(train_file, 'w') as f:
            # Write header
            f.write('\n'.join(header))
            # Write data
            data['train'].to_csv(f, index=False)
        
        # Save test data
        with open(test_file, 'w') as f:
            # Write header
            f.write('\n'.join(header))
            # Write data
            data['test'].to_csv(f, index=False)
    
    # Create a summary file
    with open(f"{output_dir}/dataset_summary.txt", 'w') as f:
        f.write(f"Kaggle Diabetes Dataset - Balanced Overlapping Split\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {', '.join(feature_names)}\n")
        f.write(f"Overall diabetes rate: {df['target'].mean():.2%}\n\n")
        
        f.write("Client Summaries:\n")
        for client_id, data in client_splits.items():
            f.write(f"\n{client_id}:\n")
            f.write(f"  - Split type: balanced-overlapping\n")
            f.write(f"  - Overlap percentage: {overlap_percentage}%\n")
            f.write(f"  - Glucose range: {data['glucose_range'][0]:.2f} to {data['glucose_range'][1]:.2f}\n")
            f.write(f"  - Class distribution: {data['negative_samples']} non-diabetic, {data['positive_samples']} diabetic\n")
            f.write(f"  - Diabetes rate: {data['diabetes_rate']:.2%}\n")
            f.write(f"  - Train samples: {len(data['train'])}\n")
            f.write(f"  - Test samples: {len(data['test'])}\n")
    
    return client_splits

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate balanced Kaggle Diabetes dataset splits for federated learning")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to the Kaggle Diabetes dataset CSV file")
    parser.add_argument("--num-clients", type=int, default=3,
                      help="Number of clients to generate data for")
    parser.add_argument("--overlap", type=int, default=30,
                      help="Percentage of overlap between clients")
    parser.add_argument("--output-dir", type=str, default=".",
                      help="Directory to save the output files")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of the data distribution")
    
    args = parser.parse_args()
    
    # Generate the dataset splits
    print(f"Generating balanced dataset splits for {args.num_clients} clients with {args.overlap}% overlap...")
    
    splits = generate_balanced_diabetes_splits(
        dataset_path=args.dataset,
        num_clients=args.num_clients,
        overlap_percentage=args.overlap,
        visualization=args.visualize,
        output_dir=args.output_dir
    )
    
    if splits:
        print(f"Done! Files created in {args.output_dir}:")
        for client_id in splits.keys():
            print(f"  {client_id}_train.txt")
            print(f"  {client_id}_test.txt")