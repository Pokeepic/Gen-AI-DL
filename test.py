from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def view_dataset():
    """View the Minecraft 16x dataset with images and labels."""
    
    print("="*60)
    print("MINECRAFT 16x DATASET VIEWER")
    print("="*60)
    print("\nLoading dataset from HuggingFace...\n")
    
    # Load the dataset
    ds = load_dataset("James-A/Minecraft-16x-Dataset")
    
    # Print dataset info
    print("Dataset splits:")
    for split in ds.keys():
        print(f"  {split}: {len(ds[split])} samples")
    
    print("\n" + "="*60)
    print("DATASET STRUCTURE")
    print("="*60)
    
    # Show structure using train split
    train_ds = ds['train']
    print(f"\nFeatures: {train_ds.features}")
    print(f"\nFirst sample keys: {train_ds[0].keys()}")
    
    # Show first sample info
    first_sample = train_ds[0]
    print("\nFirst sample details:")
    for key, value in first_sample.items():
        if key == 'image':
            print(f"  {key}: PIL Image, size={value.size}, mode={value.mode}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("VIEWING SAMPLES")
    print("="*60)
    
    # Ask user which split to view
    print("\nAvailable splits:")
    splits = list(ds.keys())
    for i, split in enumerate(splits):
        print(f"  {i+1}. {split} ({len(ds[split])} samples)")
    
    choice = input(f"\nSelect split to view (1-{len(splits)}) [default: 1]: ").strip()
    split_idx = int(choice) - 1 if choice.isdigit() else 0
    selected_split = splits[split_idx]
    
    # Ask how many samples to view
    num_samples = input("\nHow many samples to view? [default: 9]: ").strip()
    num_samples = int(num_samples) if num_samples.isdigit() else 9
    
    # Get samples
    dataset = ds[selected_split]
    num_samples = min(num_samples, len(dataset))
    
    # Calculate grid size
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    print(f"\nDisplaying {num_samples} samples from '{selected_split}' split...")
    
    # Display samples
    for i in range(num_samples):
        sample = dataset[i]
        img = sample['image']
        label = sample.get('label', 'N/A')
        
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i}\nLabel: {label}", fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Minecraft 16x Dataset - {selected_split.upper()} Split", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    # Ask if user wants to view more
    while True:
        again = input("\nView more samples? (y/n) [default: n]: ").strip().lower()
        if again == 'y':
            start_idx = input(f"Start from index (0-{len(dataset)-1}): ").strip()
            start_idx = int(start_idx) if start_idx.isdigit() else 0
            start_idx = max(0, min(start_idx, len(dataset)-1))
            
            num_samples = input("How many samples to view? [default: 9]: ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 9
            num_samples = min(num_samples, len(dataset) - start_idx)
            
            # Calculate grid size
            cols = 3
            rows = (num_samples + cols - 1) // cols
            
            # Create figure
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            # Display samples
            for i in range(num_samples):
                sample = dataset[start_idx + i]
                img = sample['image']
                label = sample.get('label', 'N/A')
                
                axes[i].imshow(img)
                axes[i].set_title(f"Sample {start_idx + i}\nLabel: {label}", fontsize=10)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_samples, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.suptitle(f"Minecraft 16x Dataset - {selected_split.upper()} Split (from idx {start_idx})", 
                         fontsize=14, fontweight='bold', y=1.02)
            plt.show()
        else:
            break
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Show label distribution if labels exist
    if 'label' in dataset.features:
        print(f"\nLabel distribution in '{selected_split}' split:")
        labels = [sample['label'] for sample in dataset]
        unique_labels = sorted(set(labels))
        print(f"  Unique labels: {len(unique_labels)}")
        
        # Count each label
        from collections import Counter
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {label}: {count}")
        
        if len(label_counts) > 20:
            print(f"  ... and {len(label_counts) - 20} more labels")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")


def quick_view_samples(split='train', num_samples=9):
    """Quick function to view samples without prompts."""
    ds = load_dataset("James-A/Minecraft-16x-Dataset")
    dataset = ds[split]
    
    num_samples = min(num_samples, len(dataset))
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(num_samples):
        sample = dataset[i]
        img = sample['image']
        label = sample.get('label', 'N/A')
        
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i}\nLabel: {label}", fontsize=10)
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Minecraft 16x Dataset - {split.upper()}", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.show()


if __name__ == "__main__":
    # Interactive viewer
    view_dataset()
    
    # Or use quick view:
    # quick_view_samples(split='train', num_samples=12)