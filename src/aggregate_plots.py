import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def aggregate_property_plots(input_dir, output_dir, properties=['MW', 'TPSA', 'LogP', 'QED']):
    """Aggregate individual property plots into composite images."""
    
    # Get all subdirectories (sample indices)
    subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    print(f"Found {len(subdirs)} sample directories")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for prop in properties:
        # Collect all images for this property
        images = []
        labels = []
        
        for subdir in subdirs:
            img_path = os.path.join(input_dir, subdir, f'{prop}.png')
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(img)
                labels.append(f'Sample {subdir}')
        
        if not images:
            print(f"No images found for {prop}")
            continue
        
        # Create composite figure
        n_images = len(images)
        n_cols = 5  # 5 columns
        n_rows = (n_images + n_cols - 1) // n_cols  # Calculate rows needed
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 4*n_rows))
        fig.suptitle(f'{prop} Distribution Across Random Samples', fontsize=16, y=1.02)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            ax.imshow(np.array(img))
            ax.axis('off')
            ax.set_title(label, fontsize=10)
        
        # Hide empty subplots
        for i in range(n_images, n_rows * n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{prop}_aggregated.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created {output_path}")

if __name__ == "__main__":
    # Settings
    fd = os.path.dirname(os.path.dirname(__file__))
    input_dir = f'{fd}/figures/physic_property/t5chem/trained/dummy/our_slice/beam/individual'
    output_dir = f'{fd}/figures/physic_property/t5chem/trained/dummy/our_slice/beam/aggregated'
    
    # Create aggregated plots
    aggregate_property_plots(input_dir, output_dir)
    print(f"Aggregation complete. Results saved to {output_dir}")