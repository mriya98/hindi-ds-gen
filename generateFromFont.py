import dataGeneration
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Generate dataset and get root directory
dataset_dir = dataGeneration.generate_dataset_from_fonts()

# Get dataset generated using fonts
# Assuming your dataset is loaded using ImageFolder
dataset = ImageFolder(root=dataset_dir, transform=None)

# Load the dataset and apply augmentaations defined in dataGeneration
# Pass the (dataset_sir, number of augmented img to generate for each sample) to the function.
augmented_dataset = dataGeneration.getAugmentedDataset(dataset_dir, 2)

# Define a DataLoader to load the augmented data
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)