import sys
sys.path.append('/mnt/disks/legacy-jupytergpu-data/cocoapi/PythonAPI')
from data_loader import get_loader
from torchvision import transforms

# Define the same transform as in 2_Training.ipynb
transform_train = transforms.Compose([ 
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# Try to create the data loader with the same parameters as in 2_Training.ipynb
try:
    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=32,
                             vocab_threshold=5,
                             vocab_from_file=False,
                             subset_size=30000,
                             cocoapi_loc='/mnt/disks/legacy-jupytergpu-data')
    print("Data loader created successfully!")
    print(f"Vocabulary size: {len(data_loader.dataset.vocab)}")
except Exception as e:
    print(f"Error creating data loader: {e}")