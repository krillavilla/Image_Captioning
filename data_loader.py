from vocabulary import Vocabulary
from torchvision import transforms
import torch.utils.data as data
import os
import torch
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import nltk
import json
import pickle
from tqdm import tqdm

def get_loader_val(transform, batch_size=1, num_workers=0, cocoapi_loc='/opt'):
    """
    Creates a data loader for the validation dataset.

    Args:
        transform: Image transform.
        batch_size: Batch size (default=1).
        num_workers: Number of workers for data loading (default=0).
        cocoapi_loc: Path to COCO API directory (default='/opt').

    Returns:
        A PyTorch DataLoader for the validation dataset.
    """
    # Define paths for validation data
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')
    vocab_file = './vocab.pkl'

    # Create a dataset similar to CoCoDataset but for validation
    class CoCoValidationDataset(data.Dataset):
        def __init__(self, transform, batch_size, annotations_file, img_folder):
            self.transform = transform
            self.batch_size = batch_size
            self.img_folder = img_folder
            self.coco = COCO(annotations_file)

            # Get image IDs from the validation set
            self.img_ids = list(self.coco.imgs.keys())

            # Load vocabulary from file
            with open(vocab_file, 'rb') as f:
                self.vocab = pickle.load(f)

            # Define start_word, end_word for caption generation
            self.vocab.start_word = "<start>"
            self.vocab.end_word = "<end>"

        def __getitem__(self, index):
            # Get image ID
            img_id = self.img_ids[index]

            # Get image path
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Load and transform image
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image, transformed image, and image ID
            return orig_image, image, img_id

        def __len__(self):
            return len(self.img_ids)

    # Create dataset
    dataset = CoCoValidationDataset(
        transform=transform,
        batch_size=batch_size,
        annotations_file=annotations_file,
        img_folder=img_folder
    )

    # Create data loader
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for validation
        num_workers=num_workers
    )

    return data_loader

def generate_captions(encoder, decoder, data_loader, device, vocab):
    """
    Generate captions for all images in the validation set.

    Args:
        encoder: Trained encoder model.
        decoder: Trained decoder model.
        data_loader: Validation data loader.
        device: Device to run models on.
        vocab: Vocabulary object.

    Returns:
        A list of dictionaries containing image IDs and generated captions.
    """
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    results = []

    print("Generating captions for validation images...")
    with torch.no_grad():
        for i, (orig_image, image, img_id) in enumerate(tqdm(data_loader)):
            # Move image to device
            image = image.to(device)

            # Generate features
            features = encoder(image).unsqueeze(1)

            # Generate caption
            output = decoder.sample(features)

            # Convert word IDs to words
            words = []
            for idx in output:
                word = vocab.idx2word[idx]
                if word == vocab.start_word:
                    continue
                if word == vocab.end_word:
                    break
                words.append(word)

            # Create caption
            caption = ' '.join(words)

            # Add to results
            results.append({
                'image_id': img_id.item(),
                'caption': caption
            })

    return results

def save_results(results, output_file='captions_val2014_results.json'):
    """
    Save generated captions to a JSON file in COCO format.

    Args:
        results: List of dictionaries with image IDs and captions.
        output_file: Output JSON file name.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f)

    print(f"Results saved to {output_file}")

def get_loader(transform, mode='train', batch_size=1, vocab_threshold=None,
               vocab_from_file=True, num_workers=0, subset_size=None, cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in training mode).
      vocab_threshold: Minimum word count threshold.
      vocab_from_file: If True, load existing vocab file.
      num_workers: Number of workers.
      subset_size: If specified, limit the dataset to the first subset_size items.
      cocoapi_loc: Path to COCO API directory.
    """
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."

    # Based on mode (train, test), define paths to COCO images and annotations
    if mode == 'train':
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_train2014.json')
    else:
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')

    # COCO caption dataset
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_from_file=vocab_from_file,
                          annotations_file=annotations_file,
                          img_folder=img_folder,
                          subset_size=subset_size)

    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # Data loader
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_from_file, 
                 annotations_file, img_folder, subset_size=None):
        """Initialize the COCO dataset.
        Args:
            transform: Image transform.
            mode: One of 'train' or 'test'.
            batch_size: Batch size.
            vocab_threshold: Minimum word count threshold.
            vocab_from_file: If True, load existing vocab file.
            annotations_file: Path to COCO annotations file.
            img_folder: Path to COCO images folder.
            subset_size: If specified, limit the dataset to the first subset_size items.
        """
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold=vocab_threshold, vocab_from_file=vocab_from_file, annotations_file=annotations_file)
        self.img_folder = img_folder

        # Load COCO dataset
        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())

        # Limit dataset size if subset_size is specified
        if subset_size is not None:
            self.ids = self.ids[:subset_size]

        # Get caption lengths for efficient batching if in training mode
        if mode == 'train':
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) 
                          for index in range(len(self.ids))]
            self.caption_lengths = [len(token) for token in all_tokens]

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]

        # Get image and caption
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        # Load image
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')

        # Transform image
        image = self.transform(image)

        # For training, get caption
        if self.mode == 'train':
            # Convert caption to word ids
            caption = str(coco.anns[ann_id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            caption = []
            caption.append(vocab(vocab.start_word))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab(vocab.end_word))
            caption = torch.Tensor(caption).long()
            return image, caption

        # For testing, just return the image and original image for visualization
        else:
            # Load original image for visualization
            orig_image = np.array(Image.open(os.path.join(self.img_folder, path)).convert('RGB'))
            return orig_image, image

    def get_train_indices(self):
        """Randomly select a caption length, and sample indices with that length."""
        # Group captions by length
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        # Select random indices
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.ids)
