import pandas as pd
import numpy as np
import torch
from PIL import Image
import math
class CustomDataloader():
    def __init__(self, dataframe, age, batch_size=1, randomize=False):
        self.dataframe = dataframe
        self.age = age
        self.batch_size = batch_size
        self.randomize = randomize
        self.num_batches_per_epoch = math.ceil(len(self.dataframe) / self.batch_size)

    def process_image(self, img_path):
        with Image.open(img_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize
            img_array = np.asarray(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add channel dimension
        return img_array

    def load_batch_images(self, start_idx, end_idx):
        batch_image_paths = self.dataframe['filename'].iloc[start_idx:end_idx]
        img_batch = np.array([self.process_image(path) for path in batch_image_paths], dtype=np.float32)
        return img_batch

    def fetch_batch(self):
        if self.iter is None:
            self.generate_iter()

        batch = next(self.iter, None)
        if batch is not None and batch['batch_idx'] == self.num_batches_per_epoch - 1:
            self.generate_iter()

        return batch

    def generate_iter(self):
        if self.randomize:
            shuffled_indices = np.random.permutation(len(self.dataframe))
            self.dataframe = self.dataframe.iloc[shuffled_indices].reset_index(drop=True)
            self.age = self.age[shuffled_indices]

        # Generate batch indices
        self.iter = iter([(i*self.batch_size, (i+1)*self.batch_size) for i in range(self.num_batches_per_epoch)])

    def fetch_batch(self):
        batch_indices = next(self.iter, None)
        if batch_indices is None:
            return None

        start_idx, end_idx = batch_indices
        img_batch = self.load_batch_images(start_idx, end_idx)
        feat_batch = self.dataframe.drop(columns=['filename']).iloc[start_idx:end_idx].to_numpy()
        age_batch = self.age[start_idx:end_idx]

        # Convert numpy arrays to PyTorch tensors
        img_batch = torch.tensor(img_batch, dtype=torch.float32)
        feat_batch = torch.tensor(feat_batch, dtype=torch.float32)
        age_batch = torch.tensor(age_batch, dtype=torch.float32)

        if batch_indices[1] >= len(self.dataframe):
            self.generate_iter()  # Reset for the next epoch

        return {'img_batch': img_batch, 'feat_batch': feat_batch, 'age_batch': age_batch, 'batch_idx': start_idx // self.batch_size}
