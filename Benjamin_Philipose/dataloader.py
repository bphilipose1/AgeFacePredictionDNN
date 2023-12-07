import numpy as np
import torch
from PIL import Image
import math
class CustomDataloader():
    def __init__(self, dataframe, age, batch_size=1, randomize=False):
        self.dataframe = dataframe
        self.age = age
        self.iter = None
        self.batch_size = batch_size
        self.randomize = randomize
        self.num_batches_per_epoch = math.ceil(len(self.dataframe) / self.batch_size)
        self.x = 0

    def process_image_numerical(self, img_path):
        with Image.open(img_path) as img:
            img = img.resize((64,64))  # Resize the image to the specified dimensions
            img = img.convert('L')  # Convert to grayscale
            img_array = np.asarray(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def load_batch_images(self, start_idx, end_idx):
        batch_image_paths = self.dataframe['filename'].iloc[start_idx:end_idx]
        img_batch = np.array([self.process_image_numerical(path) for path in batch_image_paths], dtype=np.float32)
        return img_batch

    def generate_iter(self):
        if self.randomize:
            shuffled_indices = np.random.permutation(len(self.dataframe))
            self.dataframe = self.dataframe.iloc[shuffled_indices].reset_index(drop=True)
            self.age = self.age[shuffled_indices]

        # Generate batch indices
        self.iter = iter([(i*self.batch_size, (i+1)*self.batch_size) for i in range(self.num_batches_per_epoch)])

    def fetch_batch(self):
        if self.iter is None or self.x >= self.num_batches_per_epoch:
            self.generate_iter()
            self.x = 0

        batch_indices = next(self.iter, None)
        if batch_indices is None:
            return None

        start_idx, end_idx = batch_indices
        img_batch = self.load_batch_images(start_idx, end_idx)
        feat_batch = self.dataframe.drop(columns=['filename']).iloc[start_idx:end_idx].to_numpy()
        age_batch = self.age[start_idx:end_idx]

        img_batch = torch.tensor(img_batch, dtype=torch.float32)
        feat_batch = torch.tensor(feat_batch, dtype=torch.float32)
        age_batch = torch.tensor(age_batch, dtype=torch.float32)
        
        batch = {'img_batch': img_batch, 'feat_batch': feat_batch, 'age_batch': age_batch, 'batch_idx': start_idx // self.batch_size}
        self.x+=1
        return batch

