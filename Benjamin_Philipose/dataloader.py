import torch
import math

class CustomDataloader():
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, img: torch.Tensor, age: torch.Tensor, batch_size: int = 1, randomize: bool = False):
        self.img = img
        self.age = age
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)

    def get_length(self):
        return self.img.shape[0]

    def randomize_dataset(self):
        """
        This function randomizes the dataset, while maintaining the relationship between 
        img and age tensors
        """
        indices = torch.randperm(self.x.shape[0])
        self.img = self.img[indices]
        self.age = self.age[indices]

    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'x_batch': self.img[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'y_batch': self.age[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx': b_idx,
                }
            )
        self.iter = iter(batches)
    
    def fetch_batch(self):
        """
        This function calls next on the batch iterator, and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        # if the iter hasn't been generated yet
        if self.iter is None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter, None)

        # detect if this is the final batch to avoid StopIteration error
        if batch is not None and batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()
        #print what batch is being fetched
        print(batch['batch_idx'], batch['img_batch'].shape, batch['age_batch'].shape)
        return batch
