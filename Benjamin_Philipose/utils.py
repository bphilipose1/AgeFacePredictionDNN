from PIL import Image
import torch
import numpy as np
def preprocess_and_save_image(input_path, output_path, new_size=(64, 64)):
    with Image.open(input_path) as img:
        img = img.resize(new_size)  # Resize the image to the specified dimensions
        img = img.convert('L')  # Convert to grayscale
        img.save(output_path)  # Save the preprocessed image to the specified output path
        
def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    
# Loss function
def mean_squared_error(x : np.ndarray, y : np.ndarray, theta : np.ndarray) -> np.ndarray:
    yhat = x @ theta 
    error = yhat - y 
    loss = (1 / len(y)) * np.sum(error ** 2) 
    return loss

# Gradient descent
def calculate_gradient_and_update(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float) -> tuple([float, np.ndarray]):
    gradient = (1 / len(y)) * x.T @ ((x @ theta) - y) 
    theta_new = theta - (alpha * gradient) 
    loss = mean_squared_error(x, y, theta_new)
    return loss, theta_new

def cnn_train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    for _ in range(dataloader.num_batches_per_epoch):
        optimizer.zero_grad()
        batch = dataloader.fetch_batch()
        x_batch = batch['img_batch'].to(device)
        y_batch = batch['age_batch'].to(device)
        yhat = torch.squeeze(model(x_batch))
        loss = torch.mean(loss_fn(yhat, y_batch))
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)

def cnn_val_step(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(dataloader.num_batches_per_epoch):
            batch = dataloader.fetch_batch()
            x_batch = batch['img_batch'].to(device)
            y_batch = batch['age_batch'].to(device)
            yhat = torch.squeeze(model(x_batch))
            loss = loss_fn(yhat, y_batch)
            losses.append(loss.item())
    return np.mean(losses)
