import torch
import numpy as np


#Model Checkpointing Helper Functions        
def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch



# Linear Regression Helper Functions
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



# Train, Validation, and Test Step Helper Functions for Convolutional Neural Network
def cnn_train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses_per_iteration = []
    for _ in range(dataloader.num_batches_per_epoch):
        optimizer.zero_grad()
        batch = dataloader.fetch_batch()       
        x_batch = batch['img_batch'].to(device)
        y_batch = batch['age_batch'].to(device)
        yhat = model(x_batch)
        yhat = torch.squeeze(yhat, 1)  
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        optimizer.step()
        losses_per_iteration.append(loss.item())
    return losses_per_iteration

def cnn_val_step(model, dataloader, loss_fn, device):
    model.eval()
    losses_per_iteration = []
    with torch.no_grad():
        for _ in range(dataloader.num_batches_per_epoch):
            batch = dataloader.fetch_batch()           
            x_batch = batch['img_batch'].to(device)
            y_batch = batch['age_batch'].to(device)
            yhat = model(x_batch)
            yhat = torch.squeeze(yhat, 1)  
            loss = loss_fn(yhat, y_batch)
            losses_per_iteration.append(loss.item())
    return losses_per_iteration

def cnn_test_step(model, dataloader, device):
    model.eval()  
    predictions = []
    actuals = []
    with torch.no_grad():
        for _ in range(dataloader.num_batches_per_epoch):
            batch = dataloader.fetch_batch()
            x_batch = batch['img_batch'].to(device)
            y_batch = batch['age_batch'].to(device)
            yhat = model(x_batch)
            yhat = torch.squeeze(yhat, 1)  
            predictions.extend(yhat.detach().cpu().numpy())
            actuals.extend(y_batch.detach().cpu().numpy())
    return predictions, actuals



# Train, Validation, and Test Step Helper Functions for Multi-Modal Neural Network
def mmnn_train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses_per_iteration = []
    for _ in range(dataloader.num_batches_per_epoch):
        optimizer.zero_grad()
        batch = dataloader.fetch_batch()
        x_batch = batch['img_batch'].to(device)
        num_features = batch['feat_batch'].to(device)
        y_batch = batch['age_batch'].to(device)
        yhat = model(x_batch, num_features)
        yhat = torch.squeeze(yhat, 1)  
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        optimizer.step()
        losses_per_iteration.append(loss.item())
    return losses_per_iteration

def mmnn_val_step(model, dataloader, loss_fn, device):
    model.eval()
    losses_per_iteration = []
    with torch.no_grad():
        for _ in range(dataloader.num_batches_per_epoch):
            batch = dataloader.fetch_batch()
            x_batch = batch['img_batch'].to(device)
            num_features = batch['feat_batch'].to(device)
            y_batch = batch['age_batch'].to(device)
            yhat = model(x_batch, num_features)
            yhat = torch.squeeze(yhat, 1)  
            loss = loss_fn(yhat, y_batch)
            losses_per_iteration.append(loss.item())
    return losses_per_iteration

def mmnn_test_step(model, dataloader, device):
    model.eval()  
    predictions = []
    actuals = []
    with torch.no_grad():
        for _ in range(dataloader.num_batches_per_epoch):
            batch = dataloader.fetch_batch()
            x_batch = batch['img_batch'].to(device)
            num_features = batch['feat_batch'].to(device)
            y_batch = batch['age_batch'].to(device)

            yhat = model(x_batch, num_features)
            yhat = torch.squeeze(yhat, 1)  
            
            predictions.extend(yhat.detach().cpu().numpy())
            actuals.extend(y_batch.detach().cpu().numpy())
    return predictions, actuals

