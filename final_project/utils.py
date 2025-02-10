import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score
from tqdm import tqdm

def cluster_loss(target: torch.tensor, reconstruced: torch.tensor, 
                 latent: torch.tensor, centroids: torch.tensor, 
                 regularization: str='softmin', l: float=0.1):
    '''
    Compute the loss function for the clustering autoencoder model.

    Parameters:
    - target (torch.tensor): the target tensor
    - reconstruced (torch.tensor): the reconstructed tensor
    - latent (torch.tensor): the latent tensor
    - centroids (torch.tensor): the centroids tensor
    - regularization (str): the regularization method to use (kmeans, softmin, norm)
    - l (float): the regularization parameter

    Returns:
    - loss: the loss value
    - labels: the labels assigned to each latent vector
    '''

    # Compute the reconstruction loss
    rec_loss = nn.MSELoss()(reconstruced, target)

    # Compute the distance of each latent vector to the nearest centroid
    distances = torch.cdist(latent, centroids)

    if regularization == 'kmeans':
        penalization, labels = torch.min(distances, dim=1)

    if regularization == 'softmin':
        softmin = nn.Softmin(dim=1)(distances)
        penalization, labels = torch.max(softmin, dim=1)
        penalization = -penalization

    if regularization == 'norm':
        penalization = torch.norm(distances, p=1, dim=1)
        labels = torch.min(distances, dim=1).indices

    # Compute the mean of the minimum distances
    cluster_loss = torch.mean(penalization)

    return rec_loss + l * cluster_loss, labels


def train_models(epochs: int, train_loader: torch.utils.data.DataLoader,  
                 model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 model_custom: torch.nn.Module, optimizer_custom: torch.optim.Optimizer, centroids:torch.tensor, l:float,
                 regulrization: str='kmeans', device: str='cpu'):
    '''
    Train the clustering autoencoder model.

    Parameters:
    - epochs (int): the number of epochs
    - train_loader (torch.utils.data.DataLoader): the training data loader
    - model (torch.nn.Module): the autoencoder model
    - optimizer (torch.optim.Optimizer): the optimizer for the autoencoder model
    - model_custom (torch.nn.Module): the clustering autoencoder model
    - optimizer_custom (torch.optim.Optimizer): the optimizer for the clustering autoencoder model
    - centroids (torch.tensor): the centroids tensor
    - l (float): the regularization parameter
    - regulrization (str): the regularization method to use (kmeans, softmin, norm)
    - device (str): the device to use for training

    Returns:
    - model: the trained autoencoder model
    - model_custom: the trained clustering autoencoder model
    - centroids: the updated centroids tensor
    '''
    
    tbar = tqdm(range(epochs))

    for epoch in tbar:
        model.train()
        model_custom.train()
        for data, _ in train_loader:
            data = data.to(device)
            
            optimizer_custom.zero_grad()
            output, latent = model_custom(data)
            loss_custom, labels = cluster_loss(data, output, latent, centroids, regulrization, l=l)
            loss_custom.backward()
            torch.nn.utils.clip_grad_norm_(model_custom.parameters(), 5)
            optimizer_custom.step()
            with torch.no_grad():
                centroids = update_centroids(latent, labels, centroids)

            output, _ = model(data)
            loss = nn.MSELoss()(output, data)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        
        if epoch > 45:
            l /= 0.95

        tbar.set_postfix(loss=loss.item(), loss_custom=loss_custom.item())

    return model, model_custom, centroids


def update_centroids(latent: torch.tensor, labels: torch.tensor, centroids:torch.tensor):
    '''
    Update the centroids based on the current batch.

    Parameters:
    - latent (torch.tensor): the latent tensor
    - labels (torch.tensor): the labels assigned to each latent vector
    - centroids (torch.tensor): the centroids tensor

    Returns:
    - new_centroids: the updated centroids tensor
    '''

    # Update centroids based on current batch (without overwriting for future batches)
    new_centroids = torch.stack([
            latent[labels == i].mean(dim=0) if torch.any(labels == i) else centroids[i] 
            for i in range(centroids.size(0))
        ])
    
    return new_centroids


def generate_min_potential_vectors(v:int, d:int, lr:float=0.01, steps:int=1000):

    '''
    Generate a set of vectors with minimal potential energy.

    Parameters:
    - v (int): the number of vectors to generate
    - d (int): the dimension of the vectors
    - lr (float): the learning rate
    - steps (int): the number of optimization steps

    Returns:
    - X: the generated vectors
    '''

    torch.manual_seed(0)  # Fix the random seed for reproducibility
    X = torch.randn((v, d), requires_grad=True)  # Ensure requires_grad for optimization
    optimizer = torch.optim.Adam([X], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        X_normalized = X / X.norm(dim=1, keepdim=True)  # Normalize rows to unit length
        similarity = torch.mm(X_normalized, X_normalized.T)  # Pairwise dot products
        loss = torch.sum(similarity**2) - torch.sum(torch.diag(similarity)**2)  # Minimize off-diagonal terms
        loss.backward()
        optimizer.step()
        
        # Re-normalize to avoid numerical drift
        with torch.no_grad():
            X /= X.norm(dim=1, keepdim=True)

    return X.detach()


def plot_3PC(encoded: np.array, tgt: np.array, centroids:np.array, labels_dict:dict, color_map:dict, title:str='Fashion MNIST (3 principal components)'):
    '''
    Plot the 3 principal components of the latent space.

    Parameters:
    - encoded (np.array): the encoded tensor
    - tgt (np.array): the target labels
    - centroids (np.array): the centroids tensor
    - labels_dict (dict): the labels dictionary
    - color_map (dict): the color map
    - title (str): the title of the plot

    Returns:
    - fig: the plotly figure
    '''

    fig = px.scatter_3d(
        x=encoded[:, 0], 
        y=encoded[:, 1], 
        z=encoded[:, 2],
        color=[labels_dict[label.item()] for label in tgt],
        color_discrete_map=color_map,
        width=800, 
        height=500,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.8))
    
    # Add centroids
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(color='rgb(255,0,255)', symbol='x', size=5, line=dict(width=1)),
        name='Centroids',
        showlegend=True
    ))

    
    fig.update_layout(
        title=dict(text=title, font=dict(color='black'), x=0.5, y=0.95, xanchor='center', yanchor='top'),
        scene=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='Latent Dim 3'
        )
    )
    
    return fig


def compute_ARI(assignment: np.array, target:np.array):
    '''
    Compute the Adjusted Rand Index (ARI) for a given clustering assignment.

    Parameters:
    - assignment (np.array): the clustering assignment
    - target (np.array): the target labels

    Returns:
    - ari: the Adjusted Rand Index
    '''

    return adjusted_rand_score(target, assignment)

def compute_RI(assignment: np.array, target:np.array):
    '''
    Compute the Rand Index (RI) for a given clustering assignment.

    Parameters:
    - assignment (np.array): the clustering assignment
    - target (np.array): the target labels

    Returns:
    - ri: the Rand Index
    '''

    return rand_score(target, assignment)