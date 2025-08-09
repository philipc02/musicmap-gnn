import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

### Day 3: graph construction using fma metadata

## Load and clean data

# Load metadata and audio features
tracks = pd.read_csv('../data/fma_metadata/tracks.csv', header=[0, 1], index_col=0)
genres = pd.read_csv('../data/fma_metadata/genres.csv', index_col=0)
features = pd.read_csv('../data/fma_metadata/features.csv', index_col=0, header=[0, 1, 2]) # Multi-index headers

# To check how the data looks like (how many tracks are there per top level genre? etc.) -> based on this, choose small subset of genres
print(tracks['track']['genre_top'].value_counts())
# Just like that
print(tracks['album'].columns)
print(tracks['artist'].columns)

# Choose 5 genres with decent representation
selected_genres = ['Hip-Hop', 'Rock', 'Electronic', 'Pop', 'Experimental']
mask = tracks['track']['genre_top'].isin(selected_genres)

# Clean up: only keep relevant tracks
subset_tracks = tracks[mask].copy()
subset_tracks = subset_tracks.dropna(subset=[('track', 'genre_top')]) # Drop rows with missing labels

# Load features
subset_features = features.loc[subset_tracks.index] # Align subset_features with subset_tracks
print(subset_features.shape)  # Should be [#tracks, #features]

# Map original track IDs to new node indices starting from 0
track_id_list = list(subset_tracks.index)
track_id_to_idx = {track_id: idx for idx, track_id in enumerate(track_id_list)} # map of {track ID, idx}

## Graph Construction

# Step 1: Encode genres as labels (map genres to class indices: genre -> int)
genre_to_idx = {genre: idx for idx, genre in enumerate(selected_genres)}
labels = subset_tracks[('track', 'genre_top')].map(genre_to_idx)
y = torch.tensor(labels.values, dtype=torch.long)

# Step 2: Use features as node features
# Flatten the MultiIndex columns (optional but cleaner)
subset_features.columns = ['_'.join(col).strip() for col in subset_features.columns.values]
x = torch.tensor(subset_features.loc[track_id_list].values, dtype=torch.float)  # returns features for exactly the rows listed in track_id_list, in the given order

# Step 3: Create edges – connect songs with same genre (simple idea)
edge_index = []

# Set a reasonable limit for how many edges each node should have within the genre
max_edges_per_node = 10

genre_to_track_ids = subset_tracks.groupby(('track', 'genre_top')).groups # group by genre -> each group gives list of tracks IDs belonging to that genre

for track_ids in genre_to_track_ids.values():
    track_ids = list(track_ids)
    random.shuffle(track_ids)  # Randomize to avoid systematic bias
    for i in range(len(track_ids)):
        # For each node, connect to up to 'max_edges_per_node' following nodes
        for j in range(i + 1, min(i + 1 + max_edges_per_node, len(track_ids))):
            edge_index.append((track_id_to_idx[track_ids[i]], track_id_to_idx[track_ids[j]]))
            edge_index.append((track_id_to_idx[track_ids[j]], track_id_to_idx[track_ids[i]]))  # bidirectional

# Convert to tensor (t() transposes the tensor -> from shape [num_edges, 2] to shape [2, num_edges]; contiguous(): ensures tensor is stored in memory-safe format for efficient PyG use)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges] -> 2: number of rows (source and target node), num_edges: number of columns -> one per edge 

# Step 4: Wrap into PyG Data object
data = Data(x=x, edge_index=edge_index, y=y) # x for node features, y for node label (genre), edge_index for connections between nodes (same artist)

print(data)

### Day 4: GCN model

## Define the GCN model class

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)  # in_channels: number of input features per node
        self.conv2 = GCNConv(hidden_channels, out_channels) # hidden_channels: number of hidden neurons in the first GCN layer (64 is a common choice); out_channels: number of output classes (5 here, because we are classifying into 5 genres)

    def forward(self, x, edge_index):   # takes node features and graph structure
        x = self.conv1(x, edge_index)   # graph convolution
        x = F.relu(x)   # ReLU (Rectified Linear Unit): activation function between layers in NN -> ReLU(x) = max(0, x) (if value is positive, keep, else set to zero) => helps network learn nonlinear patterns
        x = self.conv2(x, edge_index)   # graph convolution
        return x

## Instantiate the model

model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=5) # -> since we are classifying into 5 genres, the model needs to output 5 logits (raw, unnormalized output values from the final layer of a model, just before applying f.ex. softmax) per node
print(model)    # logits will be passed to F.cross_entropy() during training to compute the loss

### Day 5: train, validate and test model

## Step 1: Split the nodes into training, validation (tuning hyperparameters) and testing (final evaluation) nodes using masks
num_nodes = data.num_nodes
num_classes = len(selected_genres)  # should be 5

# Shuffle indices for reproducibility (-> set seed to get deterministic result)
# torch.manual_seed(42)
indices = torch.randperm(num_nodes) # randomly shuffles node indices so that dataset split (train/val/test) is random -> therfore avoids bias in training set (in case nodes are f.ex. grouped by genre in order)

# Define ratios and cutoffs for splits
train_ratio = 0.7
val_ratio = 0.15
train_cutoff = int(train_ratio * num_nodes)
val_cutoff = int((train_ratio + val_ratio) * num_nodes)

# Initialize masks first with zeros
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Set masks using cutoff indices
train_mask[indices[:train_cutoff]] = True
val_mask[indices[train_cutoff:val_cutoff]] = True
test_mask[indices[val_cutoff:]] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

## Step 2: Define the training loop

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()   # resets gradients from previous iteration (otherwise gradients from multiple backward passes will add up -> like this only the current gradient is used to update the weights)
    out = model(data.x, data.edge_index)   # forward pass: predict from data (using GCN class forward() function defined above)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])   # compute loss using cross entropy between predicted and desired output
    loss.backward() # compute gradients (backward pass)
    optimizer.step()    # update weights (model parameters) using gradients
    return loss.item()

@torch.no_grad()    # tells PyTorch not to track gradients when running the test() function (tracking gradients is only needed during training when we are trying to update weights)
def test(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)    # model's raw output for each node
    preds = logits.argmax(dim=1)    # picks class with highest probability for each node as predicted class

    accs = []   # vector for accuracy values
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = preds[mask] == data.y[mask]   # number of correctly predicted nodes
        acc = int(correct.sum()) / int(mask.sum())  # ratio of correctly predicted nodes to all nodes in split
        accs.append(acc)
    return accs  # [train_acc, val_acc, test_acc]

## Step 3: Train the model

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)    # Adam -> gradient descent optimizer for model's weights (uses computed gradients)

for epoch in range(1, 201): # train for 200 epochs
    loss = train(model, data, optimizer)
    train_acc, val_acc, test_acc = test(model, data)
    if epoch % 10 == 0 or epoch == 1:   # print log for every tenth epoch
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

torch.save(model.state_dict(), "../models/music_gnn_model.pth") # save model into 'models' file for later use in other scripts
print("Model saved to models/music_gnn_model.pth")
torch.save(data, "../models/music_gnn_data.pt") # save data as well for inference script to run on exact same test data as one we used for trained model
