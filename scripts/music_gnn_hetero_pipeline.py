import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv

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
subset_features.columns = ['_'.join(col).strip() for col in subset_features.columns.values]
print(subset_features.shape)  # Should be [#tracks, #features]

# Map original track IDs to new node indices starting from 0
track_id_list = list(subset_tracks.index)
track_id_to_idx = {track_id: idx for idx, track_id in enumerate(track_id_list)} # map of {track ID, idx}

# Graph Construction

# Step 1: Encode genres as labels (map genres to class indices: genre -> int)
genre_to_idx = {genre: idx for idx, genre in enumerate(selected_genres)}
labels = subset_tracks[('track', 'genre_top')].map(genre_to_idx)
y = torch.tensor(labels.values, dtype=torch.long)

# Step 2: Use features as node features
# Flatten the MultiIndex columns (optional but cleaner)
subset_features.columns = ['_'.join(col).strip() for col in subset_features.columns.values]
x = torch.tensor(subset_features.loc[track_id_list].values, dtype=torch.float)  # returns features for exactly the rows listed in track_id_list, in the given order

# Step 3: Build HeteroData object
data = HeteroData()
data['track'].x = x # audio features for track nodes; HeteroData objects store different node types so we need to specify using index ['track'] what type of node we are using for our edge list here
data['track'].y = y # genre labels for track nodes

def add_edges_from_group(col_tuple, rel_name, max_edges_per_node=10):
    edge_list = []
    groups = subset_tracks.groupby(col_tuple).groups   # group by genre/artist/album -> each group gives list of tracks IDs belonging to that genre/artist/album
    for group_ids in groups.values():
        ids = list(group_ids)
        if len(ids) < 2:
            continue    # if there is only one node in this group we don't need to add any edges
        random.shuffle(ids) # shuffle to avoid any bias
        for i in range(len(ids)):
            # For each node, connect to up to 'max_edges_per_node' following nodes
            for j in range(i + 1,  min(i + 1 + max_edges_per_node, len(ids))):
                edge_list.append((track_id_to_idx[ids[i]], track_id_to_idx[ids[j]]))
                edge_list.append((track_id_to_idx[ids[j]], track_id_to_idx[ids[i]]))    # bidirectional edges
    if edge_list:
        # Convert list to tensor (t() transposes the tensor -> from shape [num_edges, 2] to shape [2, num_edges]; contiguous(): ensures tensor is stored in memory-safe format for efficient PyG use)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Instead of wrapping tensor into PyG Data object we use it to create a same_genre/same_artist/same_album relation in our HeteroData object
        data['track', rel_name, 'track'].edge_index = edge_index

# Add the three edge types: run defined function above
add_edges_from_group(('track', 'genre_top'), 'same_genre')  # Group by genre, connect all tracks with the same genre
add_edges_from_group(('artist', 'name'), 'same_artist')   # Group by artist, connect all tracks with the same artist
add_edges_from_group(('album', 'title'), 'same_album')  # Look inside the album node table, take the title column, group albums with the same title, and connect them accordingly

print(data)

# Step 4: train/val/test split masks
num_nodes = data['track'].num_nodes

# Shuffle indices for reproducibility (-> set seed to get deterministic result)
# torch.manual_seed(42)
indices = torch.randperm(num_nodes)

# Define ratios and cutoffs for splits
train_ratio, val_ratio = 0.7, 0.15
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

data['track'].train_mask = train_mask   # training mask for 'track' nodes
data['track'].val_mask = val_mask
data['track'].test_mask = test_mask

# Define the heterogeneous GCN model class

class HeteroGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({   # aggregate values of convolutional layer by taking mean of the separate convolution applications on the three different edge types (genre, artist, album)
            rel: GCNConv(in_channels, hidden_channels)
            for rel in data.edge_types
        }, aggr='mean')  # in_channels: number of input features per node
        self.conv2 = HeteroConv({
            rel: GCNConv(hidden_channels, out_channels)
            for rel in data.edge_types
        }, aggr='mean') # hidden_channels: number of hidden neurons in the first GCN layer (64 is a common choice); out_channels: number of output classes (5 here, because we are classifying into 5 genres)

    def forward(self, x_dict, edge_index_dict):   # takes node features and graph structure; x_dict and edge_index_dict are needed as they store the features for the different node types ('track', 'album' and 'genre') and the edges for the different edge types ('same_genre', 'same_artist', 'same_album') respectively
        x_dict = self.conv1(x_dict, edge_index_dict)   # graph convolution
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}   # ReLU (Rectified Linear Unit): activation function between layers in NN -> ReLU(x) = max(0, x) (if value is positive, keep, else set to zero) => helps network learn nonlinear patterns
        x_dict = self.conv2(x_dict, edge_index_dict)    # graph convolution
        return x_dict

# Instantiate the model

model = HeteroGCN(in_channels=data['track'].num_node_features, hidden_channels=64, out_channels=5) # -> since we are classifying into 5 genres, the model needs to output 5 logits (raw, unnormalized output values from the final layer of a model, just before applying f.ex. softmax) per node
print(model)    # logits will be passed to F.cross_entropy() during training to compute the loss

# Step 6: Define the training loop

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()   # resets gradients from previous iteration (otherwise gradients from multiple backward passes will add up -> like this only the current gradient is used to update the weights)
    out_dict = model(data.x_dict, data.edge_index_dict)   # forward pass: predict from data (using GCN class forward() function defined above)
    out = out_dict['track']
    loss = F.cross_entropy(out[data['track'].train_mask], data['track'].y[data['track'].train_mask])   # compute loss using cross entropy between predicted and desired output
    loss.backward() # compute gradients (backward pass)
    optimizer.step()    # update weights (model parameters) using gradients
    return loss.item()

@torch.no_grad()    # tells PyTorch not to track gradients when running the test() function (tracking gradients is only needed during training when we are trying to update weights)
def test(model, data):
    model.eval()
    logits = model(data.x_dict, data.edge_index_dict)['track']    # model's raw output for each node
    preds = logits.argmax(dim=1)    # picks class with highest probability for each node as predicted class

    accs = []   # vector for accuracy values
    for mask in [data['track'].train_mask, data['track'].val_mask, data['track'].test_mask]:
        correct = preds[mask] == data['track'].y[mask]   # number of correctly predicted nodes
        acc = int(correct.sum()) / int(mask.sum())  # ratio of correctly predicted nodes to all nodes in split
        accs.append(acc)
    return accs  # [train_acc, val_acc, test_acc]

# Step 7: Train the model

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)    # Adam -> gradient descent optimizer for model's weights (uses computed gradients)

for epoch in range(1, 201): # train for 200 epochs
    loss = train(model, data, optimizer)
    train_acc, val_acc, test_acc = test(model, data)
    if epoch % 10 == 0 or epoch == 1:   # print log for every tenth epoch
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

# Save model and data

torch.save(model.state_dict(), "../models/music_hetero_gnn_model.pth") # save model into 'models' file for later use in other scripts
print("Model saved to models/music_hetero_gnn_model.pth")
torch.save(data, "../models/music_hetero_gnn_data.pt") # save data as well for inference script to run on exact same test data as one we used for trained model
