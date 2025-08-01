import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import random

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

## Graph Construction

# Step 1: Encode genres as labels (map genres to class indices: genre -> int)
genre_to_idx = {genre: idx for idx, genre in enumerate(selected_genres)}
labels = subset_tracks[('track', 'genre_top')].map(genre_to_idx)
y = torch.tensor(labels.values, dtype=torch.long)

# Step 2: Use features as node features
# Flatten the MultiIndex columns (optional but cleaner)
subset_features.columns = ['_'.join(col).strip() for col in subset_features.columns.values]
x = torch.tensor(subset_features.values, dtype=torch.float)

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
            edge_index.append((track_ids[i], track_ids[j]))
            edge_index.append((track_ids[j], track_ids[i]))  # bidirectional

# Convert to tensor (t() transposes the tensor -> from shape [num_edges, 2] to shape [2, num_edges]; contiguous(): ensures tensor is stored in memory-safe format for efficient PyG use)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges] -> 2: number of rows (source and target node), num_edges: number of columns -> one per edge 

# Step 4: Wrap into PyG Data object
data = Data(x=x, edge_index=edge_index, y=y) # x for node features, y for node label (genre), edge_index for connections between nodes (same artist)

print(data)