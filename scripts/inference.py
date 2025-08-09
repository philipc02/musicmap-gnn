import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 1. Define the same GCN architecture as during training

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# 2. Load data and model

data = torch.load("../models/music_gnn_data.pt", weights_only=False)    # load saved data
in_channels = data.num_node_features
hidden_channels = 64
out_channels = len(data.y.unique())  # automatically detects number of classes, should be 5

model = GCN(in_channels, hidden_channels, out_channels) # define model arcihtecture
model.load_state_dict(torch.load("../models/music_gnn_model.pth"))  # load saved model
model.eval()

print("Model and data loaded!")


# 3. Run inference

with torch.no_grad():   # run model in test mode
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)


# 4. Evaluate accuracy on test set

test_mask = data.test_mask
test_acc = (preds[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
print(f"Test Accuracy: {test_acc:.4f}") # should be same as result we got during training in pipeline script


# 5. Confusion Matrix -> shows exactly where model is making mistakes, class by class
# rows: actual labels, columns: predicted labels -> each cell: how often a given true label was predicted as another label
# perfect predictions -> all numbers only on the main diagonal
# off-diagonal numbers -> mistakes

y_true = data.y[test_mask].cpu().numpy()
y_pred = preds[test_mask].cpu().numpy()

genre_names = ['Hip-Hop', 'Rock', 'Electronic', 'Pop', 'Experimental']

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
disp.plot(cmap=plt.cm.Blues)    # plot confusion matrix stored in disp as blue-colored grid
plt.title("Confusion Matrix - Test Set")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')   # save plot showing model performance in confusion matrix
plt.show()


# 6. Show example predictions

print("\nSample predictions from test set:")
genre_map = {0: "Hip-Hop", 1: "Rock", 2: "Electronic", 3: "Pop", 4: "Experimental"}  # same as training order

test_indices = torch.where(test_mask)[0][:10]  # take first 10 test samples
for idx in test_indices:
    print(f"Track {idx.item()} | True: {genre_map[data.y[idx].item()]} | Predicted: {genre_map[preds[idx].item()]}")
