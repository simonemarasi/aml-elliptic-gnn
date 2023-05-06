import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import accuracy

def train(args, model, data):
    """Train a GNN model and return the trained model."""
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    epochs = args['epochs']
    model.train()

    for epoch in range(epochs + 1):
        # Training
        optimizer.zero_grad()
        out, _ = model((data.x, data.edge_index))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.3f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
    return model

@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out, _ = model((data.x, data.edge_index))
    acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
    return acc
