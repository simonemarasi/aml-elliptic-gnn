import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(args, model, data):
    """Train a GNN model and return the trained model."""
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9, verbose=True)
    epochs = args['epochs']
    model.train()

    best_val_loss = float('inf')
    patience = 1000
    epochs_since_best = 0

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out, _ = model((data.x, data.edge_index))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
          val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
          val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

          # Adjust learning rate
          scheduler.step(val_loss)

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Print metrics every 10 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.3f} | '
                  f'Val Acc: {val_acc*100:.2f}%')

        # Check if early stopping criteria is met
        if epochs_since_best >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return model

@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out, _ = model((data.x, data.edge_index))
    acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
    return acc
