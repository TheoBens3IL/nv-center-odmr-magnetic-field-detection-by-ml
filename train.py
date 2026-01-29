import torch
from torch import nn
from torch.utils.data import DataLoader
from split_dataset import train_val_test_split
from models import MLP

class EarlyStopping:
    '''
    Early stopping to halt training when validation loss doesn't improve.
    If no improvement after 'patience' epochs, training stops.
    '''
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience       # epochs to wait for improvement
        self.min_delta = min_delta     # minimum change to qualify as improvement
        self.best_loss = float('inf')  # best validation loss observed
        self.counter = 0               # epochs since last improvement
        self.best_state = None         # best model state

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta: # improvement observed
            self.best_loss = val_loss                  # update best loss
            self.counter = 0                           # reset counter
            self.best_state = model.state_dict()       # save best model state
        else:
            self.counter += 1                          # if no improvement, increment counter

        return self.counter >= self.patience           # return True if early stopping criterion met  


def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = "pytorch_dataset_example"
    BATCH_SIZE = 32 
    EPOCHS = 200  # Larger number of epochs to let early stopping decide when to stop
    LR = 1e-3     # Learning rate

    train_set, val_set, test_set = train_val_test_split(DATASET_DIR)

    # Get the underlying dataset to access metadata
    full_dataset = train_set.dataset

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    sample_x, sample_y = train_set[0] # get a sample to determine input/output dimensions
    input_dim = sample_x.numel()      # flatten input size
    output_dim = sample_y.numel()     # output size

    print(sample_y.min(), sample_y.max(), sample_y.mean()) # print label stats
    print(sample_x.min(), sample_x.max(), sample_x.mean()) # print input stats

    model = MLP(input_dim, output_dim, dropout=0.2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,      # optimizer to adjust
        mode='min',     # minimize validation loss
        factor=0.5,     # reduce LR by half
        patience=5,     # wait 5 epochs for improvement
        min_lr=1e-6     # minimum LR
    )

    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)

    for epoch in range(EPOCHS):
        # ===== Training Phase ===== #
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== Validation Phase ===== #
        model.eval()
        val_loss = 0.0   # validation loss
        abs_error = 0.0  # for mean absolute error per component
        sq_error = 0.0   # for root mean square error per component
        with torch.no_grad():
            for x, y in val_loader:
                x = x.view(x.size(0), -1).to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                abs_error += torch.mean(torch.abs(pred - y), dim=0)
                sq_error += torch.mean((pred - y) ** 2, dim=0)

        val_loss /= len(val_loader)  # average validation loss
        abs_error /= len(val_loader) # MAE per axis
        rmse = torch.sqrt(sq_error / len(val_loader)) # RMSE per axis
        # Normalized RMSE
        label_range = torch.tensor([full_dataset.metadata["Ax"].max() - full_dataset.metadata["Ax"].min(),
                                    full_dataset.metadata["Ay"].max() - full_dataset.metadata["Ay"].min(),
                                    full_dataset.metadata["Az"].max() - full_dataset.metadata["Az"].min()],
                                    dtype=torch.float32, device=DEVICE)
        nrmse = rmse / label_range

        # Scheduler step
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        print(f"Epoch {epoch+1:03d} | Train: {train_loss:.4e} | Val: {val_loss:.4e} | "
              f"LR: {optimizer.param_groups[0]['lr']:.3e} | "
              f"MAE: Ax {abs_error[0]:.3e} | Ay {abs_error[1]:.3e} | Az {abs_error[2]:.3e} | "
              f"NRMSE: Ax {nrmse[0]*100:.3e} | Ay {nrmse[1]*100:.3e} | Az {nrmse[2]*100:.3e}")

        # Early stopping check
        if early_stopping.step(val_loss, model):
            print("Early stopping triggered")
            break

    # Restore best model (whether early stopping was triggered or not)
    model.load_state_dict(early_stopping.best_state)
    torch.save(model.state_dict(), "mlp_odmr_best.pt")
    print(f"Best model saved as mlp_odmr_best.pt (val_loss: {early_stopping.best_loss:.4e})")


if __name__ == "__main__":
    train()