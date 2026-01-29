import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from split_dataset import train_val_test_split
from models import ODMR_CNN

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

    full_dataset = train_set.dataset              # full dataset for input/output dimensions
    input_channels = full_dataset[0][0].shape[0]  # = 1
    output_dim = 3                                # (Ax, Ay, Az)

    model = ODMR_CNN(input_channels, output_dim).to(DEVICE)
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
            x = x.to(DEVICE)
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
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                abs_error += torch.mean(torch.abs(pred - y), dim=0)
                sq_error += torch.mean((pred - y) ** 2, dim=0) * x.size(0)  # sum squared error

        val_loss /= len(val_loader)  # average validation loss
        abs_error /= len(val_loader) # MAE per axis
        rmse = torch.sqrt(sq_error / len(val_loader.dataset)) # RMSE per axis
        # Normalized RMSE
        label_range = torch.tensor([full_dataset.metadata["Ax"].max() - full_dataset.metadata["Ax"].min(),
                                    full_dataset.metadata["Ay"].max() - full_dataset.metadata["Ay"].min(),
                                    full_dataset.metadata["Az"].max() - full_dataset.metadata["Az"].min()],
                                    dtype=torch.float32, device=DEVICE)
        nrmse = rmse / label_range

        # Scheduler step
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train_loss {train_loss:.3e} | Val_loss {val_loss:.3e} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"MAE Ax {abs_error[0]:.2e} Ay {abs_error[1]:.2e} Az {abs_error[2]:.2e} | "
            f"NRMSE Ax {nrmse[0]*100:.2e}% Ay {nrmse[1]*100:.2e}% Az {nrmse[2]*100:.2e}%"
        )

        # Early stopping check
        if early_stopping.step(val_loss, model):
            print("Early stopping triggered")
            break

    # Restore best model (whether early stopping was triggered or not)
    model.load_state_dict(early_stopping.best_state)
    
    # Save the best model
    save_dir = Path("saved_models")
    model_path = save_dir / f"cnn_odmr_loss_{early_stopping.best_loss:.4e}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved as {model_path} (val_loss: {early_stopping.best_loss:.3e})")


if __name__ == "__main__":
    train()