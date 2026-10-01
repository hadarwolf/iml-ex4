import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from helpers import *
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)

class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        #### YOUR CODE HERE ####
        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long
        df = pd.read_csv(csv_file)
        # Convert lat/long columns to float32 tensor
        self.features = torch.tensor(df[['long','lat']].values, dtype= torch.float32)
        # Convert country column to long (integer) tensor
        self.labels = torch.tensor(df[['country']].values, dtype= torch.long).squeeze()
        #### END OF YOUR CODE ####
        pass

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        return len(self.labels)
        pass

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        return self.features[idx], self.labels[idx]
        pass
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        #### YOUR CODE HERE ####
        layers = []
        input_dim = 2

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        #### YOUR CODE HERE ####
        return self.model(x)


# def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=100, batch_size=256):
#
#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
#     testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
#
#     #### YOUR CODE HERE ####
#     # initialize your criterion and optimizer here
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     train_accs = []
#     val_accs = []
#     test_accs = []
#     train_losses = []
#     val_losses = []
#     test_losses = []
#
#
#     for ep in range(epochs):
#
#         # ==========================
#         #    TRAINING LOOP (PUT YOUR CODE HERE)
#         # ==========================
#         model.train()
#         running_loss = 0.0
#         train_correct = 0
#         train_total = 0
#
#         for inputs, labels in trainloader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # Track Loss
#             running_loss += loss.item() * inputs.size(0)
#
#             # Track Accuracy
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()
#
#         # Store Train Stats for this epoch
#         train_losses.append(running_loss / train_total)
#         train_accs.append(train_correct / train_total)
#
#         # ==========================
#         #    VALIDATION LOOP
#         # ==========================
#         model.eval()
#         with torch.no_grad():
#             # --- Validation ---
#             val_running_loss = 0.0
#             val_correct = 0
#             val_total = 0
#
#             for inputs, labels in valloader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#                 val_running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
#
#             val_losses.append(val_running_loss / val_total)
#             val_accs.append(val_correct / val_total)
#
#             # --- Test ---
#             test_running_loss = 0.0
#             test_correct = 0
#             test_total = 0
#
#             for inputs, labels in testloader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#                 test_running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 test_total += labels.size(0)
#                 test_correct += (predicted == labels).sum().item()
#
#             test_losses.append(test_running_loss / test_total)
#             test_accs.append(test_correct / test_total)
#
#         print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))
#
#     return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses

# A special model class JUST for this experiment (No Batch Norm)
class MLP_NoBN(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(MLP_NoBN, self).__init__()
        layers = []
        input_dim = 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            # MISSING BATCH NORM HERE ON PURPOSE
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=100, batch_size=256):
    # 1. Setup Loaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs, val_accs, test_accs = [], [], []
    train_losses, val_losses, test_losses = [], [], []
    all_batch_losses = []

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in trainloader:
            # SAFETY CHECK FOR BATCH SIZE 1 (Prevents Crash)
            if inputs.size(0) == 1:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm1d): m.eval()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Reset BN to train mode
            if inputs.size(0) == 1: model.train()

            all_batch_losses.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Store Epoch Stats
        train_losses.append(running_loss / train_total)
        train_accs.append(train_correct / train_total)

        # Validation Loop
        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            for inputs, labels in valloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_losses.append(val_running_loss / val_total)
            val_accs.append(val_correct / val_total)

            # Test Loop
            test_running_loss = 0.0
            test_correct = 0
            test_total = 0
            for inputs, labels in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            test_losses.append(test_running_loss / test_total)
            test_accs.append(test_correct / test_total)

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, all_batch_losses
#
# def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):
#     # 1. Setup Loaders
#     # Note: We keep drop_last=False because we want to see the noise of single items
#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
#     testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     train_accs, val_accs, test_accs = [], [], []
#     train_losses, val_losses, test_losses = [], [], []
#     all_batch_losses = []
#
#     for ep in range(epochs):
#         model.train()  # Default: Set entire model to train mode
#
#         running_loss = 0.0
#         train_correct = 0
#         train_total = 0
#
#         for inputs, labels in trainloader:
#
#             # === SAFETY CHECK FOR BATCH SIZE 1 ===
#             # If batch has only 1 item, BatchNorm will crash.
#             # We fix this by freezing BN stats just for this batch.
#             if inputs.size(0) == 1:
#                 for m in model.modules():
#                     if isinstance(m, nn.BatchNorm1d):
#                         m.eval()  # Turn off training for BN only
#             # =====================================
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # Reset BN to train mode if we changed it
#             if inputs.size(0) == 1:
#                 model.train()
#
#             # Record per-batch loss
#             all_batch_losses.append(loss.item())
#
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()
#
#         # Store Epoch Stats
#         train_losses.append(running_loss / train_total)
#         train_accs.append(train_correct / train_total)
#
#         # Validation Loop
#         model.eval()
#         with torch.no_grad():
#             val_running_loss = 0.0
#             val_correct = 0
#             val_total = 0
#             for inputs, labels in valloader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
#             val_losses.append(val_running_loss / val_total)
#             val_accs.append(val_correct / val_total)
#
#             # Test Loop
#             test_running_loss = 0.0
#             test_correct = 0
#             test_total = 0
#             for inputs, labels in testloader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 test_running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 test_total += labels.size(0)
#                 test_correct += (predicted == labels).sum().item()
#             test_losses.append(test_running_loss / test_total)
#             test_accs.append(test_correct / test_total)
#
#     return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, all_batch_losses


# if __name__ == '__main__':
#     # seed for reproducibility
#     torch.manual_seed(0)
#
#     train_dataset = EuropeDataset('train.csv')
#     val_dataset = EuropeDataset('validation.csv')
#     test_dataset = EuropeDataset('test.csv')
#
#     #### YOUR CODE HERE #####
#     # Find the number of classes, e.g.:
#     output_dim = len(train_dataset.labels.unique())
    # 2. Experiment Loop
    # learning_rates = [1.0, 0.01, 0.001, 0.00001]
    #
    # plt.figure(figsize=(10, 6))
    #
    # for lr in learning_rates:
    #     print(f"Training with LR={lr}...")
    #
    #     # CRITICAL: Re-initialize model for every LR so they start fresh
    #     model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=output_dim)
    #
    #     # Train and capture ONLY validation losses
    #     _, _, _, _, _, val_losses, _ = train(
    #         train_dataset, val_dataset, test_dataset,
    #         model, lr=lr, epochs=100, batch_size=256
    #     )
    #
    #     # Plot this specific line
    #     plt.plot(val_losses, label=f'LR={lr}')
    #
    # # 3. Finalize Plot
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation Loss')
    # plt.title('Effect of Learning Rate on Validation Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # ==========================================
    # # QUESTION: Batch Size Experiment
    # # ==========================================
    # print("\n--- Running Batch Size Experiment ---")
    #
    # # 1. Define the Table from the PDF
    # # Format: {Batch_Size : Number_of_Epochs}
    # experiments = {
    #     1: 1,
    #     16: 10,
    #     128: 50,
    #     1024: 50
    # }
    #
    # # 2. Setup Figures
    # plt.figure(figsize=(10, 6))  # Figure for Accuracy
    # fig_loss, ax_loss = plt.subplots(figsize=(10, 6))  # Figure for Stability
    #
    # for b_size, n_epochs in experiments.items():
    #     print(f"Training with Batch Size: {b_size}...")
    #
    #     # Calculate speed for your report
    #     iters_per_epoch = len(train_dataset) / b_size
    #     print(f" -> Iterations per epoch: {iters_per_epoch:.1f}")
    #
    #     # Reset Model
    #     model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=output_dim)
    #
    #     # Train
    #     # Note: We capture 'batch_losses' (the last variable) for the stability plot
    #     _, _, val_accs, _, _, _, _, batch_losses = train(
    #         train_dataset, val_dataset, test_dataset,
    #         model, lr=0.001, epochs=n_epochs, batch_size=b_size
    #     )
    #
    #     # Plot Accuracy Curve
    #     plt.plot(val_accs, label=f'Batch={b_size}')
    #
    #     # Plot Stability (Loss noise) - First 500 steps only so we can see the noise
    #     ax_loss.plot(batch_losses[:500], label=f'Batch={b_size}', alpha=0.6)
    #
    # # 3. Finalize Accuracy Graph
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation Accuracy')
    # plt.title('Effect of Batch Size on Accuracy')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # 4. Finalize Stability Graph
    # ax_loss.set_xlabel('Batch Iterations (First 500)')
    # ax_loss.set_ylabel('Training Loss')
    # ax_loss.set_title('Effect of Batch Size on Stability (Noise)')
    # ax_loss.legend()
    # ax_loss.grid(True)
    # plt.show()
    # model = MLP(6, 16, output_dim)
    


    # model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256)
    #
    # plt.figure()
    # plt.plot(train_losses, label='Train', color='red')
    # plt.plot(val_losses, label='Val', color='blue')
    # plt.plot(test_losses, label='Test', color='green')
    # plt.title('Losses')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(train_accs, label='Train', color='red')
    # plt.plot(val_accs, label='Val', color='blue')
    # plt.plot(test_accs, label='Test', color='green')
    # plt.title('Accs.')
    # plt.legend()
    # plt.show()
    #
    #
    #
    # train_data = pd.read_csv('train.csv')
    # val_data = pd.read_csv('validation.csv')
    # test_data = pd.read_csv('test.csv')
    # plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)
#
#
# #part 3
# if __name__ == '__main__':
#     # 1. Setup
#     np.random.seed(42)
#     torch.manual_seed(42)
#     train_dataset = EuropeDataset('train.csv')
#     val_dataset = EuropeDataset('validation.csv')
#     test_dataset = EuropeDataset('test.csv')
#     output_dim = len(torch.unique(train_dataset.labels))
#
#     # ==========================================
#     # QUESTION 6.1.2 Part 3: Batch Norm Comparison
#     # ==========================================
#     print("\n--- Running Batch Norm Experiment ---")
#     plt.close('all')  # Close old plots
#
#     # 1. Train Standard Model (With BN)
#     print("Training WITH Batch Norm...")
#     model_bn = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=output_dim)
#
#     # Notice we unpack the return values (index 5 is val_losses)
#     results_bn = train(
#         train_dataset, val_dataset, test_dataset, model_bn, lr=0.001, epochs=50, batch_size=256
#     )
#     val_losses_bn = results_bn[5]
#
#     # 2. Train Modified Model (Without BN)
#     print("Training WITHOUT Batch Norm...")
#     model_nobn = MLP_NoBN(num_hidden_layers=6, hidden_dim=16, output_dim=output_dim)
#
#     results_nobn = train(
#         train_dataset, val_dataset, test_dataset, model_nobn, lr=0.001, epochs=50, batch_size=256
#     )
#     val_losses_nobn = results_nobn[5]
#
#     # 3. Plot Comparison
#     plt.figure(figsize=(10, 6))
#     plt.plot(val_losses_bn, label='With Batch Norm', color='blue')
#     plt.plot(val_losses_nobn, label='Without Batch Norm', color='red')
#
#     plt.xlabel('Epochs')
#     plt.ylabel('Validation Loss')
#     plt.title('Effect of Batch Normalization')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#part 4
# if __name__ == '__main__':
#     # 1. Setup
#     np.random.seed(42)
#     torch.manual_seed(42)
#     train_dataset = EuropeDataset('train.csv')
#     val_dataset = EuropeDataset('validation.csv')
#     test_dataset = EuropeDataset('test.csv')
#     output_dim = len(torch.unique(train_dataset.labels))
#
#     # 2. Experiment Settings
#     experiments = {
#         1: 1,
#         16: 10,
#         128: 50,
#         1024: 50
#     }
#
#     # 3. Create Two Separate Windows explicitly
#     plt.close('all')  # Close any old stuck windows
#
#     # Window 1: Accuracy
#     fig_acc = plt.figure(figsize=(10, 6))
#     plt.title('Effect of Batch Size on Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Validation Accuracy')
#     plt.grid(True)
#
#     # Window 2: Stability (Loss)
#     fig_loss = plt.figure(figsize=(10, 6))
#     plt.title('Effect of Batch Size on Stability (Noise)')
#     plt.xlabel('Batch Iterations (First 500)')
#     plt.ylabel('Training Loss')
#     plt.grid(True)
#
#     # 4. Run Loop
#     for b_size, n_epochs in experiments.items():
#         print(f"Training with Batch Size: {b_size}...")
#
#         # Initialize Model
#         model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=output_dim)
#
#         # Train
#         _, _, val_accs, _, _, _, _, batch_losses = train(
#             train_dataset, val_dataset, test_dataset,
#             model, lr=0.001, epochs=n_epochs, batch_size=b_size
#         )
#
#         # --- PLOT ON WINDOW 1 (Accuracy) ---
#         plt.figure(fig_acc.number)  # Switch to Window 1
#         plt.plot(val_accs, label=f'Batch={b_size}')
#         plt.legend()  # Update legend
#
#         # --- PLOT ON WINDOW 2 (Loss/Stability) ---
#         plt.figure(fig_loss.number)  # Switch to Window 2
#         # Only plot first 500 steps to keep it readable
#         plt.plot(batch_losses[:500], label=f'Batch={b_size}', alpha=0.6)
#         plt.legend()  # Update legend
#
#     # 5. Show Both Windows
#     print("Showing plots...")
#     plt.show()

#6.2.1
def all_nodels_experiment():
    # 1. Setup Data
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')
    output_dim = len(torch.unique(train_dataset.labels))

    # ==========================================
    # SECTION 6.2: Architecture Search
    # ==========================================
    print("\n--- Running Architecture Search (Section 6.2) ---")

    # 1. Define the 7 Contestants (Depth, Width)
    configs = [
        (1, 16),
        (2, 16),
        (6, 16),
        (10, 16),
        (6, 8),
        (6, 32),
        (6, 64)
    ]

    best_val_acc = 0.0
    best_config = None
    best_model_path = "best_model_6.2.pth"

    # 2. The Tournament Loop
    print(f"{'Depth':<10} {'Width':<10} {'Val Acc':<10}")
    print("-" * 35)

    for depth, width in configs:
        model = MLP(num_hidden_layers=depth, hidden_dim=width, output_dim=output_dim)

        # CHANGE IS HERE: We capture all 8 items in 'results'
        results = train(
            train_dataset, val_dataset, test_dataset,
            model, lr=0.001, epochs=50, batch_size=256
        )

        # We need Validation Accuracy, which is at index 2
        final_val_acc = results[2][-1]

        print(f"{depth:<10} {width:<10} {final_val_acc:.4f}")

        # Check for winner
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_config = (depth, width)
            torch.save(model.state_dict(), best_model_path)

    print("-" * 35)
    print(f"WINNER: Depth={best_config[0]}, Width={best_config[1]} with Acc={best_val_acc:.4f}")

    # 3. Analyze the Winner
    print(f"\n--- Plotting Best Model ---")

    # Reload the winner
    best_model = MLP(num_hidden_layers=best_config[0], hidden_dim=best_config[1], output_dim=output_dim)
    best_model.load_state_dict(torch.load(best_model_path))

    # Run once to get data for plotting
    results_best = train(
        train_dataset, val_dataset, test_dataset,
        best_model, lr=0.001, epochs=50, batch_size=256
    )

    # CHANGE IS HERE: Extract indices 4, 5, 6 for the losses
    train_losses = results_best[4]
    val_losses = results_best[5]
    test_losses = results_best[6]

    # Plot 1: Losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title(f'Best Model Losses (D={best_config[0]}, W={best_config[1]})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot 2: Decision Boundaries
    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(
        best_model,
        test_data[['long', 'lat']].values,
        test_data['country'].values,
        f'Best Model Boundaries (D={best_config[0]}, W={best_config[1]})',
        implicit_repr=False
    )
    plt.show()

#6.2.2
def run_worst_model_experiment():
    # 1. Setup Data
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')
    output_dim = len(torch.unique(train_dataset.labels))

    # ==========================================
    # RUNNING ONLY THE WORST MODEL
    # ==========================================
    print("--- Training Worst Model (Depth=1, Width=16) ---")
    
    # 1. Build the Worst Model (Shallow)
    # Depth 1 is usually the worst because it can't learn curves well
    worst_model = MLP(num_hidden_layers=1, hidden_dim=16, output_dim=output_dim)
    
    # 2. Train it
    results = train(
        train_dataset, val_dataset, test_dataset, 
        worst_model, lr=0.001, epochs=50, batch_size=256
    )
    
    # 3. Get Data for Plots
    train_losses = results[4]
    val_losses = results[5]
    test_losses = results[6]
    final_acc = results[2][-1]

    print(f"Final Accuracy: {final_acc:.4f}")

    # 4. Plot 1: Losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Worst Model Losses (Depth=1, Width=16)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. Plot 2: Decision Boundaries
    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(
        worst_model, 
        test_data[['long', 'lat']].values, 
        test_data['country'].values, 
        'Worst Model Boundaries (Depth=1, Width=16)', 
        implicit_repr=False
    )
    plt.show()




def run_depth_experiment():    # 1. Setup Data
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')
    output_dim = len(torch.unique(train_dataset.labels))

    # ==========================================
    # EXPERIMENT: Effect of Depth (Width fixed at 16)
    # ==========================================
    print("\n--- Running Depth Experiment (Width=16) ---")
    
    # We only test these depths, keeping width constant at 16
    depths_to_test = [1, 2, 6, 10]
    fixed_width = 16
    
    # Lists to store the final results for plotting
    final_train_accs = []
    final_val_accs = []
    final_test_accs = []

    print(f"{'Depth':<10} {'Width':<10} {'Val Acc':<10}")
    print("-" * 35)

    for depth in depths_to_test:
        model = MLP(num_hidden_layers=depth, hidden_dim=fixed_width, output_dim=output_dim)
        
        # Train
        results = train(
            train_dataset, val_dataset, test_dataset, 
            model, lr=0.001, epochs=50, batch_size=256
        )
        
        # Capture the FINAL accuracy (last epoch)
        # Index 1 = train_accs, Index 2 = val_accs, Index 3 = test_accs
        final_train_accs.append(results[1][-1])
        final_val_accs.append(results[2][-1])
        final_test_accs.append(results[3][-1])
        
        print(f"{depth:<10} {fixed_width:<10} {results[2][-1]:.4f}")

    # Plotting: Accuracy vs Depth
    plt.figure(figsize=(10, 6))
    plt.plot(depths_to_test, final_train_accs, label='Train Accuracy', marker='o')
    plt.plot(depths_to_test, final_val_accs, label='Validation Accuracy', marker='o')
    plt.plot(depths_to_test, final_test_accs, label='Test Accuracy', marker='o')
    
    plt.title('Effect of Network Depth on Accuracy (Width=16)')
    plt.xlabel('Number of Hidden Layers (Depth)')
    plt.ylabel('Accuracy')
    plt.xticks(depths_to_test) # Show only 1, 2, 6, 10 on x-axis
    plt.legend()
    plt.grid(True)
    plt.show()

def run_width_experiment():
    # 1. Setup Data
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')
    output_dim = len(torch.unique(train_dataset.labels))

    # ==========================================
    # EXPERIMENT: Effect of Width (Depth fixed at 6)
    # ==========================================
    print("\n--- Running Width Experiment (Depth=6) ---")
    
    # We test these widths, keeping depth constant at 6
    widths_to_test = [8, 16, 32, 64]
    fixed_depth = 6
    
    # Lists to store results
    final_train_accs = []
    final_val_accs = []
    final_test_accs = []

    print(f"{'Depth':<10} {'Width':<10} {'Val Acc':<10}")
    print("-" * 35)

    for width in widths_to_test:
        model = MLP(num_hidden_layers=fixed_depth, hidden_dim=width, output_dim=output_dim)
        
        # Train
        results = train(
            train_dataset, val_dataset, test_dataset, 
            model, lr=0.001, epochs=50, batch_size=256
        )
        
        # Capture FINAL accuracy (last epoch)
        final_train_accs.append(results[1][-1])
        final_val_accs.append(results[2][-1])
        final_test_accs.append(results[3][-1])
        
        print(f"{fixed_depth:<10} {width:<10} {results[2][-1]:.4f}")

    # Plotting: Accuracy vs Width
    plt.figure(figsize=(10, 6))
    plt.plot(widths_to_test, final_train_accs, label='Train Accuracy', marker='o')
    plt.plot(widths_to_test, final_val_accs, label='Validation Accuracy', marker='o')
    plt.plot(widths_to_test, final_test_accs, label='Test Accuracy', marker='o')
    
    plt.title('Effect of Network Width on Accuracy (Depth=6)')
    plt.xlabel('Number of Neurons (Width)')
    plt.ylabel('Accuracy')
    plt.xticks(widths_to_test) # Show only 8, 16, 32, 64
    plt.legend()
    plt.grid(True)
    plt.show()

def run_gradient_monitoring_experiment():
    # 1. Setup Data
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')
    output_dim = len(torch.unique(train_dataset.labels))

    # ==========================================
    # MONITORING GRADIENTS (100 Layers)
    # ==========================================
    print("--- Running Gradient Monitoring Experiment ---")
    
    # 1. Initialize Deep & Narrow Model
    # 100 Hidden Layers, 4 Neurons each
    model = MLP(num_hidden_layers=100, hidden_dim=4, output_dim=output_dim)
    
    # We need a custom loop here to access the gradients manually
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Layers to track (indices correspond to the linear layers)
    layers_of_interest = [0, 30, 60, 90, 95, 99]
    
    # Storage for plot: {layer_idx: [epoch1_avg, epoch2_avg, ...]}
    grads_history = {l: [] for l in layers_of_interest}

    print(f"{'Epoch':<10} {'Status':<10}")
    print("-" * 20)

    for ep in range(10): # 10 Epochs
        model.train()
        
        # Temporary storage for this epoch's batch sums
        epoch_grad_sums = {l: 0.0 for l in layers_of_interest}
        num_batches = 0

        for inputs, labels in trainloader:
            # Safety check for BN
            if inputs.size(0) == 1: continue 

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # --- CAPTURE GRADIENTS HERE ---
            # Get all Linear layers in the model
            linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
            
            for l_idx in layers_of_interest:
                layer = linear_layers[l_idx]
                
                # Calculate Gradient Magnitude ||grad||^2
                # We sum weight grad + bias grad (if bias exists)
                w_grad = layer.weight.grad.norm(2).item() ** 2
                b_grad = layer.bias.grad.norm(2).item() ** 2 if layer.bias is not None else 0
                
                epoch_grad_sums[l_idx] += (w_grad + b_grad)
            
            optimizer.step()
            num_batches += 1

        # End of Epoch: Calculate Average and Store
        for l_idx in layers_of_interest:
            avg_grad = epoch_grad_sums[l_idx] / num_batches
            grads_history[l_idx].append(avg_grad)
        
        print(f"{ep+1:<10} {'Done'}")

    # 2. Plotting
    plt.figure(figsize=(10, 6))
    for l_idx in layers_of_interest:
        plt.plot(grads_history[l_idx], label=f'Layer {l_idx}', marker='o')

    plt.title('Mean Gradient Magnitude per Layer (100-Layer Network)')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude ||grad||^2')
    plt.yscale('log') # Log scale helps see the huge differences
    plt.legend()
    plt.grid(True)
    plt.show()

# --- BONUS: RESIDUAL MLP CLASS ---
class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # The "Skip Connection": output = new_stuff + old_stuff
        out = self.linear(x)
        out = self.bn(out)
        return self.relu(out + x) # Adding x (the input) to the output

class ResMLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(ResMLP, self).__init__()
        # Input layer (transforms 2 -> hidden_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Hidden Layers (now using ResBlocks)
        self.hidden_layers = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_hidden_layers)])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def run_residual_gradient_modified():
    # 1. Setup Data
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')
    output_dim = len(torch.unique(train_dataset.labels))

    # ==========================================
    # BONUS: RESIDUAL NETWORK GRADIENTS
    # ==========================================
    print("--- Running Bonus Experiment (Residual Network) ---")
    
    # Initialize ResNet (100 Hidden Layers)
    model = ResMLP(num_hidden_layers=100, hidden_dim=4, output_dim=output_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # We track indices 0, 30, 60, 90, 99
    # Note: in ResMLP, hidden layers are in a list 'model.hidden_layers'
    layers_of_interest = [0, 30, 60, 90, 99]
    grads_history = {l: [] for l in layers_of_interest}

    print(f"{'Epoch':<10} {'Status':<10}")
    print("-" * 20)

    for ep in range(10):
        model.train()
        epoch_grad_sums = {l: 0.0 for l in layers_of_interest}
        num_batches = 0

        for inputs, labels in trainloader:
            if inputs.size(0) == 1: continue 

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Capture Gradients from the ResBlocks
            for l_idx in layers_of_interest:
                # Access the linear layer inside the ResBlock
                block = model.hidden_layers[l_idx]
                layer = block.linear 
                
                w_grad = layer.weight.grad.norm(2).item() ** 2
                b_grad = layer.bias.grad.norm(2).item() ** 2 if layer.bias is not None else 0
                epoch_grad_sums[l_idx] += (w_grad + b_grad)
            
            optimizer.step()
            num_batches += 1

        for l_idx in layers_of_interest:
            grads_history[l_idx].append(epoch_grad_sums[l_idx] / num_batches)
        
        print(f"{ep+1:<10} {'Done'}")

    # Plotting
    plt.figure(figsize=(10, 6))
    for l_idx in layers_of_interest:
        plt.plot(grads_history[l_idx], label=f'Layer {l_idx}', marker='o')

    plt.title('Bonus: Gradients in Residual Network (100 Layers)')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude ||grad||^2')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_worst_model_experiment()
    run_depth_experiment()
    run_width_experiment()
    run_gradient_monitoring_experiment()
    run_residual_gradient_modified()