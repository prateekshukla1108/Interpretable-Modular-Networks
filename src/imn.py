import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os

# Hyperparameters including mixed-modality settings.
class Hyperparameters:
    def __init__(self):
        self.batch_size = 64
        self.num_modules = 2  # Two modalities: image and text
        self.image_input_size = 3072  # 32x32x3 flattened
        self.text_input_size = 100    # Dimensionality for text modality
        self.hidden_img = 512
        self.hidden_text = 128
        self.output_dim = 10          # Number of classes
        self.lambda1 = 0.1            # Alignment penalty strength
        self.lambda3 = 0.01           # Transparency (sparsity) strength
        self.num_epochs = 100
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

# Basic linear layer with Xavier initialization.
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

# A layer that applies a linear transformation followed by ReLU.
class Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Layer, self).__init__()
        self.linear = LinearLayer(in_dim, out_dim)

    def forward(self, x):
        linear_output = self.linear(x)
        return F.relu(linear_output)

# Modified Module: Only the hidden layer uses ReLU; the final layer is linear.
class Module(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Module, self).__init__()
        self.layer1 = Layer(in_dim, hidden_dim)
        self.layer2 = LinearLayer(hidden_dim, out_dim)  # No ReLU here!

    def forward(self, x):
        hidden = self.layer1(x)
        output = self.layer2(hidden)
        return output

# Dynamic attention layer: computes attention weights based on module outputs.
class AttentionLayer(nn.Module):
    def __init__(self, num_mods):
        super(AttentionLayer, self).__init__()
        # Use an FC layer to transform module summaries into attention scores.
        self.fc = nn.Linear(num_mods, num_mods)
        # Initialize with small random values (instead of zeros) to help learning.
        nn.init.uniform_(self.fc.weight, a=-0.01, b=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, module_outputs):
        # module_outputs: list of tensors, each shape (batch_size, output_dim)
        # Compute a summary (mean) for each module.
        summaries = [out.mean(dim=1, keepdim=True) for out in module_outputs]  # Each: (batch_size, 1)
        summary_cat = torch.cat(summaries, dim=1)  # Shape: (batch_size, num_mods)
        scores = self.fc(summary_cat)              # Pre-softmax attention scores: (batch_size, num_mods)
        weights = F.softmax(scores, dim=-1)         # Post-softmax attention weights.
        return weights, scores

# Alignment loss: penalizes high average attention on a specific module.
class AlignmentLoss(nn.Module):
    def __init__(self, module_index_to_penalize, penalty_strength=0.1):
        super(AlignmentLoss, self).__init__()
        self.module_index_to_penalize = module_index_to_penalize  # e.g., module 0 (color-biased)
        self.penalty_strength = penalty_strength

    def forward(self, attention_weights):
        # attention_weights shape: (batch_size, num_mods)
        avg_weight = attention_weights[:, self.module_index_to_penalize].mean()
        penalty = self.penalty_strength * avg_weight
        return penalty

# Transparency regularization loss: apply L1 norm to pre-softmax attention scores.
class TransparencyRegularizationLoss(nn.Module):
    def __init__(self, sparsity_lambda=0.01):
        super(TransparencyRegularizationLoss, self).__init__()
        self.sparsity_lambda = sparsity_lambda

    def forward(self, attention_scores):  # Use pre-softmax scores for regularization
        l1_norm = torch.mean(torch.abs(attention_scores))
        return self.sparsity_lambda * l1_norm

# Custom dataset that wraps CIFAR10 and returns an additional text feature.
class MixedCIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False, text_input_size=100):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        self.text_input_size = text_input_size
        # Generate random text features.
        # NOTE: These are synthetic noise. Replace with real text embeddings if available.
        self.text_features = [
            torch.tensor(np.random.randn(text_input_size), dtype=torch.float)
            for _ in range(len(self.cifar10))
        ]

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, label = self.cifar10[index]
        text_vector = self.text_features[index]
        return image, text_vector, label

# Network now handles two modalities (image and text) and uses dynamic, input-dependent attention.
class Network(nn.Module):
    def __init__(self, hp):
        super(Network, self).__init__()
        self.hp = hp
        self.module_list = nn.ModuleList([
            Module(hp.image_input_size, hp.hidden_img, hp.output_dim),   # Module 0: Image
            Module(hp.text_input_size, hp.hidden_text, hp.output_dim)      # Module 1: Text
        ])
        self.attention = AttentionLayer(hp.num_modules)
        self.frozen_modules_mask = [False] * hp.num_modules

    def forward(self, image, text):
        batch_size = image.size(0)
        # Process image: flatten the tensor.
        image = image.view(batch_size, -1)
        image_output = self.module_list[0](image)
        text_output = self.module_list[1](text)
        module_outputs = [image_output, text_output]
        # Compute dynamic attention weights and pre-softmax scores based on module outputs.
        attention_weights, attention_scores = self.attention(module_outputs)  # Shapes: (batch_size, num_mods)
        # Stack module outputs to shape (batch_size, num_mods, output_dim)
        outputs_stack = torch.stack(module_outputs, dim=1)
        # Multiply attention weights (expanded to match output dims) with module outputs.
        weighted_outputs = outputs_stack * attention_weights.unsqueeze(2)
        predictions = torch.sum(weighted_outputs, dim=1)  # Aggregate across modules.
        return predictions, attention_weights, attention_scores

    def freeze_module(self, module_index):
        if 0 <= module_index < len(self.module_list):
            self.frozen_modules_mask[module_index] = True
            for param in self.module_list[module_index].parameters():
                param.requires_grad = False
        else:
            raise IndexError("Module index out of range.")

    def unfreeze_module(self, module_index):
        if 0 <= module_index < len(self.module_list):
            self.frozen_modules_mask[module_index] = False
            for param in self.module_list[module_index].parameters():
                param.requires_grad = True
        else:
            raise IndexError("Module index out of range.")

def main():
    hp = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms for the image modality.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # For Windows compatibility: use num_workers=0 if running on Windows.
    num_workers = 0 if os.name == 'nt' else 4

    # Create mixed datasets that yield (image, text, label).
    train_dataset = MixedCIFAR10Dataset(root='./data', train=True, transform=transform,
                                          download=True, text_input_size=hp.text_input_size)
    test_dataset = MixedCIFAR10Dataset(root='./data', train=False, transform=transform,
                                         download=True, text_input_size=hp.text_input_size)

    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=num_workers)

    network = Network(hp).to(device)
    optimizer = optim.Adam(network.parameters(), lr=hp.lr, betas=(hp.beta1, hp.beta2), eps=hp.epsilon)
    criterion_ce = nn.CrossEntropyLoss()
    # Alignment loss penalizes module 0 (assumed to be overly color sensitive).
    criterion_align = AlignmentLoss(module_index_to_penalize=0, penalty_strength=hp.lambda1)
    # Transparency regularization loss is applied to the pre-softmax attention scores.
    criterion_trans = TransparencyRegularizationLoss(sparsity_lambda=hp.lambda3)

    best_val_accuracy = 0.0
    adam_step = 0

    # Accumulators for data-driven control.
    alignment_loss_accum = 0.0
    attention_module0_accum = 0.0
    num_batches = 0

    for epoch in range(hp.num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        network.train()

        for batch_idx, (image, text, target) in enumerate(train_loader):
            image, text, target = image.to(device), text.to(device), target.to(device)

            optimizer.zero_grad()
            predictions, attention_weights, attention_scores = network(image, text)
            loss_ce = criterion_ce(predictions, target)
            loss_align = criterion_align(attention_weights)
            loss_trans = criterion_trans(attention_scores)
            batch_loss_val = loss_ce + loss_align + loss_trans

            batch_loss_val.backward()
            optimizer.step()
            adam_step += 1

            epoch_loss += batch_loss_val.item()
            alignment_loss_accum += loss_align.item()
            attention_module0_accum += attention_weights[:, 0].mean().item()
            num_batches += 1

            if batch_idx % 100 == 0:
                avg_attention = attention_weights.mean(dim=0).detach().cpu().numpy()
                print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {batch_loss_val.item():.4f} CE: {loss_ce.item():.4f} "
                      f"Align: {loss_align.item():.4f} Trans: {loss_trans.item():.4f} "
                      f"Attention Weights: {['M'+str(m)+':'+f'{avg_attention[m]:.4f}' for m in range(hp.num_modules)]}")
            
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_alignment_loss = alignment_loss_accum / num_batches
        avg_attention_module0 = attention_module0_accum / num_batches

        print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f} "
              f"Epoch Time: {epoch_duration:.2f} seconds.")
        print(f"Epoch {epoch+1} Average Alignment Loss: {avg_alignment_loss:.4f}, "
              f"Average Attention Weight for Module 0: {avg_attention_module0:.4f}")

        # Reset accumulators for the next epoch.
        alignment_loss_accum = 0.0
        attention_module0_accum = 0.0
        num_batches = 0

        # Data-driven control: adjust lambda1 based on the average attention weight for module 0.
        if epoch > 5 and epoch % 5 == 0:
            if avg_attention_module0 > 0.5:  # Example threshold
                hp.lambda1 *= 1.2
                criterion_align = AlignmentLoss(module_index_to_penalize=0, penalty_strength=hp.lambda1)
                print(f"Epoch {epoch+1}: Increasing lambda1 to {hp.lambda1:.4f} due to high average attention on module 0.")

        # Validation phase.
        network.eval()
        total_correct = 0
        with torch.no_grad():
            for image, text, target in test_loader:
                image, text, target = image.to(device), text.to(device), target.to(device)
                predictions, _, _ = network(image, text)
                _, predicted_labels = torch.max(predictions, 1)
                total_correct += (predicted_labels == target).sum().item()

        val_accuracy = total_correct / len(test_dataset)
        print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy * 100.0:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"Epoch {epoch+1} Best Validation Accuracy: {best_val_accuracy * 100.0:.2f}% - Model improved!")

    print("Training finished.")

if __name__ == '__main__':
    main()

