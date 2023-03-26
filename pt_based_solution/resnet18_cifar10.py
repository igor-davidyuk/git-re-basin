from model_interface import ModelSupportingPermutations
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.use_deterministic_algorithms(False)
np.random.seed(0)


class RN18CIFAR(ModelSupportingPermutations):
    def __init__(
            self,
            weight_init_seed: int = 0,
            batching_seed: int = 0,
            dataset_rank: int = 1,
            dataset_worldsize: int = 1
    ) -> None:
        super().__init__(weight_init_seed, batching_seed, dataset_rank, dataset_worldsize)

        self.valid_dataset = None

    def get_model(self, weight_init_seed: int):
        torch.manual_seed(weight_init_seed)

        return resnet18(num_classes=10)

    def get_train_loader(self, bs=128):
        train_transform = transforms.Compose([
            transforms.RandomAffine(0, scale=[0.8, 1.2]),
            transforms.RandomCrop(32, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(30),
            transforms.ToTensor()
        ])
        train_dataset = CIFAR10(
            '.', train=True, transform=train_transform, download=True)
        indices = np.random.permutation(len(train_dataset))
        # Splitting datasets in two parts
        indices = np.array_split(
            indices, self.dataset_worldsize
            )[self.dataset_rank - 1]
        train_dataset = Subset(train_dataset, indices)
        return DataLoader(train_dataset, batch_size=bs, shuffle=True)

    def get_valid_loader(self):
        if not self.valid_dataset:
            valid_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            self.valid_dataset = CIFAR10(
                '.', train=False, transform=valid_transform, download=True)
        return DataLoader(self.valid_dataset, batch_size=128)

    def train(self, bs=512, lr=5e-3, epoches=20, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
                )
        print(f'Using device {device}')

        torch.manual_seed(self.batching_seed)
        np.random.seed(self.batching_seed)

        train_loader = self.get_train_loader(bs)

        optimizer = SGD(self.model.parameters(), momentum=0.9, lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epoches, eta_min=1e-6)
        loss_fn = nn.CrossEntropyLoss()

        self.model.to(device)
        for epoch in range(epoches):
            self.model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Reporting on epoch results
            train_loss = loss.item()
            acc, val_loss = self.validate(device, loss_fn)
            print(f'Epoch {epoch+1}:\t'
                  f'Train loss: {train_loss:4f}\t'
                  f'Test loss: {val_loss:4f}\t'
                  f'Test accuracy: {acc:4f}')

    def validate(self, device=None, loss_fn=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
                )
        valid_loader = self.get_valid_loader()
        self.model.to(device)
        self.model.eval()

        total_samples = 0
        val_score = 0
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            outputs = self.model(data)
            pred = outputs.argmax(dim=1)
            val_score += pred.eq(target).sum().cpu().numpy()
            total_samples += len(target)
        accuracy = val_score / total_samples
        if loss_fn is not None:
            loss = loss_fn(outputs, target).item()
        else:
            loss = -1
            print('(Accuracy, Loss=-1)')
        return accuracy, loss

    @property
    def weight_name_to_perm_vector(self):
        "Get Weight Name to Permutation Vector dict."
        if self._weight_name_to_perm_vector is None:
            conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
            norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
            dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

            # This is for easy blocks that use a residual connection, without any change in the number of channels.
            easyblock = lambda name, p: {
                **conv(f"{name}.conv1", p, f"P_{name}_inner"),
                **norm(f"{name}.bn1", f"P_{name}_inner"),
                **conv(f"{name}.conv2", f"P_{name}_inner", p),
                **norm(f"{name}.bn2", p)
            }

            # This is for blocks that use a residual connection, but change the number of channels via a Conv.
            shortcutblock = lambda name, p_in, p_out: {
                **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
                **norm(f"{name}.bn1", f"P_{name}_inner"),
                **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
                **norm(f"{name}.bn2", p_out),
                **conv(f"{name}.downsample.0", p_in, p_out),
                **norm(f"{name}.downsample.1", p_out),
            }
            self._weight_name_to_perm_vector = {
                **conv("conv1", None, "P_bg0"),
                **norm("bn1", "P_bg0"),
                #
                **easyblock("layer1.0", "P_bg0"),
                **easyblock("layer1.1", "P_bg0"),
                #
                **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
                **easyblock("layer2.1", "P_bg1"),

                **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
                **easyblock("layer3.1", "P_bg2"),

                **shortcutblock("layer4.0", "P_bg2", "P_bg3"),
                **easyblock("layer4.1", "P_bg3"),
                #
                **dense("fc", "P_bg3", None),
            }

        return self._weight_name_to_perm_vector


if __name__ == "__main__":
    model = RN18CIFAR(model_id=0)
    model.train()
