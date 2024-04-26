# %%
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import os
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
from SCOD.nn_ood.posteriors import SCOD
from SCOD.nn_ood.distributions import CategoricalLogit

from baseline.data_module import (get_cifar10_train, get_cifar10_near, get_cifar10_far, 
                         get_mnist_train, get_mnist_near, get_mnist_far1, get_mnist_far2)

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_classes = 10

        # mnist images are (1, 32, 32) (channels, width, height)
        self.layers = nn.Sequential(*[
            nn.Conv2d(1, 8, 3, 1), # (b, 8, 30, 30)
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, 1), # (b, 16, 26, 26)
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, 1), # (b, 32, 20, 20)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (b, 32, 10, 10)
            nn.Flatten(), # (b, 32*10*10)
            nn.Linear(32*10*10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ])

    def forward(self, x):
        # x: (batch_size, 1, 28, 28)
        return self.layers(x)

# %%
# Training
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target, _) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

model = Net()
model.to(device)

cifar10_train = get_cifar10_train()['dataset']
cifar10_train_loader = DataLoader(cifar10_train, batch_size=64, shuffle=True)
mnist_train = get_mnist_train()['dataset']
mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    print(f'Epoch {epoch}')
    # train(model, cifar10_train_loader, optimizer, criterion, device)
    train(model, mnist_train_loader, optimizer, criterion, device)

# %%
## Load test datasets
# # MNIST near
# mnist_near = get_mnist_near(is_train=False)["dataset"]
# mnist_near_loader = DataLoader(mnist_near, batch_size=64, shuffle=False)
# # MNIST far1
# mnist_far1 = get_mnist_far1(is_train=False)["dataset"]
# mnist_far1_loader = DataLoader(mnist_far1, batch_size=64, shuffle=False)
# # MNIST far2
# mnist_far2 = get_mnist_far2(is_train=False)["dataset"]
# mnist_far2_loader = DataLoader(mnist_far2, batch_size=64, shuffle=False)

# CIFAR near
cifar_near = get_cifar10_near(is_train=False)["dataset"]
cifar_near_loader = DataLoader(cifar_near, batch_size=64, shuffle=False)
# CIFAR far
cifar_far = get_cifar10_far(is_train=False)["dataset"]
cifar_far_loader = DataLoader(cifar_far, batch_size=64, shuffle=False)

# %%
# model.eval()

# mnist_near_features_id = []
# mnist_near_features_ood = []
# mnist_far_features_id = []
# mnist_far_features_ood = []

# # MNIST near
# with torch.no_grad():
#     for batch_idx, (data, image_labels, ood_labels) in enumerate(tqdm(mnist_near_loader)):
#         output = model(data.to(device))
#         # Check if the data is in-distribution or OOD
#         for i in range(len(image_labels)):
#             if ood_labels[i] == 0:
#                 mnist_near_features_id.append(output[i].cpu().numpy())
#             else:
#                 mnist_near_features_ood.append(output[i].cpu().numpy())

# mnist_near_features_id = np.array(mnist_near_features_id)
# mnist_near_features_ood = np.array(mnist_near_features_ood)
# print(mnist_near_features_id.shape, mnist_near_features_ood.shape)


# # MNIST far
# with torch.no_grad():
#     for batch_idx, (data, image_labels, ood_labels) in enumerate(tqdm(mnist_far1_loader)):
#         output = model(data.to(device))
#         # Check if the data is in-distribution or OOD
#         for i in range(len(image_labels)):
#             if ood_labels[i] == 0:
#                 mnist_far_features_id.append(output[i].cpu().numpy())
#             else:
#                 mnist_far_features_ood.append(output[i].cpu().numpy())

# mnist_far_features_id = np.array(mnist_far_features_id)
# mnist_far_features_ood = np.array(mnist_far_features_ood)
# print(mnist_far_features_id.shape, mnist_far_features_ood.shape)

# %%
model.eval()

cifar_near_features_id = []
cifar_near_features_ood = []
cifar_far_features_id = []
cifar_far_features_ood = []

# CIFAR near
with torch.no_grad():
    for batch_idx, (data, image_labels, ood_labels) in enumerate(tqdm(cifar_near_loader)):
        output = model(data.to(device))
        # Check if the data is in-distribution or OOD
        for i in range(len(image_labels)):
            if ood_labels[i] == 0:
                cifar_near_features_id.append(output[i].cpu().numpy())
            else:
                cifar_near_features_ood.append(output[i].cpu().numpy())

cifar_near_features_id = np.array(cifar_near_features_id)
cifar_near_features_ood = np.array(cifar_near_features_ood)
print(cifar_near_features_id.shape, cifar_near_features_ood.shape)


# CIFAR-10 far
with torch.no_grad():
    for batch_idx, (data, image_labels, ood_labels) in enumerate(tqdm(cifar_far_loader)):
        output = model(data.to(device))
        # Check if the data is in-distribution or OOD
        for i in range(len(image_labels)):
            if ood_labels[i] == 0:
                cifar_far_features_id.append(output[i].cpu().numpy())
            else:
                cifar_far_features_ood.append(output[i].cpu().numpy())

cifar_far_features_id = np.array(cifar_far_features_id)
cifar_far_features_ood = np.array(cifar_far_features_ood)
print(cifar_far_features_id.shape, cifar_far_features_ood.shape)

# %%
# mu_id_mnist_near = np.mean(mnist_near_features_id, axis=0)
# mu_ood_mnist_near = np.mean(mnist_near_features_ood, axis=0)
# sigma_id_mnist_near = np.cov(mnist_near_features_id, rowvar=False)
# sigma_ood_mnist_near = np.cov(mnist_near_features_ood, rowvar=False)

# %%
# mu_id_mnist_far = np.mean(mnist_far_features_id, axis=0)
# mu_ood_mnist_far = np.mean(mnist_far_features_ood, axis=0)
# sigma_id_mnist_far = np.cov(mnist_far_features_id, rowvar=False)
# sigma_ood_mnist_far = np.cov(mnist_far_features_ood, rowvar=False)

# %%
mu_id_cifar_near = np.mean(cifar_near_features_id, axis=0)
mu_ood_cifar_near = np.mean(cifar_near_features_ood, axis=0)
sigma_id_cifar_near = np.cov(cifar_near_features_id, rowvar=False)
sigma_ood_cifar_near = np.cov(cifar_near_features_ood, rowvar=False)

# %%
mu_id_cifar_far = np.mean(cifar_far_features_id, axis=0)
mu_ood_cifar_far = np.mean(cifar_far_features_ood, axis=0)
sigma_id_cifar_far = np.cov(cifar_far_features_id, rowvar=False)
sigma_ood_cifar_far = np.cov(cifar_far_features_ood, rowvar=False)

# %%
def kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2):
    k = len(mu1)
    
    Sigma1 = np.array(Sigma1)
    Sigma2 = np.array(Sigma2)
    
    Sigma2_inv = np.linalg.inv(Sigma2)
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)
    
    term1 = np.log(det_Sigma2 / det_Sigma1)
    term2 = np.trace(Sigma2_inv @ Sigma1)
    diff_mu = mu2 - mu1
    term3 = diff_mu.T @ Sigma2_inv @ diff_mu
    term4 = k
    
    kl = 0.5 * (term1 + term2 + term3 - term4)
    return kl

# %%
wrapped_model = SCOD(
    model,
    CategoricalLogit(),
    args={
        "device": device,
        "num_eigs": 16
    },
)
# wrapped_model.process_dataset(mnist_train.to(device))
wrapped_model.process_dataset(cifar10_train.to(device))

# %%
# # MNIST near
# test = get_mnist_near(is_train=False)["dataset"]
# test_loader = DataLoader(test, batch_size=64, shuffle=False)

# images, uncs, ood_labels = [], [], []
# for data, _, ood in tqdm(test_loader):
#     data = data.to(device)
#     output, unc = wrapped_model(data)
#     images.append(data.cpu().numpy())
#     uncs.append(unc.cpu().numpy())
#     ood_labels.append(ood)
# images = np.concatenate(images)
# uncs = np.concatenate(uncs)
# ood_labels = np.concatenate(ood_labels)

# os.makedirs('results', exist_ok=True)
# np.save('results/mnist_near_images.npy', images)
# np.save('results/mnist_near_unc.npy', uncs)
# np.save('results/mnist_near_labels.npy', ood_labels)

# %%
# # MNIST far1
# test = get_mnist_far1(is_train=False)["dataset"]
# test_loader = DataLoader(test, batch_size=64, shuffle=False)

# images, uncs, ood_labels = [], [], []
# for data, _, ood in tqdm(test_loader):
#     data = data.to(device)
#     output, unc = wrapped_model(data)
#     images.append(data.cpu().numpy())
#     uncs.append(unc.cpu().numpy())
#     ood_labels.append(ood)
# images = np.concatenate(images)
# uncs = np.concatenate(uncs)
# ood_labels = np.concatenate(ood_labels)

# os.makedirs('results', exist_ok=True)
# np.save('results/mnist_far1_images.npy', images)
# np.save('results/mnist_far1_unc.npy', uncs)
# np.save('results/mnist_far1_labels.npy', ood_labels)

# %%
# # MNIST far2
# test = get_mnist_far2(is_train=False)["dataset"]
# test_loader = DataLoader(test, batch_size=64, shuffle=False)

# images, uncs, ood_labels = [], [], []
# for data, _, ood in tqdm(test_loader):
#     data = data.to(device)
#     output, unc = wrapped_model(data)
#     images.append(data.cpu().numpy())
#     uncs.append(unc.cpu().numpy())
#     ood_labels.append(ood)
# images = np.concatenate(images)
# uncs = np.concatenate(uncs)
# ood_labels = np.concatenate(ood_labels)

# os.makedirs('results', exist_ok=True)
# np.save('results/mnist_far2_images.npy', images)
# np.save('results/mnist_far2_unc.npy', uncs)
# np.save('results/mnist_far2_labels.npy', ood_labels)

# %%
# # CIFAR near
# test = get_cifar10_near(is_train=False)["dataset"]
# test_loader = DataLoader(test, batch_size=64, shuffle=False)

# images, uncs, ood_labels = [], [], []
# for data, _, ood in tqdm(test_loader):
#     data = data.to(device)
#     output, unc = wrapped_model(data)
#     images.append(data.cpu().numpy())
#     uncs.append(unc.cpu().numpy())
#     ood_labels.append(ood)
# images = np.concatenate(images)
# uncs = np.concatenate(uncs)
# ood_labels = np.concatenate(ood_labels)

# os.makedirs('results', exist_ok=True)
# np.save('results/cifar_near_images.npy', images)
# np.save('results/cifar_near_unc.npy', uncs)
# np.save('results/cifar_near_labels.npy', ood_labels)

# %%
# CIFAR far
test = get_cifar10_far(is_train=False)["dataset"]
test_loader = DataLoader(test, batch_size=64, shuffle=False)

images, uncs, ood_labels = [], [], []
for data, _, ood in tqdm(test_loader):
    data = data.to(device)
    output, unc = wrapped_model(data)
    images.append(data.cpu().numpy())
    uncs.append(unc.cpu().numpy())
    ood_labels.append(ood)
images = np.concatenate(images)
uncs = np.concatenate(uncs)
ood_labels = np.concatenate(ood_labels)

os.makedirs('results', exist_ok=True)
np.save('results/cifar_far_images.npy', images)
np.save('results/cifar_far_unc.npy', uncs)
np.save('results/cifar_far_labels.npy', ood_labels)

# %%
# Draw ROC curves
def draw_roc_curve(uncs, ood_labels, title):
    print(uncs.shape)
    print(ood_labels.shape)
    fpr, tpr, _ = roc_curve(ood_labels, uncs)
    plt.plot(fpr, tpr, label=title)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


# AUROC for each dataset
def calc_auroc(uncs, ood_labels, title):
    auroc = roc_auc_score(ood_labels, uncs)
    return auroc

# %%
# # MNIST near
# images_mnist_near = np.load('results/mnist_near_images.npy')
# uncs_mnist_near = np.load('results/mnist_near_unc.npy')
# ood_labels_mnist_near = np.load('results/mnist_near_labels.npy')

# # MNIST far1
# images_mnist_far1 = np.load('results/mnist_far1_images.npy')
# uncs_mnist_far1 = np.load('results/mnist_far1_unc.npy')
# ood_labels_mnist_far1 = np.load('results/mnist_far1_labels.npy')

# # # MNIST far2
# # images_mnist_far2 = np.load('results/mnist_far2_images.npy')
# # uncs_mnist_far2 = np.load('results/mnist_far2_unc.npy')
# # ood_labels_mnist_far2 = np.load('results/mnist_far2_labels.npy')

# # draw_roc_curve(uncs_mnist_near, ood_labels_mnist_near, 'MNIST near')
# # draw_roc_curve(uncs_mnist_far1, ood_labels_mnist_far1, 'MNIST far1')
# # draw_roc_curve(uncs_mnist_far2, ood_labels_mnist_far2, 'MNIST far2')
# # draw_roc_curve(uncs_cifar_near, ood_labels_cifar_near, 'CIFAR near')
# # draw_roc_curve(uncs_cifar_far, ood_labels_cifar_far, 'CIFAR far')



# auroc_mnist_near = calc_auroc(uncs_mnist_near, ood_labels_mnist_near, 'MNIST near')
# auroc_mnist_far1 = calc_auroc(uncs_mnist_far1, ood_labels_mnist_far1, 'MNIST far1')
# # auroc_mnist_far2 = calc_auroc(uncs_mnist_far2, ood_labels_mnist_far2, 'MNIST far2')

# # AP for each dataset
# ap_mnist_near = average_precision_score(ood_labels_mnist_near, uncs_mnist_near)
# ap_mnist_far1 = average_precision_score(ood_labels_mnist_far1, uncs_mnist_far1)
# # ap_mnist_far2 = average_precision_score(ood_labels_mnist_far2, uncs_mnist_far2)

# kl_div_mnist_near = kl_divergence_gaussian(mu_id_mnist_near, sigma_id_mnist_near, mu_ood_mnist_near, sigma_ood_mnist_near)
# kl_div_mnist_far1=kl_divergence_gaussian(mu_id_mnist_far, sigma_id_mnist_far, mu_ood_mnist_far, sigma_ood_mnist_far)

# print(f'MNIST near KL: {kl_div_mnist_near}')
# print(f'MNIST far1 KL: {kl_div_mnist_far1}')

# print(f'MNIST near AUROC: {auroc_mnist_near}')
# print(f'MNIST far1 AUROC: {auroc_mnist_far1}')

# print(f'MNIST near AP: {ap_mnist_near}')
# print(f'MNIST far1 AP: {ap_mnist_far1}')

# %%
# CIFAR near
images_cifar_near = np.load('results/cifar_near_images.npy')
uncs_cifar_near = np.load('results/cifar_near_unc.npy')
ood_labels_cifar_near = np.load('results/cifar_near_labels.npy')

# CIFAR far
images_cifar_far = np.load('results/cifar_far_images.npy')
uncs_cifar_far = np.load('results/cifar_far_unc.npy')
ood_labels_cifar_far = np.load('results/cifar_far_labels.npy')

# draw_roc_curve(uncs_mnist_near, ood_labels_mnist_near, 'MNIST near')
# draw_roc_curve(uncs_mnist_far1, ood_labels_mnist_far1, 'MNIST far1')
# draw_roc_curve(uncs_mnist_far2, ood_labels_mnist_far2, 'MNIST far2')
# draw_roc_curve(uncs_cifar_near, ood_labels_cifar_near, 'CIFAR near')
# draw_roc_curve(uncs_cifar_far, ood_labels_cifar_far, 'CIFAR far')

auroc_cifar_near = calc_auroc(uncs_cifar_near, ood_labels_cifar_near, 'CIFAR near')
auroc_cifar_far = calc_auroc(uncs_cifar_far, ood_labels_cifar_far, 'CIFAR far')

# AP for each dataset
ap_cifar_near = average_precision_score(ood_labels_cifar_near, uncs_cifar_near)
ap_cifar_far = average_precision_score(ood_labels_cifar_far, uncs_cifar_far)

kl_div_cifar_near = kl_divergence_gaussian(mu_id_cifar_near, sigma_id_cifar_near, mu_ood_cifar_near, sigma_ood_cifar_near)
kl_div_cifar_far1=kl_divergence_gaussian(mu_id_cifar_far, sigma_id_cifar_far, mu_ood_cifar_far, sigma_ood_cifar_far)

print(f'CIFAR near KL: {kl_div_cifar_near}')
print(f'CIFAR far KL: {kl_div_cifar_far1}')

print(f'CIFAR near AUROC: {auroc_cifar_near}')
print(f'CIFAR far AUROC: {auroc_cifar_far}')

print(f'CIFAR near AP: {ap_cifar_near}')
print(f'CIFAR far AP: {ap_cifar_far}')


# %%



