import torch
from torch import nn, optim
import time
import copy
from unet import UNet
import shutil
import os
from dataset import TextSegDataset
from sklearn.model_selection import train_test_split


model_name = 'unet'
origin_data_path = 'origindata'
data_dir = 'data'
train_path = os.path.join(data_dir, 'train')
val_path = os.path.join(data_dir, 'val')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    for x in ['train', 'val']:
        for y in ['image', 'mask']:
            if not os.path.exists(os.path.join(data_dir, x, y)):
                os.mkdir(os.path.join(data_dir, x, y))

    all_img = os.listdir(os.path.join(origin_data_path, 'image'))
    train_path, val_path = train_test_split(all_img, test_size=0.3)
    print("train_n:", len(train_path), 'val_n:', len(val_path))
    for x in train_path:
        for p in ['image', 'mask']:
            src = os.path.join(origin_data_path, p, x)
            dst = os.path.join(data_dir, 'train', p, x)
            shutil.copy(src, dst)
    for y in val_path:
        for q in ['image', 'mask']:
            src = os.path.join(origin_data_path, q, x)
            dst = os.path.join(data_dir, 'val', q, x)
            shutil.copy(src, dst)

image_datasets = {x: TextSegDataset(os.path.join(data_dir, x))
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=2)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            with open('metrics_{}.txt'.format(model_name), 'a') as mt:
                mt.write('{},{:.4f}\n'.format(phase, epoch_loss))
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest val loss: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train():
    model = UNet()
    # num_ftrs = model_ft.classifier.in_features
    # model_ft.classifier = nn.Linear(num_ftrs, len(face_expressions))
    # num_ftrs = model_ft.fc.in_features
    # # nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, len(face_expressions))

    model_ft = model.to(device)

    criterion = nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.02, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=60, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=100)
    torch.save(model_ft, 'model_{}.pth'.format(model_name))
