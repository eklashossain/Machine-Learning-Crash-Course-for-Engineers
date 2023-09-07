# Dataset: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset
!pip install -q segmentation-models-pytorch
import os, torch
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import AdamW as AW
import torch.nn as nn
from torch.utils.data import Dataset as ds
from torch.utils.data import DataLoader as DL
from torchvision import transforms as Trans
import torch.nn.functional as F
import segmentation_models_pytorch as smpy
from PIL import Image
import albumentations as ae
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR as ocLR
# ---------------Read the dataset & Preprocess-------------------
full_path = []
image_folder = "../input/semantic-drone-dataset/dataset/semantic_drone_dataset/original_images/"
label_folder = "../input/semantic-drone-dataset/dataset/semantic_drone_dataset/label_images_semantic/"

for _, _, file_name_list in os.walk(image_folder):
    for file_name in file_name_list:
        full_path.append(file_name.split('.')[0])

df = pd.DataFrame({'index_id': full_path}, index = np.arange(0, len(full_path)))


Xtrain_val, Xtest = train_test_split(df['index_id'].values, test_size=0.2)
Xtrain, Xval = train_test_split(Xtrain_val, test_size=0.2)


# ----------------Define dataset & dataloader--------------------
class CreateTrainDataset(ds):
    def __init__(self, image_dir, mask_dir, list_files, mean, std, transform=None, patch=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.list_files = list_files
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, id):
        images = cv.imread(self.image_dir + self.list_files[id] + '.jpg')
        images = cv.cvtColor(images, cv.COLOR_BGR2RGB)
        masks = cv.imread(self.mask_dir + self.list_files[id] + '.png', cv.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=images, mask=masks)
            images = Image.fromarray(aug['image'])
            masks = aug['mask']

        if self.transform is None:
            images = Image.fromarray(images)

        tr = Trans.Compose([Trans.ToTensor(), Trans.Normalize(self.mean, self.std)])
        images = tr(images)
        masks = torch.from_numpy(masks).long()

        if self.patches:
            images, masks = self.divide_into_patches(images, masks)

        return images, masks

    def divide_into_patches(self, images, masks):
        image_patches = images.unfold(1, 512, 512).unfold(2, 768, 768)
        image_patches = image_patches.contiguous().view(3, -1, 512, 768)
        image_patches = image_patches.permute(1, 0, 2, 3)
        masks_patches = masks.unfold(0, 512, 512).unfold(1, 768, 768)
        masks_patches = masks_patches.contiguous().view(-1, 512, 768)


        return image_patches, masks_patches



# ------Transformation for train and validation dataset----------
transformation_train = ae.Compose([ae.Resize(800, 1216, interpolation=cv.INTER_NEAREST),
                                   ae.HorizontalFlip(),
                                   ae.VerticalFlip(),
                                   ae.GridDistortion(p=0.2),
                                   ae.RandomBrightnessContrast((0,0.5),(0,0.5)),
                                   ae.GaussNoise()])

transformation_val = ae.Compose([ae.Resize(800, 1216, interpolation=cv.INTER_NEAREST),
                                 ae.HorizontalFlip(),
                                 ae.GridDistortion(p=0.2)])

# ---------------Dataset & dataloader creation-------------------
mean_values = [0.485, 0.456, 0.406]
std_values  = [0.229, 0.224, 0.225]
training_dataset = CreateTrainDataset(image_folder, label_folder, Xtrain, mean_values, std_values, transformation_train, patch=False)
validation_dataset = CreateTrainDataset(image_folder, label_folder, Xval, mean_values, std_values, transformation_val, patch=False)


bs= 3 #Batch Size
train_dataloader = DL(training_dataset, 
                      batch_size=bs, shuffle=True)
val_dataloader = DL(validation_dataset, 
                    batch_size=bs, shuffle=True)


# -----------------------Define model----------------------------
Unet_model = smpy.Unet('mobilenet_v2',
                 encoder_weights='imagenet',
                 classes=23,
                 activation=None,
                 encoder_depth=5,
                 decoder_channels=[256, 128, 64, 32, 16])



# ------------------Define training functions--------------------
def get_accuracy_score(output_tensor, mask_tensor):
    with torch.no_grad():
        output_tensor = torch.argmax(F.softmax(output_tensor, dim=1), dim=1)
        correct_predictions = torch.eq(output_tensor, mask_tensor).int()
        accuracy_score = float(correct_predictions.sum()) / float(correct_predictions.numel())
    return accuracy_score

def lr_from_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(num_of_epochs, Unet_model, train_dataloader, val_dataloader, loss_criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    training_losses = []
    testing_losses = []
    validation_accuracy = []
    training_accuracy = []
    learning_rates = []
    min_val_loss = np.inf
    decrease_counter = 1;
    no_improvement_count = 0

    Unet_model.to(device)
    for e in range(num_of_epochs):
        running_loss = 0
        accuracy_score = 0
        Unet_model.train() # Set model to training mode
        for i, data in enumerate(train_dataloader): # Iterate over training data loader
            image_tiles, mask_tiles = data
            if patch: # If patch-based training enabled, image and mask need to be flattened
                bs, num_tiles, channels, height, width = image_tiles.size()
                image_tiles = image_tiles.view(-1, channels, height, width)
                mask_tiles = mask_tiles.view(-1, height, width)

            image = image_tiles.to(device); # Forward pass image
            mask_tensor = mask_tiles.to(device);
            output_tensor = Unet_model(image)
            loss = loss_criterion(output_tensor, mask_tensor) # Calculate the loss
            accuracy_score += get_accuracy_score(output_tensor, mask_tensor) #Accuracy measurement for evaluation
            loss.backward() # Backpropagate the loss
            optimizer.step() # Update model weights
            optimizer.zero_grad() # Reset gradient
            learning_rates.append(lr_from_optimizer(optimizer)) # Update learning rate
            scheduler.step()
            running_loss += loss.item()

        else:
            Unet_model.eval() # Set model to evaluation mode
            testing_loss = 0
            testing_accuracy = 0
            with torch.no_grad(): # Iterate over validation data loader
                for i, data in enumerate(val_dataloader):
                    image_tiles, mask_tiles = data
                    if patch:
                        bs, num_tiles, channels, height, width = image_tiles.size()
                        image_tiles = image_tiles.view(-1, channels, height, width)
                        mask_tiles = mask_tiles.view(-1, height, width)

                    image = image_tiles.to(device);
                    mask_tensor = mask_tiles.to(device);
                    output_tensor = Unet_model(image)
                    testing_accuracy += get_accuracy_score(output_tensor, mask_tensor)
                    loss = loss_criterion(output_tensor, mask_tensor)
                    testing_loss += loss.item()

            training_losses.append(running_loss / len(train_dataloader))
            testing_losses.append(testing_loss / len(val_dataloader))

            if min_val_loss > (testing_loss / len(val_dataloader)):
                print('Loss Decreasing {:.3f} >> {:.3f} '.format(min_val_loss, (testing_loss / len(val_dataloader))))
                min_val_loss = (testing_loss / len(val_dataloader))
                decrease_counter += 1
                if decrease_counter % 5 == 0:
                    print('Model saved')
                    torch.save(Unet_model, 'Unet-Mobilenet_v2_acc-{:.3f}.pt'.format(testing_accuracy / len(val_dataloader)))

            if (testing_loss / len(val_dataloader)) > min_val_loss:
                no_improvement_count += 1
                min_val_loss = (testing_loss / len(val_dataloader))
                print(f'The loss has not decreased for {no_improvement_count} iterations.')
                if no_improvement_count == 50:
                    print('The loss has not decreased in the last 50 iterations, so stop training.')
                    break

            training_accuracy.append(accuracy_score / len(train_dataloader))
            validation_accuracy.append(testing_accuracy / len(val_dataloader))
            print("\nEpoch {}/{}:".format(e + 1, num_of_epochs),
                  "\nTrain Loss: {:.3f}".format(running_loss / len(train_dataloader)),
                  "Val Loss: {:.3f}".format(testing_loss / len(val_dataloader)),
                  "\nTrain Accuracy: {:.3f}".format(accuracy_score / len(train_dataloader)),
                  "Val Accuracy: {:.3f}".format(testing_accuracy / len(val_dataloader)) )

    train_results = {'training_losses': training_losses, 'testing_losses': testing_losses,
               'training_accuracy': training_accuracy, 'validation_accuracy': validation_accuracy}
    return train_results


# ------------------------Training loop--------------------------
max_learning_rate = 1e-3
epoch_no = 100
w_decay = 1e-4

loss_criterion = nn.CrossEntropyLoss()
optimizer = AW(Unet_model.parameters(), lr=max_learning_rate, weight_decay=w_decay)
sched = ocLR(optimizer, max_learning_rate, epochs=epoch_no,
                                            steps_per_epoch=len(train_dataloader))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_results = fit(epoch_no, Unet_model, train_dataloader, val_dataloader, loss_criterion, optimizer, sched)

torch.save(Unet_model, 'Unet-Mobilenet.pt')

# ------------------------Plot results---------------------------
plt.plot(train_results['testing_losses'], label='val_loss', marker='o')
plt.plot(train_results['training_losses'], label='train_loss', marker='o')
plt.plot(train_results['training_accuracy'], label='train_accuracy', marker='P')
plt.plot(train_results['validation_accuracy'], label='val_accuracy', marker='P')
plt.title('Loss/Accuracy per epoch');
plt.ylabel('loss');
plt.xlabel('epoch')
plt.legend(), plt.grid()
plt.show()

train_results_df = pd.DataFrame(train_results, columns = ['training_losses','testing_losses','training_accuracy','validation_accuracy'])
train_results_df.to_csv("plot_data.csv", index = False)


# -------------------------Evaluation----------------------------
class CreateTestDataset(ds):

    def __init__(self, image_dir, mask_dir, list_files, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.list_files = list_files
        self.transform = transform

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, id):
        images = cv.imread(self.image_dir + self.list_files[id] + '.jpg')
        images = cv.cvtColor(images, cv.COLOR_BGR2RGB)
        masks = cv.imread(self.mask_dir + self.list_files[id] + '.png', cv.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=images, mask=masks)
            images = Image.fromarray(aug['image'])
            masks = aug['mask']

        if self.transform is None:
            images = Image.fromarray(images)

        masks = torch.from_numpy(masks).long()

        return images, masks


transformation_test = ae.Resize(800, 1216, interpolation=cv.INTER_NEAREST)
test_dataset = CreateTestDataset(image_folder, label_folder, Xtest, transform=transformation_test)



def predict_image_mask_acc(Unet_model, image, masks, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    Unet_model.eval()
    tr = Trans.Compose([Trans.ToTensor(), Trans.Normalize(mean, std)])
    image = tr(image)
    Unet_model.to(device);
    image = image.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        masks = masks.unsqueeze(0)

        output_tensor = Unet_model(image)
        acc = get_accuracy_score(output_tensor, masks)
        masked = torch.argmax(output_tensor, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def acc(Unet_model, test_dataset):
    accuracy_score = []
    for i in range(len(test_dataset)):
        images, masks = test_dataset[i]
        pred_mask, acc = predict_image_mask_acc(Unet_model, images, masks)
        accuracy_score.append(acc)
    return accuracy_score

t_acc = acc(Unet_model, test_dataset)

print('Test Accuracy: ', np.mean(t_acc))


# -------------Cross validate with ground truth------------------
path = '../working/fig'
if not os.path.exists(path):
    os.makedirs('../working/fig')

for n in (32, 34, 36):
    image, masks = test_dataset[n]
    pred_mask, score = predict_image_mask_acc(Unet_model, image, masks)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.imshow(image)
    ax1.set_title('Picture {:d}'.format(n));
    ax1.set_axis_off()
    ax2.imshow(masks)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()
    ax3.imshow(pred_mask)
    ax3.set_title('Predicted | Accuracy {:.3f}'.format(score))
    ax3.set_axis_off()
    plt.savefig('../working/fig/' + str(n) + '.png', format='png', dpi=300, facecolor='white', bbox_inches='tight',
                pad_inches=0.25)
    plt.show()