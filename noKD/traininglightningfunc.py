import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.utils as vutils
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A

torch.set_float32_matmul_precision('medium')

class SegmentationDataset(Dataset):
    """Custom Dataset for loading images and masks for segmentation."""
    def __init__(self, images, masks, transform=None):
        self.images = images  # List of image tensors
        self.masks = masks    # List of mask tensors
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        image = np.transpose(image, (2, 0, 1))
        return image, mask

def load_mask(mask_path):
    try:
        mask = Image.open(mask_path).convert("L")
        return np.array(mask) > 0  # Convert to binary (True/False)
    except Exception as e:
        print(f"Error loading mask at {mask_path}: {e}")
        return np.zeros((512, 512), dtype=bool)  # Return an all-black placeholder

def combine_masks(tumor_mask_path, other_mask_path, background_mask_path):
    try:
        tumor_mask = load_mask(tumor_mask_path)
        other_mask = load_mask(other_mask_path)
        background_mask = load_mask(background_mask_path)
        
        combined_mask = np.zeros((512, 512), dtype=np.uint8)
        combined_mask[tumor_mask > 0] = 1  # Class index for tumor
        combined_mask[other_mask > 0] = 2   # Class index for other
        combined_mask[background_mask > 0] = 0  # Class index for background
        
        return combined_mask
    except Exception as e:
        print(f"Error combining masks: {e}")
        return np.zeros((512, 512), dtype=np.uint8)

def prepare_data(data_dir):
    case_numbers = [f"{i:03}" for i in range(1, 257)]
    images, masks = [], []
    
    for case_number in case_numbers:
        tumor_path = os.path.join(data_dir, "Tumors", f"case{case_number}_tumor.png")
        other_path = os.path.join(data_dir, "Others", f"case{case_number}_other.png")
        background_path = os.path.join(data_dir, "Backgrounds", f"case{case_number}_background.png")
        image_path = os.path.join(data_dir, "Images", f"case{case_number}_image.png")

        image = Image.open(image_path).convert("LA")
        combined_mask = combine_masks(tumor_path, other_path, background_path)

        images.append(np.array(image))
        masks.append(combined_mask)

    return images, masks

def split_data(images, masks):
    train_images, val_images, train_masks, val_masks = train_test_split(
        images,
        masks,
        test_size=0.2,
        random_state=42,
        shuffle=True)
    
    return (train_images, train_masks), (val_images, val_masks)


class SegmentationModel(pl.LightningModule):
    def __init__(self, hp: dict):
        super(SegmentationModel, self).__init__()
        for key in hp.keys():
            self.hparams[key]=hp[key]
        self.save_hyperparameters(self.hparams)

        if self.hparams['Arch'] == 'DLV3+':
            self.model = smp.DeepLabV3Plus(
            encoder_name=self.hparams['encoder'],
            encoder_depth=self.hparams['depth'],
    encoder_output_stride=self.hparams['encoder_output_stride'], 
     decoder_channels=self.hparams['channels'],
      decoder_atrous_rates=self.hparams['decoder_atrous_rates'],
       encoder_weights='imagenet',
        in_channels=2,
         classes = 3,
          activation = 'identity',
        aux_params = None)
        elif self.hparams['Arch'] == 'Unet':
            self.model = smp.Unet(encoder_name=self.hparams['encoder'],
   encoder_depth=self.hparams['depth'],
     decoder_use_batchnorm = self.hparams['Bnorm'],
     decoder_attention_type = self.hparams['Attn'],
      decoder_channels = self.hparams['channels'],
       encoder_weights='imagenet',
        in_channels=2,
         classes = 3,
          activation = 'identity',
        aux_params = None)
        elif self.hparams['Arch'] == 'MANet':
            self.model = smp.MAnet(
        encoder_name=self.hparams['encoder'],
        encoder_depth=self.hparams['depth'],
        decoder_use_batchnorm=self.hparams['Bnorm'],
        decoder_channels=self.hparams['channels'],
        decoder_pab_channels = self.hparams['decoder_pab_channels'],
        encoder_weights='imagenet',
        in_channels=2,
        classes=3,
        activation='identity',
        aux_params = None
    )
        elif self.hparams['Arch'] == 'PAN':
            self.model = smp.PAN(
                encoder_name=self.hparams['encoder'],
                decoder_channels=self.hparams['channels'],
                encoder_weights='imagenet',
                in_channels=2,
                classes=3,
                activation='identity',
                aux_params = None
            )
        else:
            raise AssertionError('Model architecture must be one of: DLV3+, Unet, MANet, PAN')
        
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.005)

        
    def forward(self, x):
        logits = self.model(x)    

        return logits
    

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images.float())
        
        loss = self.criterion(outputs, masks.long())
        self.log('train_loss', loss ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        dice_ = DiceLoss('multiclass', log_loss=False, from_logits=True)(outputs, masks.long())
        jacc_ = JaccardLoss('multiclass', from_logits = True, smooth = 0.001, log_loss=False)(outputs, masks.long())
        self.log_dict({
            'train_dice': 1 - dice_,
            'train_jacc': 1 - jacc_
        }, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # Log images and predictions to TensorBoard
        if batch_idx == 0:  # Log only for the first batch to avoid cluttering TensorBoard
            self.log_images(images, outputs.argmax(dim=1), masks.long(), None, "Training")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images.float())
        
        loss = self.criterion(outputs, masks.long())
        self.log('val_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        dice_ = DiceLoss('multiclass', log_loss=False, from_logits=True)(outputs, masks.long())
        jacc_ = JaccardLoss('multiclass', from_logits = True, smooth = 0.001, log_loss=False)(outputs, masks.long())
        self.log_dict({
            'val_dice': 1 - dice_,
            'val_jacc': 1 - jacc_
        }, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if batch_idx == 0:  # Log only once per validation epoch to avoid cluttering TensorBoard
            self.log_images(images, outputs.argmax(dim=1), masks.long(), None, "Validation")  # No teacher prediction during validation

        
    def log_images(self, images, preds, masks, teacher_preds, dataset_type):
        """Log images and predictions to TensorBoard."""
        
        # Pick a random index from the batch to visualize
        idx = torch.randint(0, images.size(0), (1,)).item()
        
        # Prepare the images for logging (convert to grid format)
        img_grid = vutils.make_grid(images[idx], normalize=True)
        
        pred_grid_student = vutils.make_grid(preds[idx].unsqueeze(0).float(), normalize=True)  # Student prediction
        pred_grid_teacher = vutils.make_grid(teacher_preds[idx].unsqueeze(0).float(), normalize=True) if teacher_preds is not None else None  # Teacher prediction
        
        mask_grid = vutils.make_grid(masks[idx].unsqueeze(0).float(), normalize=True)  # Target mask
        
        # Log to TensorBoard with dataset type included in the name
        self.logger.experiment.add_image(f'{dataset_type}/Image', img_grid, self.current_epoch)
        self.logger.experiment.add_image(f'{dataset_type}/Student Prediction', pred_grid_student, self.current_epoch)
        
        if pred_grid_teacher is not None:
            self.logger.experiment.add_image(f'{dataset_type}/Teacher Prediction', pred_grid_teacher, self.current_epoch)
        
        self.logger.experiment.add_image(f'{dataset_type}/Mask', mask_grid, self.current_epoch)
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams['lr_init'],
                                weight_decay=self.hparams['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        threshold=0.005,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=10)
        return [optimizer], [{'scheduler': scheduler,
                              'monitor': 'val_loss'}]