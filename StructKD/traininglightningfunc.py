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

class SaveModelEveryNepochs(pl.Callback):
    def __init__(self, save_dir: str, save_interval: int = 10):
        self.save_dir = save_dir
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch % self.save_interval == 0:
            model = pl_module.student_model  # Assuming your model is stored in pl_module
            model.save_pretrained(f"{self.save_dir}/model_epoch_{current_epoch}")

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

class DistillationLoss:
    def __init__(self, temperature=2.0, reduction='mean'):
        self.temperature = temperature
        self.reduction = reduction

    def __call__(self, student_logits, teacher_logits):
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        
        return nn.KLDivLoss(reduction=self.reduction)(student_probs, teacher_probs) * (self.temperature ** 2)

def L2(f_):
    
    norm = (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8
    return norm

# Compute similarity matrix
def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)  # Flatten spatial dimensions
    return torch.einsum('icm,icn->imn', [feat, feat])  # Pairwise similarity

# Compute similarity distance loss
def sim_dis_compute(f_S, f_T):
    sim_T = similarity(f_T)
    sim_S = similarity(f_S)
    sim_err = ((sim_T - sim_S) ** 2) / ((f_T.shape[2] * f_T.shape[3]) ** 2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    
    return sim_dis
class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    """
    Pair-wise similarity loss for specific feature maps after applying pooling.

    Args:
        feat_ind (int): Index of the feature map to extract from the models.

    Input Shapes:
        preds_S (torch.Tensor): Feature maps from the student model.
            Shape: [batch_size, num_features, height, width].
        preds_T (torch.Tensor): Feature maps from the teacher model.
            Shape: [batch_size, num_features, height, width].

    Output Shapes:
        torch.Tensor: Scalar tensor representing the pair-wise similarity loss.
    """
    def __init__(self, feat_ind):
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
    
    def forward(self, preds_S, preds_T):
        """
        Forward pass to compute similarity loss on pooled feature maps.

        Args:
            preds_S (torch.Tensor): Feature maps from the student model.
                Shape: [batch_size, num_features, height, width].
            preds_T (torch.Tensor): Feature maps from the teacher model.
                Shape: [batch_size, num_features, height, width].

        Returns:
            torch.Tensor: Pair-wise similarity loss.
                Shape: Scalar value (loss).
        """
        # Extract the target feature map

        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind].detach()  # Ensure teacher's features are not backpropagated

        # Calculate pooling dimensions
        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = total_w // 2, total_h // 2  # Fixed pooling to reduce dimensions by 2

        # Apply max pooling
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)
        feat_S_pooled = maxpool(feat_S)
        feat_T_pooled = maxpool(feat_T)

        # Compute pair-wise similarity loss
        loss = self.criterion(feat_S_pooled, feat_T_pooled)
        return loss

class SegmentationModel(pl.LightningModule):
    def __init__(self, hp: dict, teacherpath: str):
        super(SegmentationModel, self).__init__()
        for key in hp.keys():
            self.hparams[key]=hp[key]
        self.save_hyperparameters(self.hparams)

        self.teacher_model = smp.from_pretrained(teacherpath).eval()
        if self.hparams['Arch'] == 'DLV3+':
            self.student_model = smp.DeepLabV3Plus(
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
            self.student_model = smp.Unet(encoder_name=self.hparams['encoder'],
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
            self.student_model = smp.MAnet(
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
            self.student_model = smp.PAN(
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
        
        self.segmentation_loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.005)
        self.pairwise_loss_fn = CriterionPairWiseforWholeFeatAfterPool(-1)
        self.distillation_loss_fn = DistillationLoss(self.hparams['temperature'], 'mean')

        
    def forward(self, x):
        features = self.student_model.encoder(x)
        decoder_output = self.student_model.decoder(*features)
        student_logits = self.student_model.segmentation_head(decoder_output)     

        return (student_logits, features) 
    

    def training_step(self, batch, batch_idx):
        images, masks = batch
        with torch.no_grad():
            teacher_features = self.teacher_model.encoder(images.float())
            decoder_output = self.teacher_model.decoder(*teacher_features)
            teacher_logits = self.teacher_model.segmentation_head(decoder_output)
        
        student_logits, student_features = self.forward(images.float())
        if self.hparams['pa']:
            similarity_loss = self.pairwise_loss_fn(student_features, teacher_features)
        else:
            similarity_loss = 0

        if self.hparams['pi']:
            distillation_loss = self.distillation_loss_fn(student_logits, teacher_logits)
        else:
            distillation_loss = 0

        if self.hparams['ho']:
            NotImplementedError("Holistic Loss is not implemented yet")
        else:
            holistic_loss = 0
        segmentation_loss = self.segmentation_loss_fn(student_logits, masks.long())

        total_loss = segmentation_loss + self.hparams['lambda_1']*(similarity_loss + distillation_loss) + self.hparams['lambda_2']*holistic_loss
        dice_ = DiceLoss('multiclass', log_loss=False, from_logits=True)(student_logits, masks.long())
        jacc_ = JaccardLoss('multiclass', from_logits = True, smooth = 0.001, log_loss=False)(student_logits, masks.long())
        self.log_dict({
            'train_dice': 1 - dice_,
            'train_jacc': 1 - jacc_,
            'similarity_loss': similarity_loss,
            'KDloss': distillation_loss,
            'Seg_train_loss':segmentation_loss,
            'total_loss': total_loss
        }, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # Log images and predictions to TensorBoard
        if batch_idx == 0:  # Log only for the first batch to avoid cluttering TensorBoard
            self.log_images(images, student_logits.argmax(dim=1), masks.long(), teacher_logits.argmax(dim=1), "Training")
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        model_output = self.forward(images.float())
    
        
        student_logits, student_features = model_output  # This is where unpacking happens

        loss = self.segmentation_loss_fn(student_logits, masks.long())

        dice_ = DiceLoss('multiclass', log_loss=False, from_logits=True)(student_logits, masks.long())
        jacc_ = JaccardLoss('multiclass', from_logits = True, smooth = 0.001, log_loss=False)(student_logits, masks.long())
        self.log_dict({
            'val_dice': 1 - dice_,
            'val_jacc': 1 - jacc_,
            'val_loss': loss
        }, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if batch_idx == 0:  # Log only once per validation epoch to avoid cluttering TensorBoard
            self.log_images(images, student_logits.argmax(dim=1), masks.long(), None, "Validation")  # No teacher prediction during validation

        
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
        optimizer = optim.AdamW(self.student_model.parameters(), lr=self.hparams['lr_init'],
                                weight_decay=self.hparams['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        threshold=0.005,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=10)
        return [optimizer], [{'scheduler': scheduler,
                              'monitor': 'val_loss'}]