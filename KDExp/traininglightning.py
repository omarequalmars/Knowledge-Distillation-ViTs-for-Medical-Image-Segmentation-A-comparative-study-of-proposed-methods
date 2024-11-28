import traininglightningfunc as tlf
import albumentations as A
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss, TverskyLoss, FocalLoss, JaccardLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
import pytorch_lightning as pl
import torch

if __name__ == "__main__":
    data_directory ="Datasets\Breast Ultrasound Dataset\Breast Clean Data"
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)    
    images,masks=tlf.prepare_data(data_directory)
    
    (train_images ,train_masks),(val_images,val_masks)=tlf.split_data(images,masks)

    transforms = A.Compose([
        A.GaussNoise(var_limit=(10,40)),
        A.Normalize(normalization='min_max'),
         A.ElasticTransform(alpha=10, sigma=2, p = 0.1),
         A.HorizontalFlip(),
         A.VerticalFlip(),
         A.RandomRotate90(),
         A.RandomResizedCrop(256, 512, 512)
        ]
     )

    train_dataset = tlf.SegmentationDataset(train_images ,train_masks ,transform=transforms)
    val_dataset = tlf.SegmentationDataset(val_images ,val_masks,transform=A.Normalize(normalization='min_max'))
    batch_size = 16
    train_loader = tlf.DataLoader(train_dataset ,batch_size=batch_size ,shuffle=True)
    val_loader = tlf.DataLoader(val_dataset ,batch_size=batch_size,shuffle=True)
    
    encoder = 'tu-mobilevit_xxs'
    h_params_Unet = {
        'encoder': encoder,
        'dropout': 0.5,
        'depth': 5,
        'Bnorm': True,
        'Attn': 'scse',
        'channels': (256, 128, 64, 32, 16),
        'lr_init': 1e-3,
        'weight_decay': 0,
        'temperature': 2,
        'Arch': "Unet",
        'alpha': 0.1,
        'pixelwise': True
    }

    h_params_DLV3plus = {
        'encoder': encoder,
        'dropout': 0.5,
        'depth': 5,
        'encoder_output_stride': 16,
        'channels':512,
        'decoder_atrous_rates':(16, 32, 128),
        'lr_init': 1e-3,
        'weight_decay': 0,
        'temperature': 2,
        'Arch': "DLV3+",
        'alpha': 0.1,
        'pixelwise': True
    }
    h_params_MANet = {
    'encoder': encoder,               # Backbone encoder
    'dropout': 0.5,                      # Dropout rate
    'depth': 5,                          # Number of encoder stages
    'Bnorm': True,                       # Use batch normalization in the decoder
    'channels': (256, 128, 64, 32, 16),      # Number of channels in each decoder layer
    'decoder_pab_channels':64,
    'lr_init': 1e-3,                     # Initial learning rate
    'weight_decay': 0,                   # Weight decay for regularization
    'temperature': 2,                     # Temperature for distillation
    'Arch': "MANet",                     # Architecture name
    'alpha': 0.1,                        # Weighting factor for loss components
    'pixelwise': True                    # Whether to use pixel-wise loss
}
    h_params_PAN = {
    'encoder': encoder,        # Backbone encoder
    'dropout': 0.5,                      # Dropout rate
    'encoder_output_stride': 16,
    'channels': 512,           # Number of channels in each decoder layer
    'lr_init': 1e-3,                     # Initial learning rate
    'weight_decay': 0,                   # Weight decay for regularization
    'temperature': 2,                     # Temperature for distillation
    'Arch': "PAN",                       # Architecture name
    'alpha': 0.1,                        # Weighting factor for loss components
    'pixelwise': True                    # Whether to use pixel-wise loss
}
    teacherpath=r'DeepLabV3Plus_Training\mysaves\resnet101_bestyet2\model_epoch_120'
    dicts = [h_params_DLV3plus, h_params_MANet, h_params_PAN, h_params_Unet]
    for hp in dicts:
        model = tlf.SegmentationModel(hp=h_params_PAN, teacherpath=teacherpath)

        early_stop_callback = EarlyStopping(
        monitor='val_loss',     # Metric to monitor
        min_delta=0.005,         # Minimum change to qualify as an improvement
        patience=100,             # How many epochs to wait for improvement
        verbose=True,           # Print messages when stopping
        mode='min'              # Mode can be 'min' or 'max'
    )
        
        lr_monitor = LearningRateMonitor(logging_interval = 'epoch')
        logger = TensorBoardLogger(r"KDExp\MobileViT\_VanillaKD\runs/", name=f"{encoder}_{hp['Arch']}")
        trainer = pl.Trainer(max_epochs=500, callbacks=[lr_monitor,early_stop_callback], logger = logger,
                            deterministic=False)  # 

        print("Starting training...")
            
        trainer.fit(model ,train_loader ,val_loader)
