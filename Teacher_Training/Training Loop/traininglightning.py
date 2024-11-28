import traininglightningfunc as tlf
import albumentations as A
import segmentation_models_pytorch as smp
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
    
    (train_images ,train_masks),(val_images,val_masks),(test_images,test_masks)=tlf.split_data(images,masks)

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
    batch_size = 20
    train_loader = tlf.DataLoader(train_dataset ,batch_size=batch_size ,shuffle=True)
    val_loader = tlf.DataLoader(val_dataset ,batch_size=batch_size)
    
    encoder = 'resnet101'

    h_params_DLV3plus = {
        'encoder': encoder,
        'dropout': 0.5,
        'depth': 5,
        'encoder_output_stride': 16,
        'decoder_channels':512,
        'decoder_atrous_rates':(16, 32, 128),
        'lr_init': 1e-3,
        'weight_decay': 1
    }

    model = tlf.SegmentationModel(hp=h_params_DLV3plus)
    
    dir_weights = f'DeepLabV3Plus_Training\mysaves/{encoder}_bestyet2'
    model.model = smp.from_pretrained(r'DeepLabV3Plus_Training\mysaves\resnet101_bestyet\model_epoch_160')
    cp_cb = tlf.SaveModelEveryNepochs(dir_weights, 20)
    early_stop_callback = EarlyStopping(
    monitor='val_loss',     # Metric to monitor
    min_delta=0.001,         # Minimum change to qualify as an improvement
    patience=50,             # How many epochs to wait for improvement
    verbose=True,           # Print messages when stopping
    mode='min'              # Mode can be 'min' or 'max'
)
    

    lr_monitor = LearningRateMonitor(logging_interval = 'epoch')
    logger = TensorBoardLogger(r"DeepLabV3Plus_Training\runs\experiment_diceloss_bigconvs/", name=encoder, log_graph=True)
    trainer = pl.Trainer(max_epochs=200, callbacks=[lr_monitor,cp_cb, early_stop_callback], logger = logger,
                         deterministic=False)  # 

    print("Starting training...")
    
    trainer.fit(model ,train_loader ,val_loader)
    print("Test data saved to 'test_images.npy' and 'test_masks.npy'.")