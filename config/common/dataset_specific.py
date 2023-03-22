from geoseg.datasets.potsdam_dataset import *
import PIL

def get_training_transform():
    train_transform = [
    #    albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=375, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img = img.resize(ORIGIN_IMG_SIZE)
    mask = mask.resize(ORIGIN_IMG_SIZE)
    img, mask = np.array(img), np.array(mask)
    if np.isin(7, mask):
        print("Found 7 - fixing")
        mask[mask == 7] = 6
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def val_aug(img, mask):
    img = img.resize(ORIGIN_IMG_SIZE)
    mask = mask.resize(ORIGIN_IMG_SIZE)
    img, mask = np.array(img), np.array(mask)
    if np.isin(7, mask):
        print("Found 7 - fixing")
        mask[mask == 7] = 6
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def ade20k_train_aug(img, mask):
    #crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
    #                    SmartCropV1(crop_size=375, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    #img, mask = crop_aug(img, mask)
    img = img.resize(ORIGIN_IMG_SIZE, resample=PIL.Image.NEAREST)
    mask = mask.resize(ORIGIN_IMG_SIZE, resample=PIL.Image.NEAREST)
    img, mask = np.array(img), np.array(mask)
    
    #if np.isin(150, mask):
    #    mask[mask == 150] = 0
    #if np.isin(0, mask):
    #    mask[mask == 0] = 150
        
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def ade20k_val_aug(img, mask):
    img = img.resize(ORIGIN_IMG_SIZE, resample=PIL.Image.NEAREST)
    mask = mask.resize(ORIGIN_IMG_SIZE, resample=PIL.Image.NEAREST)
    img, mask = np.array(img), np.array(mask)
    
    #if np.isin(150, mask):
    #    mask[mask == 150] = 0
    #if np.isin(0, mask):
    #    mask[mask == 0] = 150
        
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask