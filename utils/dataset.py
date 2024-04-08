# IMPORTS
import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

# this dict helps tackle issues with inconsistent naming for these examples across data files
replace_name_dict = {
    'Excellent urethral plate (Intact prepuce circumcised)': 'Excellent urethral plate',
    'Flat plate but good spongionsum (TIP)': 'Flat plate but good spongionsum',
    'Proximal hypospadias (good)': 'Proximal hypospadias',
    'Photo 2016-08-16, 8 41 14 AM (1)': 'Photo 2016-08-16, 8 41 14 AM',
    'Photo 2016-11-13, 2 59 46 PM (1)': 'Photo 2016-11-13, 2 59 46 PM'
}

class HypospadiasDataset(Dataset):
    def __init__(self, config, score_df, preprocessing, dataset_type='train', include_masks=True) -> None:
        super().__init__()
        self.config = config
        
        self.image_dir = fr'{config["data_path"]}/{dataset_type}/images'
        self.mask_dir = fr'{config["data_path"]}/{dataset_type}/masks'
        self.images = os.listdir(self.image_dir)
        
        self.score_df = score_df
        self.preprocessing = preprocessing
        
        self.include_masks=include_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Retrieve image path and its name + extention
        img_path = os.path.join(self.image_dir, self.images[index])
        file_extention = Path(img_path).suffix
        img_name = Path(img_path).stem

        # print image name (debug purposes)
        print(img_name)

        # Open the image in RGB mode
        img = np.asarray(ImageOps.exif_transpose(Image.open(img_path)).convert("RGB"))

        if self.config["debug"]:
            print(f'[1] img shape: {img.shape}')

        # Get the image label score
        score_name = img_name
        if img_name in replace_name_dict.keys():
            score_name = replace_name_dict[img_name]
        
        # Get GMS score labels
        labels = []
        if self.config['gms_classification_heads']:
            for bodypart in self.config['anatomy_part']:
                s = self.score_df.stack().str.contains(score_name, na=False)
                img_row = self.score_df.iloc[[s[s].index[0][0]]]

                score = img_row[f'{bodypart} Concensus'].item()
                if pd.isna(score):
                    score = img_row[f'{bodypart} 1'].item()

                score = int(score)
                img_label = score - 1

                labels.append(img_label)
        
        
        # Get HOPE score labels
        hope_labels = []
        if self.config['hope_classification_heads']:
            for hope_component in self.config['hope_components']:
                s = self.score_df.stack().str.contains(score_name, na=False)
                img_row = self.score_df.iloc[[s[s].index[0][0]]]

                score = img_row[f'{hope_component} Concensus'].item()
                if pd.isna(score) or isinstance(score, str):
                    score = img_row[f'{hope_component} 2'].item()

                score = int(score)
                
                if hope_component == 'Torsion':
                    img_label = score
                else:
                    img_label = score - 1

                hope_labels.append(img_label)   

        # Get the score part segmentation mask; read mask if available, else create
        # a mask with all zeros
        if self.include_masks:
            masks = []
            for bodypart in (self.config['anatomy_part'] + self.config['overlap_bodyparts']):
                if bodypart == "Urethral Plate":
                    bodypart = "UrethralPlate"
                
                mask_path = os.path.join(self.mask_dir, img_name)
                mask_path = [n for n in glob(f'{mask_path}*{bodypart.lower()}.*')]

                assert len(mask_path) <= 1

                if len(mask_path) == 1:
                    # The '.convert("L") makes the mask into a grayscale image (i.e. 1 channel)
                    mask = np.array(ImageOps.exif_transpose(Image.open(mask_path[0])).convert("L"))

                    # This ensures that the mask range [0, 255] becomes {0, 1}
                    mask[mask > 0] = 1
                    
                else:
                    mask = np.zeros((img.shape[0], img.shape[1]))

                assert torch.all(torch.isin(torch.tensor(mask), torch.tensor([0, 1]))) == True

                masks.append(mask)
            

        # apply preprocessing transform to image and mask
        if self.preprocessing:
            if self.config["debug"]:
                print('Applying preprocessing...')

            if self.include_masks:
                transformed = self.preprocessing(image=img, masks=masks)
                img = transformed['image']
                masks = transformed['masks']
            else:
                transformed = self.preprocessing(image=img)
                img = transformed['image']

        # return values depend on whether to include masks or not
        if self.include_masks:
            masks = torch.from_numpy(np.stack(masks, axis=2)).permute(2, 0, 1).byte()

            if self.config["debug"]:
                print(f'[2] img shape: {img.shape}')
                print(f'[2] mask shape: {mask.shape}')

            if self.config['gms_classification_heads'] and not self.config['hope_classification_heads']:
                # return img_name, img (X), img_label (y1), and mask
                return img_name, img, labels, masks
            
            elif not self.config['gms_classification_heads'] and self.config['hope_classification_heads']:
                # return img_name, img (X), img_hope_label(y2), and mask
                return img_name, img, hope_labels, masks
            
            else:
                # return img_name, img (X), img_label (y1), img_hope_label(y2), and mask
                return img_name, img, labels, hope_labels, masks

        else:
            if self.config["debug"]:
                print(f'[2] img shape: {img.shape}')
            
            if self.config['gms_classification_heads'] and not self.config['hope_classification_heads']:
                # return img_name, img (X), img_label (y1), and mask
                return img_name, img, labels
            elif not self.config['gms_classification_heads'] and self.config['hope_classification_heads']:
                # return img_name, img (X), img_hope_label(y2), and mask
                return img_name, img, hope_labels
            else:
                # return img_name, img (X), img_label (y1), img_hope_label(y2), and mask
                return img_name, img, labels, hope_labels
