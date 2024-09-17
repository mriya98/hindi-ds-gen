# Hindi Dataset Generation
Hindi Character dataset generation using Google Fonts. The aim is to generate a dataset which can be used for character recognition, text recognition, or other visual text tasks. Fonts that support Indian languages are downloaded from [Google Fonts](https://fonts.google.com/) and used to construct glyphs for each character. These fonts usually support a huge range of characters, and so we use Unicodes to extract only the useful ones to curate this dataset. Augmentation techniques are applied to introduce variance in glyphs and add some distortion.

## Install dependencies
Run the following command to install the dependencies
```bash
pip install -r requirements.txt
```
## Configure Directories
A default directory and font is provided in the _config.py_ file. This information can be updated in `GLYPH_DATASET_DIR` and `FONT_DIR` variable in _config.py_.

## Configure Unicodes
The _dataGeneration.py_ file has unicodes which are used to identify the characters we need in our dataset. This can be modified to include or exclude different sets. Currently, the unicode ranges include those of Devanagiri characters, math symbols, currency sumbols, and some special characters - comment or uncomment the ranges depending on what you want to generate. Latin alphabets, numerals, etc., can also be made part of the dataset by including their unicode ranges. Refer the Unicode documentation [here](https://www.unicode.org/charts/) for more information.

## Dataset Generation
Run the following command to generate data using font types stored in `./fonts` folder. To add more fonts, install the font family from Google Fonts, and their .ttf files to `./fonts` folder.
```bash
python generateFromFont.py
```

## Augmented Dataset and Notebook
The [glpyh_dataset_create_visualise.ipynb](https://github.com/mriya98/hindi-ds-gen/blob/main/glpyh_dataset_create_visualise.ipynb) notebook uses the same logic to create the dataset. A variety of augmentations are defined and applied randomly based on the probabilty assigned to them. A glimpse of the augmented images:
![glyph_batch_img](https://github.com/user-attachments/assets/8961a017-bd0e-41eb-91b4-ff4d59a0cf4b)

[TMNIST](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist) is also used to render glyphs to add more samples for each character. Code for this can be found in [glpyh_dataset_create_visualise.ipynb](https://github.com/mriya98/hindi-ds-gen/blob/main/glpyh_dataset_create_visualise.ipynb) notebook.

