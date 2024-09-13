# hindi Dataset Generation
Hindi Characters dataset generation using Google Fonts. The python helps to create your own dataset by using different fonts
and Unicodes of characters. 

## Install dependencies
```bash
pip install -r requirements.txt
```
## Configure Directories
`config.py` can be used to configure your own directories for storing generated images, and also for updating location of fonts
directory.

## Configure Unicodes
The `dataGeneration.py` file has unicodes which are used to identify the characters we need in our dataset. This can be modified
to include or exclude different sets. Currently, the unicode ranges include those of Devanagiri characters, math symbols,
currency sumbols, and some special characters - comment or uncomment the ranges depending on what you want to generate.
Latin alphabetsl, numerals, etc., can also be maed part of the dataset by including their unicode ranges.

## Dataset Generation
Run the following command to generate data using font types stored in `./fonts` folder. More fonts can be installed from Google Fonts,
and their .ttf files should be added to `./fonts` folder to included in data generation process.
```bash
python generateFromFont.py
```
