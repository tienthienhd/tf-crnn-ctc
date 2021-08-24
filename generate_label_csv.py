import csv
import glob
import os

import pandas as pd

image_dir = '/media/data_it/Data_set/database_image/ocr/captcha/garena'

filenames = os.listdir(image_dir)

filenames = sorted(filenames)

labels = [""] * len(filenames)

df = pd.DataFrame({
    "filename": filenames,
    "label": labels
})


df.to_csv(image_dir + "_.csv", quoting=csv.QUOTE_ALL, index=False)

