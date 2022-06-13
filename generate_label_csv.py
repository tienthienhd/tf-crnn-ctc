import csv
import glob
import os

import pandas as pd

image_dir = '/media/thiennt/projects/mine/cv_end_to_end/training/ocr/crnn_ctc/datasets/gplx/train'

filenames = os.listdir(image_dir)

filenames = sorted(filenames)
# labels = [f.split('.')[0].split('_')[-1] for f in filenames]
#
labels = [""] * len(filenames)

df = pd.DataFrame({
    "filename": filenames,
    "label": labels
})


df.to_csv(image_dir + "_.csv", quoting=csv.QUOTE_ALL, index=False)

