import csv
import glob
import os
import re

import pandas as pd

image_dir = '/media/thiennt/data/backup_old_ubuntu/data_id/projects/mine/cv_end_to_end/training/ocr/crnn_ctc/datasets/vietcombank2/image'

filenames = os.listdir(image_dir)

filenames = sorted(filenames)
labels = [f.split('.')[0].split('_')[-1] for f in filenames]
labels = [re.sub(r"\D", "", l) for l in labels]
labels = []
#
# labels = [""] * len(filenames)

df = pd.DataFrame({
    "filename": filenames,
    "label": labels
})


df.to_csv(image_dir + "_.csv", quoting=csv.QUOTE_ALL, index=False)

