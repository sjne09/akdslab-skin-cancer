import os
import shutil

src = "/opt/gpudata/data2/renyu/skin_lesion_dataset/best_digital_pathology"
dest = "/opt/gpudata/skin-cancer/akdslab-skin-cancer/data/slides"
for dname, dirnames, fnames in os.walk(src):
    for fname in fnames:
        shutil.copy(os.path.join(dname, fname), os.path.join(dest, fname))
