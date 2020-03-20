import zipfile
with zipfile.ZipFile("GAN/real/landscape-pictures.zip", 'r') as zip_ref:
    zip_ref.extractall("GAN/real/real_landscapes")