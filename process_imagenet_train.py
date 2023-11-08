import os
from tqdm import tqdm


tars_folder = '/data/harsha/quantization/imagenet2012'
train_folder = '/data/harsha/quantization/imagenet2012/train'

files = os.listdir(tars_folder)
for file in tqdm(files):
    if 'tar' in file:
        name, ext = file.split('.')
        dir_path = os.path.join(train_folder, name)
        if os.path.exists(dir_path):
            print(f'{dir_path} already exists. Skippin...')
            continue
        else:
            # make the class directory
            os.makedirs(dir_path)
            # extract files to this directory
            tar_path = os.path.join(tars_folder, file)
            command = f'tar -xf {tar_path} -C {dir_path}'
            print(command)
            exit_code = os.system(command)

