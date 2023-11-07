import os

image_folder = '/data/harsha/quantization/imagenet2012/val'
val_labels_file = '/data/harsha/quantization/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

files = sorted(os.listdir(image_folder))

with open(val_labels_file, 'r') as f:
    label_indices = f.readlines()

# Loop through the files
files = [f + ' ' + l.strip() for f,l in zip(files, label_indices)] 
for file in files:
    print(file)

imagenet_val_labels_file = '/data/harsha/quantization/imagenet2012/val_labels.txt'

with open(imagenet_val_labels_file, 'w') as f:
    for line in files:
        f.write(line)
        f.write('\n')

