import torch
import timm

names = {'vit-t': 'vit_tiny_patch16_224',
         'vit-s': 'vit_small_patch16_224',
         'vit-s32': 'vit_small_patch32_224',
         'vit-b': 'vit_base_patch16_224',
         'deit-t': 'deit_tiny_patch16_224',
         'deit-s': 'deit_small_patch16_224',
         'deit-b': 'deit_base_patch16_224'}

for k,v in names.items():
    model = timm.create_model(v, pretrained=True)
    with open(f'models/definitions/{k}.model' ,'w') as h:
        print(model, file=h)