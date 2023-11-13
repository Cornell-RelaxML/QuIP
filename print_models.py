import torch
import timm

# names = {'vit-t': 'vit_tiny_patch16_224',
#          'vit-s': 'vit_small_patch16_224',
#          'vit-s32': 'vit_small_patch32_224',
#          'vit-b': 'vit_base_patch16_224',
#          'deit-t': 'deit_tiny_patch16_224',
#          'deit-s': 'deit_small_patch16_224',
#          'deit-b': 'deit_base_patch16_224', 
#          }

# names = {'swin-s': 'swin_small_patch4_window7_224',
#          'swin-b': 'swin_base_patch4_window7_224',
#          'swin-b384': 'swin_base_patch4_window12_384', 
#          }
names = {'vit-h': 'vit_huge_patch14_clip_224.laion2b_ft_in1k', 
         'vit-l': 'vit_large_patch16_224.augreg_in21k_ft_in1k'}

for k,v in names.items():
    model = timm.create_model(v, pretrained=True)
    with open(f'models/definitions/{k}.model' ,'w') as h:
        print(model, file=h)
        print(k, sum([p.numel() for p in model.parameters()]))