import torch
import timm

names = {
    'beit-b384': 'beit_base_patch16_384.in22k_ft_in22k_in1k',
    'beit-l512': 'beit_large_patch16_512.in22k_ft_in22k_in1k'}

sizes = {}
for k,v in names.items():
    model = timm.create_model(v, pretrained=True)
    with open(f'models/definitions/{k}.model' ,'w') as h:
        print(model, file=h)
        num_params = sum([p.numel() for p in model.parameters()])/1e6
        print(k, num_params)
        print(k, num_params, file=h)
    # sizes[k] = num_params


# for k,v in sizes.items():
#     print(k,v)

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

# names = {'vit-h': 'vit_huge_patch14_clip_224.laion2b_ft_in1k', 
#          'vit-l': 'vit_large_patch16_224.augreg_in21k_ft_in1k'}

# names = {   'coatnet-2': 'coatnet_2_rw_224.sw_in12k_ft_in1k',
#             'coatnet-rmlp1': 'coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k',
#             'coatnet-rmlp2': 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k',
#             'coatnet-rmlp2_384': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'}

# names = {   'beit-b224': 'beit_base_patch16_224.in22k_ft_in22k_in1k',
#             'beit-b384': 'beit_base_patch16_384.in22k_ft_in22k_in1k',
#             'beit-l224': 'beit_large_patch16_224.in22k_ft_in22k_in1k',
#             'beit-l384': 'beit_large_patch16_384.in22k_ft_in22k_in1k',
#             'beit-l512': 'beit_large_patch16_512.in22k_ft_in22k_in1k'}

# names ={    'cait-m36': 'cait_m36_384.fb_dist_in1k',
#             'cait-m48': 'cait_m48_448.fb_dist_in1k',
#             'cait-s24_224': 'cait_s24_224.fb_dist_in1k',
#             'cait-s24_384': 'cait_s24_384.fb_dist_in1k',
#             'cait-s36_384': 'cait_s36_384.fb_dist_in1k',
#             'cait-xs24_384': 'cait_xs24_384.fb_dist_in1k',
#             'cait-xs24_224': 'cait_xxs24_224.fb_dist_in1k',
#             'cait-xxs24_384': 'cait_xxs24_384.fb_dist_in1k',
#             'cait-xxs36_224': 'cait_xxs36_224.fb_dist_in1k',
#             'cait-xxs36_384': 'cait_xxs36_384.fb_dist_in1k'
# }

# names = {
# 'pvt_v2_b0': 'pvt_v2_b0.in1k',
# 'pvt_v2_b1': 'pvt_v2_b1.in1k',
# 'pvt_v2_b2': 'pvt_v2_b2.in1k',
# 'pvt_v2_b2': 'pvt_v2_b2_li.in1k',
# 'pvt_v2_b3': 'pvt_v2_b3.in1k',
# 'pvt_v2_b4': 'pvt_v2_b4.in1k',
# 'pvt_v2_b5': 'pvt_v2_b5.in1k',
# }

# names = {
# 'levit-128': 'levit_128.fb_dist_in1k',
# 'levit-128s': 'levit_128s.fb_dist_in1k',
# 'levit-192': 'levit_192.fb_dist_in1k',
# 'levit-256': 'levit_256.fb_dist_in1k',
# 'levit-384': 'levit_384.fb_dist_in1k',
# 'levit-conv-128': 'levit_conv_128.fb_dist_in1k',
# 'levit-conv-128s': 'levit_conv_128s.fb_dist_in1k',
# 'levit-conv-192': 'levit_conv_192.fb_dist_in1k',
# 'levit-conv-256': 'levit_conv_256.fb_dist_in1k',
# 'levit-conv-384': 'levit_conv_384.fb_dist_in1k'
# }

# names = {
# 'pit_b': 'pit_b_224.in1k',
# 'pit_b_dist': 'pit_b_distilled_224.in1k',
# 'pit_s': 'pit_s_224.in1k',
# 'pit_s_dist': 'pit_s_distilled_224.in1k',
# 'pit_ti': 'pit_ti_224.in1k',
# 'pit_ti_dist': 'pit_ti_distilled_224.in1k',
# 'pit_xs': 'pit_xs_224.in1k',
# 'pit_xs_dist': 'pit_xs_distilled_224.in1k',
# }