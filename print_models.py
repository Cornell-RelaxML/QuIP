import torch
import timm
import os



os.makedirs(f'models/definitions/{class_name}/', exist_ok=True)
sizes = {}
for k,v in names.items():
    model = timm.create_model(v, pretrained=True)
    with open(f'models/definitions/{class_name}/{k}.model' ,'w') as h:
        print(model, file=h)
        num_params = sum([p.numel() for p in model.parameters()])/1e6
        print(k, num_params)
        print(k, num_params, file=h)
    # sizes[k] = num_params

# Deit
# names ={
# 'deit3_b224': "deit3_base_patch16_224.fb_in22k_ft_in1k",
# 'deit3_b384': "deit3_base_patch16_384.fb_in22k_ft_in1k",
# 'deit3_h224': "deit3_huge_patch14_224.fb_in22k_ft_in1k",
# 'deit3_l224': "deit3_large_patch16_224.fb_in22k_ft_in1k",
# 'deit3_l384': "deit3_large_patch16_384.fb_in22k_ft_in1k",
# 'deit3_m224': "deit3_medium_patch16_224.fb_in22k_ft_in1k",
# 'deit3_s224': "deit3_small_patch16_224.fb_in22k_ft_in1k",
# 'deit3_s384': "deit3_small_patch16_384.fb_in22k_ft_in1k",
# 'deit_b224_dist': "deit_base_distilled_patch16_224.fb_in1k",
# 'deit_b384_dist': "deit_base_distilled_patch16_384.fb_in1k",
# 'deit_b224': "deit_base_patch16_224.fb_in1k",
# 'deit_b384': "deit_base_patch16_384.fb_in1k",
# 'deit_s224_dist': "deit_small_distilled_patch16_224.fb_in1k",
# 'deit_s224': "deit_small_patch16_224.fb_in1k",
# 'deit_t224_dist': "deit_tiny_distilled_patch16_224.fb_in1k",
# 'deit_t224': "deit_tiny_patch16_224.fb_in1k",
# }
# class_name = 'DeiT'


# ViT
# names = {
# 'vit_b_p8ar2' : "vit_base_patch8_224.augreg2_in21k_ft_in1k",
# 'vit_b_p8ar1' : "vit_base_patch8_224.augreg_in21k_ft_in1k",
# 'vit_b_p16ar2' : "vit_base_patch16_224.augreg2_in21k_ft_in1k",
# 'vit_b_p16ar1' : "vit_base_patch16_224.augreg_in21k_ft_in1k",
# 'vit_b_p16ori' : "vit_base_patch16_224.orig_in21k_ft_in1k",
# 'vit_b_p16mii' : "vit_base_patch16_224_miil.in21k_ft_in1k",
# 'vit_b_p16_384ar' : "vit_base_patch16_384.augreg_in21k_ft_in1k",
# 'vit_b_p16_384ori' : "vit_base_patch16_384.orig_in21k_ft_in1k",
# 'vit_b_p16_clipl' : "vit_base_patch16_clip_224.laion2b_ft_in1k",
# 'vit_b_p16_clipo' : "vit_base_patch16_clip_224.openai_ft_in1k",
# 'vit_b_p16_clipl384' : "vit_base_patch16_clip_384.laion2b_ft_in1k",
# 'vit_b_p16_clipo384' : "vit_base_patch16_clip_384.openai_ft_in1k",
# 'vit_b_p32_ar' : "vit_base_patch32_224.augreg_in21k_ft_in1k",
# 'vit_b_p32_384' : "vit_base_patch32_384.augreg_in21k_ft_in1k",
# 'vit_b_p32_clipl' : "vit_base_patch32_clip_224.laion2b_ft_in1k",
# 'vit_b_p32_clipo' : "vit_base_patch32_clip_224.openai_ft_in1k",
# 'vit_b_r50_ori' : "vit_base_r50_s16_384.orig_in21k_ft_in1k",
# 'vit_h_p14_cli' : "vit_huge_patch14_clip_224.laion2b_ft_in1k",
# 'vit_h_p14_cli336' : "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k",
# 'vit_h_p14_gap' : "vit_huge_patch14_gap_224.in1k_ijepa",
# 'vit_h_p16_gap' : "vit_huge_patch16_gap_448.in1k_ijepa",
# 'vit_l_p14_clipl' : "vit_large_patch14_clip_224.laion2b_ft_in1k",
# 'vit_l_p14_clipo' : "vit_large_patch14_clip_224.openai_ft_in1k",
# 'vit_l_p14_clipl336' : "vit_large_patch14_clip_336.laion2b_ft_in1k",
# 'vit_l_p14_clipo336' : "vit_large_patch14_clip_336.openai_ft_in12k_in1k",
# 'vit_l_p16_ar' : "vit_large_patch16_224.augreg_in21k_ft_in1k",
# 'vit_l_p16_ar384' : "vit_large_patch16_384.augreg_in21k_ft_in1k",
# 'vit_l_p32_384_ori' : "vit_large_patch32_384.orig_in21k_ft_in1k",
# 'vit_l_r50_ar' : "vit_large_r50_s32_224.augreg_in21k_ft_in1k",
# 'vit_l_r50_ar384' : "vit_large_r50_s32_384.augreg_in21k_ft_in1k",
# 'vit_m' : "vit_medium_patch16_gap_256.sw_in12k_ft_in1k",
# 'vit_m_384' : "vit_medium_patch16_gap_384.sw_in12k_ft_in1k",
# 'vit_s_p16' : "vit_small_patch16_224.augreg_in21k_ft_in1k",
# 'vit_s_p16_384' : "vit_small_patch16_384.augreg_in21k_ft_in1k",
# 'vit_s_p32' : "vit_small_patch32_224.augreg_in21k_ft_in1k",
# 'vit_s_p32_384' : "vit_small_patch32_384.augreg_in21k_ft_in1k",
# 'vit_s_r26' : "vit_small_r26_s32_224.augreg_in21k_ft_in1k",
# 'vit_s_r26_384' : "vit_small_r26_s32_384.augreg_in21k_ft_in1k",
# 'vit_t_p16' : "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
# 'vit_t_p16_ar' : "vit_tiny_patch16_384.augreg_in21k_ft_in1k",
# 'vit_t_r' : "vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k",
# 'vit_t_r384' : "vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k",
# }
# class_name = 'vit'


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
# class_name = 'cait'

# names = {'poolform-m36': 'poolformer_m36.sail_in1k',
#         'poolform-m48': 'poolformer_m48.sail_in1k',
#         'poolform-s12': 'poolformer_s12.sail_in1k',
#         'poolform-s24': 'poolformer_s24.sail_in1k',
#         'poolform-s36': 'poolformer_s36.sail_in1k',
# }
# class_name = 'poolformer'
# names = {   'poolformv2-m36': 'poolformerv2_m36.sail_in1k',
#             'poolformv2-m48': 'poolformerv2_m48.sail_in1k',
#             'poolformv2-s12': 'poolformerv2_s12.sail_in1k',
#             'poolformv2-s24': 'poolformerv2_s24.sail_in1k',
#             'poolformv2-s36': 'poolformerv2_s36.sail_in1k',
# }
# class_name = 'poolformerV2'

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