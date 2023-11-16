import timm
import torch
import argparse
import imageNet_utils as datasets

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import datetime
from datetime import timedelta
from datautils import set_seed
from tqdm import tqdm

import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_thread',
                        help="name of the experiment")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--modelIndex', type=int, default=0,
                        help="index of the model name in this list")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size for validation")
    parser.add_argument('--model_name', type=str, default=None,
                        help="name of the model")
    parser.add_argument('--save_path', type=str, default=None,
                        help="path to save the results")
    parser.add_argument('--parent_dir', type=str, default=None,
                        help="path to save the results")
    parser.add_argument('--n_gpu', type=int, default=1,
                        help="path to save the results")
    parser.add_argument('--img_size', type=int, default=224,
                        help="image size")

    args = parser.parse_args()

    exp_name = args.exp_name # 'mlp_attn_quant_weiner_full'
    results_dir = "output_new"
    if exp_name != 'debug_thread':
        if args.save_path is None:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            directory_path = os.path.join(results_dir, f'{args.parent_dir}', f'{exp_name}_{current_datetime}')
            args.save_path = directory_path
        else:
            directory_path = args.save_path

        os.makedirs(directory_path, exist_ok=True)

        logging.basicConfig(filename= os.path.join(directory_path, 'log.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set seed
    set_seed(args)
    print(f'##########################################################')
    logging.info(f'##########################################################')
    print('args = ')
    logging.info('args = ')
    for k,v in vars(args).items():
        print(f'{k}: {v}')
        logging.info(f'{k}: {v}')
    print(f'##########################################################')
    logging.info(f'##########################################################')

    names = [
        "vit_tiny_patch16_224",
        "vit_small_patch32_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "vit_base_patch16_384",

        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
        "deit_base_patch16_384",

        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        "swin_base_patch4_window12_384",
        "vit_huge_patch14_clip_224.laion2b_ft_in1k",
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "beit_base_patch16_384.in22k_ft_in22k_in1k",
        "beit_large_patch16_512.in22k_ft_in22k_in1k",
        "beit_base_patch16_224.in22k_ft_in22k_in1k",
        "beit_large_patch16_224.in22k_ft_in22k_in1k",
        "poolformer_m36.sail_in1k",
    ]
    if args.model_name is None:
        name = names[args.modelIndex]
    else:
        name = args.model_name

    # get the model
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = timm.create_model(name, pretrained=True).to(device)
        num_params = sum([p.numel() for p in net.parameters()])
        print(f'num_params = {num_params/ 1e6}')
        logging.info(f'num_params = {num_params/ 1e6}')

        # load the imagenet dataset
        g=datasets.ViTImageNetLoaderGenerator('/data/harsha/quantization/imagenet2012','imagenet',args.batch_size,args.batch_size,16, kwargs={"model":net, "img_size":args.img_size})
        test_set = g.test_loader()

        # Now you can iterate through the validation loader to get batches of images and their corresponding labels
        all_labels = []
        all_preds = []
        num_examples = 0
        correct_counts = 0
        for batch_idx, (images, labels) in tqdm(enumerate(test_set)):
            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            preds = torch.argmax(logits, dim=-1)

            all_labels.append(labels.cpu().unbind())
            all_preds.append(preds.cpu().unbind())
            # all_preds.append(preds.cpu().item())
            num_examples += len(labels)
            correct_counts += sum(preds == labels)

        val_acc = correct_counts  / num_examples
        print(f'{name = }, {val_acc = }')
        logging.info(f'{name = }, {val_acc = }')

    if exp_name != 'debug_thread':
        torch.save(net.state_dict(), os.path.join(directory_path, 'model.pth'))

        import csv
        results  = [name, num_params/1e6, val_acc]
        csv_file_path = os.path.join(results_dir, f'{args.parent_dir}', 'results.csv')
        with open(csv_file_path, mode='a', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(results)