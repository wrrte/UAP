import sys, os
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
# from image_dataset import create_dataset
from utils import load_model, WrapperModel
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def evaluate(test_dataset, test_dataloader, uap):
    all_model_names = ['ResNet18', 'ResNet50', 'inception_v3_timm', 'inception_v4_timm',
                       'DenseNet121', 'adv_inception_v3']

    source_model_names = ['ResNet50']
    # source_model_names = source_model_names[args.start:args.end]
    transfer_model_names = [x for x in all_model_names]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pre-process input image
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transfer_models = [load_model(x).to(device) for x in transfer_model_names]
    transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(), lambda x:(x-0.5)*2])

    # Create Dataset
    # dataset = create_dataset(transform, num_images=args.num_images)


    # batch_size = args.batch_size
    for model_i, source_model_name in enumerate(source_model_names):
        print(source_model_name)
        torch.cuda.empty_cache()

        # load models
        source_model = load_model(source_model_name).to(device)
        source_model = source_model.eval()

        # Attack and Save images
        attack_methods = ["MNI-DTS"]

        def iter_source():
            # print("batch: ", batch_size)
            dl = test_dataloader
            target_accs = {m: {k: 0. for k in attack_methods} for m in transfer_model_names}
            num_images = 0
            for i, data in enumerate(dl):
                num_images += data[1].shape[0]

                img, label = data
                img = img.to(device)
                label = label.to(device)

                source_model.eval()
                output_dict = {method: normalize((img+uap).clamp(0,1)) for method in
                               attack_methods}

                # get labels for transfer
                for j, mod in enumerate(transfer_models):
                    mod.eval()
                    with torch.no_grad():
                        transfer_results_dict = {key: F.softmax(mod(value), dim=1).max(dim=1) for key, value in
                                                 output_dict.items()}
                    for a in attack_methods:
                        target_accs[transfer_model_names[j]][a] += (
                            torch.sum((transfer_results_dict[a][1] == label).float())).item()
                        acc = (target_accs[transfer_model_names[j]][a]) / num_images
                        print(f'[{i + 1}/{len(dl)}] SR (%) on {transfer_model_names[j]} with {a} : '
                              f'{100 - acc * 100:.2f}')
            return target_accs

        target_accs = iter_source()

        # final accuracy and scoring
        sr_values = {}
        for attack_name in attack_methods:
            for j, mod in enumerate(transfer_models):
                model_name = transfer_model_names[j]
                acc = (target_accs[model_name][attack_name]) / len(test_dataset)
                sr = 100.0 - acc * 100.0
                if model_name not in sr_values:
                    sr_values[model_name] = {}
                sr_values[model_name][attack_name] = sr
                print(f'Final SR (%) on {model_name} with {attack_name} : {sr:.2f}')

        robust_model_names = {
            "DenseNet121",
            "inception_v4_timm",
            "adv_inception_v3",
        }

        white_box_srs = []
        if source_model_name in sr_values:
            for attack_name in attack_methods:
                white_box_srs.append(sr_values[source_model_name][attack_name])
            white_box_score = sum(white_box_srs) / float(len(white_box_srs))
        else:
            white_box_score = 0.0

        weighted_sum = 0.0
        weight_total = 0.0
        for attack_name in attack_methods:
            for model_name in transfer_model_names:
                if model_name == source_model_name:
                    continue
                sr = sr_values[model_name][attack_name]
                weight = 2.0 if model_name in robust_model_names else 1.0
                weighted_sum += sr * weight
                weight_total += weight

        if weight_total > 0.0:
            weighted_transfer_score = weighted_sum / weight_total
        else:
            weighted_transfer_score = 0.0

        alpha = 0.4
        beta = 0.6
        final_score = alpha * white_box_score + beta * weighted_transfer_score

        print(f"White-box SR score on {source_model_name} (avg over attacks): {white_box_score:.2f}")
        print(f"Transfer SR score (weighted by robustness): {weighted_transfer_score:.2f}")
        print(f"final_score : {final_score:.2f}")

        # store per-source final score for later overall averaging
        if "per_source_final_scores" not in globals():
            global per_source_final_scores  # global dict to collect scores
            per_source_final_scores = {}
        per_source_final_scores[source_model_name] = final_score  # save score for this source

    # # compute eval_final_score after processing all source models (added)
    # if "per_source_final_scores" in globals() and len(per_source_final_scores) > 0:
    #     scores = list(per_source_final_scores.values())
    #     eval_final_score = sum(scores) / float(len(scores))
    #     print("eval_final_score : {:.2f}".format(eval_final_score))


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="batch_size as an integer")
    parser.add_argument("--num_images", default=1000, type=int, help="number of images to use")
    parser.add_argument("--start", default=0, type=int, help="source model index start")
    parser.add_argument("--end", default=9, type=int, help="source model index end")
    parser.add_argument("--save_images", default=False, help="T/F")
    parser.add_argument("--excel_path", default="auto_results_seo.xlsx", help="auto_results.xlsx")
    return parser


# if __name__ == "__main__":
#     args = argument_parsing().parse_args()
#     main(args)
