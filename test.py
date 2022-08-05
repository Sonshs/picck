# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str, recalculate_spans

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            for key, input_value in input_data_item.items():
                # print("input_data_item.items: {}".format(input_data_item.items()))
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)

            # For easier debug.
            image_names = input_data_item["filenames"]
            values = list(input_data_item.values())
            keys = list(input_data_item.keys())
            # for i, value in enumerate(values):
            #     print("{}".format(keys[i]))
            print("text_length: {}".format(input_data_item["text_length"]))
            # print("image_names: {}".format(image_names))

            output = pick_model(**input_data_item)
            logits = output['logits']  # (B, N*T, out_dim)
            print("logits shape: {}".format(logits.shape))
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            transcripts = input_data_item['transcripts']
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            print("len best_paths: {}".format(len(best_paths)))
            # print("best_paths: {}".format(best_paths))
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)


            for decoded_tags, decoded_texts, image_index, transcript in zip(decoded_tags_list, decoded_texts_list, image_indexs, transcripts):
                # List[ Tuple[str, Tuple[int, int]] ]
                # spans = bio_tags_to_spans(decoded_tags, [])
                # spans = sorted(spans, key=lambda x: x[1][0])
                # print("spans: {}".format(spans))
                # print("spans: {}".format(spans))
                # print("transcript: {} - {}".format(len(transcript), transcript))
                # print("decoded_tags_list: {} - {}".format(len(transcript), decoded_tags_list))
                # print("decoded_texts_list: {} - {}".format(len(decoded_texts_list),decoded_texts_list))
                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.json')
                print(result_file)
                spans = recalculate_spans(transcript, decoded_tags)

    
                entities = []  # exists one to many case
                for key_n_score, range_tuple in spans:
                    entity_name, score = key_n_score
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                # result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.json')
                with result_file.open(mode='w') as f:
                    f.write('{\n')
                    if len(entities):
                        for item in entities[:-1]:
                            f.write("    \"{}\": \"{}\",\n".format(item['entity_name'], item['text']))
                        f.write("    \"{}\": \"{}\"\n".format(entities[-1]['entity_name'], entities[-1]['text']))
                    f.write('}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('-g', '--gpu', default=-1, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()
    main(args)

# PICK-pytorch/saved/models/PICK_Default/test_0323_115727/model_best.pth
""" 
python test.py --checkpoint saved/models/PICK_Default/model6/model_best.pth --boxes_transcripts data/debug_data/boxes_and_transcripts --images_path data/debug_data/images --output_folder saved/test_output/debug --gpu 1 --batch_size 2
python test.py --checkpoint saved/models/PICK_Default/model2/model_best.pth --boxes_transcripts data/cavet_pick_test/boxes_and_transcripts --images_path data/cavet_pick_test/images --output_folder saved/test_output/model2 --gpu 1 --batch_size 2
python test.py --checkpoint /storage/sontda/project/giayto/PICK-pytorch/saved/models/PICK_Default/model3/model_best.pth --boxes_transcripts /storage/sontda/project/giayto/mapping_ocr/data/cavet_icdar_v2/PICK_convert/test/cavet_test_v1/cavet_public_infer/boxes_and_transcripts --images_path /storage/sontda/project/giayto/mapping_ocr/data/cavet_icdar_v2/PICK_convert/test/cavet_test_v1/cavet_public_infer/images --output_folder saved/test_output/model3 --gpu 1 --batch_size 2
python test.py \
--checkpoint /storage/sontda/project/giayto/PICK-pytorch/saved/models/PICK_Default/model3/model_best.pth \
--boxes_transcripts /storage/sontda/project/giayto/mapping_ocr/data/cavet_icdar_v2/output_convert/boxes_and_transcripts \
--images_path /storage/sontda/project/giayto/mapping_ocr/data/cavet_icdar_v2/output_convert/images \
--output_folder saved/test_output/model3 --gpu 0 --batch_size 2

python test.py \
--checkpoint /storage/sontda/project/giayto/PICK-pytorch/saved/models/PICK_Default/model6/model_best.pth \
--boxes_transcripts /storage/sontda/project/giayto/PICK-pytorch/data/public_test_v2/public_test_100/public_test_100_pick_infer/boxes_and_transcripts \
--images_path /storage/sontda/project/giayto/PICK-pytorch/data/public_test_v2/public_test_100/public_test_100_pick_infer/images \
--output_folder saved/test_output/model6/pb_100 --gpu 0 --batch_size 2

python test.py \
--checkpoint /storage/sontda/project/giayto/PICK-pytorch/saved/models/PICK_Default/model_shk/model_best.pth \
--boxes_transcripts /storage/sontda/project/giayto/PICK-pytorch/data/shk/boxes_and_transcripts \
--images_path /storage/sontda/project/giayto/PICK-pytorch/data/shk/images \
--output_folder saved/test_output/model_shk --gpu 0 --batch_size 2


### single test
python test.py \
--checkpoint /storage/sontda/project/giayto/PICK-pytorch/saved/models/PICK_Default/model_shk/model_best.pth \
--boxes_transcripts /storage/sontda/project/giayto/PICK-pytorch/data/test_data_example/boxes_and_transcripts \
--images_path /storage/sontda/project/giayto/PICK-pytorch/data/test_data_example/images \
--output_folder saved/test_output/single_test --gpu 0 --batch_size 2
"""
