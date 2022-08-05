# -*- coding: utf-8 -*-

from typing import *
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from data_utils.documents import MAX_TRANSCRIPT_LEN

import torch

from .class_utils import keys_vocab_cls, iob_labels_vocab_cls
from data_utils import documents


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def iob2entity(tag):
    '''
    iob label to entity
    :param tag:
    :return:
    '''
    if len(tag) == 1 and tag != 'O':
        raise TypeError('Invalid tag!')
    elif len(tag) == 1 and tag == 'O':
        return tag
    elif len(tag) > 1:
        e = tag[2:]
        return e


def iob_index_to_str(tags: List[List[int]]):
    decoded_tags_list = []
    # print("\n --- debug iob_index_to_str -----")
    # print("tags: {}".format(tags))
    # print("tag_len: {}".format(len(tags[0])))
    for doc in tags:
        # print("doc_len: {}".format(len(doc)))
        # print("   doc in tags: {}".format(doc))
        decoded_tags = []
        for i, tag in enumerate(doc):
            # print("     tag in doc: {}".format(tag))
            s = iob_labels_vocab_cls.itos[tag]
            # print("{} - {}".format(i, s))
            # print("       s before: {}".format(s))
            if s == '<unk>' or s == '<pad>':
                s = ' '

            # print("       s after : {}".format(s))
            decoded_tags.append(s)
        decoded_tags_list.append(decoded_tags)
    # print("decoded_tags_list in_funct: {}".format(decoded_tags_list))
    return decoded_tags_list


def text_index_to_str(texts: torch.Tensor, mask: torch.Tensor):
    # union_texts: (B, num_boxes * T)
    union_texts = texts_to_union_texts(texts, mask)
    # print("texts: {}".format(texts))
    # print("union_texts: {}".format(union_texts))
    B, NT = union_texts.shape

    decoded_tags_list = []
    for i in range(B):
        decoded_text = []
        for text_index in union_texts[i]:
            text_str = keys_vocab_cls.itos[text_index]
            # print("text_str: {}".format(text_str))
            if text_str == '<unk>' or text_str == '<pad>':
                text_str = ' '
            decoded_text.append(text_str)
        decoded_tags_list.append(decoded_text)
    # print("decoded_tags_list: {}".format(decoded_tags_list))
    return decoded_tags_list


def texts_to_union_texts(texts, mask):
    '''

    :param texts: (B, N, T)
    :param mask: (B, N, T)
    :return:
    '''

    B, N, T = texts.shape

    texts = texts.reshape(B, N * T)
    mask = mask.reshape(B, N * T)

    # union tags as a whole sequence, (B, N*T)
    union_texts = torch.full_like(texts, keys_vocab_cls['<pad>'], device=texts.device)

    max_seq_length = 0
    for i in range(B):
        valid_text = torch.masked_select(texts[i], mask[i].bool())
        valid_length = valid_text.size(0)
        union_texts[i, :valid_length] = valid_text

        if valid_length > max_seq_length:
            max_seq_length = valid_length

    # max_seq_length = documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN
    # (B, N*T)
    union_texts = union_texts[:, :max_seq_length]

    # (B, N*T)
    # print("union_texts: {}".format(union_texts))
    return union_texts


def iob_tags_to_union_iob_tags(iob_tags, mask):
    '''

    :param iob_tags: (B, N, T)
    :param mask: (B, N, T)
    :return:
    '''

    B, N, T = iob_tags.shape

    iob_tags = iob_tags.reshape(B, N * T)
    mask = mask.reshape(B, N * T)

    # union tags as a whole sequence, (B, N*T)
    union_iob_tags = torch.full_like(iob_tags, iob_labels_vocab_cls['<pad>'], device=iob_tags.device)

    max_seq_length = 0
    for i in range(B):
        valid_tag = torch.masked_select(iob_tags[i], mask[i].bool())
        valid_length = valid_tag.size(0)
        union_iob_tags[i, :valid_length] = valid_tag

        if valid_length > max_seq_length:
            max_seq_length = valid_length

    # max_seq_length = documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN
    # (B, N*T)
    union_iob_tags = union_iob_tags[:, :max_seq_length]

    # (B, N*T)
    # print("union_iob_tags: {}".format(union_iob_tags))
    return union_iob_tags

# def recalculate_spans(transcript, decoded_tags):
#     transcripts_len = [len(f) for f in transcript]
#     # concat_decoded_tags = []
#     # for i in decoded_tags:
#     #     concat_decoded_tags += i
#     # print("concat_decoded_tags len: {}".format(len(concat_decoded_tags[0])))
#     # print("sum transcripts_len: {}".format(sum(transcripts_len)))
#     rc_spans = []
#     count = 0
#     for ele in transcripts_len:
#         single_dict = {}
#         start_idx = count
#         for i in range(ele):
#             # try:
#             ## tab
#             # print(count)
#             # print("concat_decoded_tags: {}".format(concat_decoded_tags[0][count]))
#             cur_char = concat_decoded_tags[count]
#             if cur_char not in ['O', '<unk>', '<pad>']:
#                 class_tag = cur_char.split('-')[-1]
#                 if class_tag not in single_dict and class_tag not in []:
#                     single_dict[class_tag] = 1
#                 else:
#                     single_dict[class_tag] += 1
#             count += 1
#             ## end-tab
#             # except Exception as e:
#             #     # print("concat_decoded_tags: {}".format(concat_decoded_tags))
#             #     print(e)
#         end_idx = count-1
        
#         if single_dict:
#             main_key = max(single_dict, key=single_dict.get)
            
#             span_tpl = (main_key, (start_idx, end_idx))
#             rc_spans.append(span_tpl)
#     return rc_spans

def recalculate_spans(transcript, decoded_tags):
    # for i, line in enumerate(transcript):
    #     transcript[i] = line.replace("'"," ").replace("):","")
    transcripts_len = [len(f) for f in transcript]
    for i, transcript_len in enumerate(transcripts_len):
        if transcript_len > MAX_TRANSCRIPT_LEN:
            transcripts_len[i] = MAX_TRANSCRIPT_LEN
    rc_spans = []
    count = 0
    print("len decoded_tags: {}".format(len(decoded_tags)))
    # print("len decoded_texts: {}".format(len(decoded_texts)))
    print("len transcripts: {} - {}".format(transcripts_len, sum(transcripts_len)))


    # print("decoded_tags: {}".format(decoded_tags))
    # print("transcripts_len: {}".format(transcripts_len))
    # print("decoded_texts: \n{}".format(''.join(decoded_texts)))
    print("transcript: \n{}".format(''.join(transcript)))

    for f_idx, ele in enumerate(transcripts_len):
        single_dict = {}
        start_idx = count
        for i in range(ele):
            cur_char = decoded_tags[count]
            if cur_char not in ['O', '<unk>', '<pad>']:
                class_tag = cur_char.split('-')[-1]
                if class_tag not in single_dict:
                    single_dict[class_tag] = 1
                else:
                    single_dict[class_tag] += 1
            count += 1
        end_idx = count-1
        
        if single_dict:
            # we need a score of main_key to compare with duplicate result later
            main_key = max(single_dict, key=single_dict.get)
            key_n_score = (main_key, single_dict[main_key])
            span_tpl = (key_n_score, (start_idx, end_idx))
            rc_spans.append(span_tpl)
    # print("rc_spans: {}".format(rc_spans))

    # handling duplicate keys 
    dict_t = {}
    for ele in rc_spans:
        key_n_score, coor = ele
        key, score = key_n_score
        if key in dict_t:
            if score > dict_t[key][0]:
                dict_t[key] = [score, coor]
        else:
            dict_t[key] = [score, coor]

    final_spans = []
    for key in dict_t:
        row = ((key, dict_t[key][0]), dict_t[key][1])
        final_spans.append(row)

    return final_spans