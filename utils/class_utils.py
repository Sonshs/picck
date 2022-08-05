# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:26 PM


from collections import Counter
from torchtext.vocab import Vocab
from pathlib import Path

from . import entities_list


class ClassVocab(Vocab):

    def __init__(self, classes, specials=['<pad>', '<unk>'], **kwargs):
        '''
        convert key to index(stoi), and get key string by index(itos)
        :param classes: list or str, key string or entity list
        :param specials: list, special tokens except <unk> (default: {['<pad>', '<unk>']})
        :param kwargs:
        '''
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            ### old method: instance is CHAR
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)

            # #### new method: instance is WORD
            # with p.open(encoding='utf8') as f:
            #     classes = f.read()
            #     cls_list = classes.split('\n')
        elif isinstance(classes, list):
            cls_list = classes
        c = Counter(cls_list)
        self.special_count = len(specials)
        super().__init__(c, specials=specials, **kwargs)


def entities2iob_labels(entities: list):
    '''
    get all iob string label by entities
    :param entities:
    :return:
    '''
    # print("\n---- debug entities2iob_labels ----")
    # print("entities: {}".format(entities))
    tags = []
    for e in entities:
        # print("   e: {}".format(e))
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    # print("final tag: {}".format(tags))
    # print("--- end debug ----")
    return tags

### old dict: "keys.txt"
keys_vocab_cls = ClassVocab(Path(__file__).parent.joinpath('keys.txt'), specials_first=False)

# ### new dict: "w_keys.txt" - seperate by "\n" 
# keys_vocab_cls = ClassVocab(Path(__file__).parent.joinpath('w_keys.txt'), specials_first=False)


iob_labels_vocab_cls = ClassVocab(entities2iob_labels(entities_list.Entities_list), specials_first=False)
entities_vocab_cls = ClassVocab(entities_list.Entities_list, specials_first=False)

# print("iob all ele: {}".format(iob_labels_vocab_cls.itos))
# print("e2iobL {}".format(entities2iob_labels(entities_list.Entities_list)))
# print("entities_list {}".format(entities_list.Entities_list))
