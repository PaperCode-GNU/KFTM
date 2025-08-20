import json
import argparse
import numpy as np
import pickle as pk



from data_util import seg_sentence, get_cutter

from keras.models import load_model
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array, sequence_padding



checkpoint_path = "./chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "./chinese_wwm_ext_L-12_H-768_A-12/vocab.txt"
config_path = "./chinese_wwm_ext_L-12_H-768_A-12/bert_config.json"

tokenizers = Tokenizer(dict_path, do_lower_case=True, word_maxlen=512)

def main(args):
    save_path = args.path + args.pkl

    file_list = ['Rtrain', 'Rvalid', 'Rtest']  # 数据集列表

    thu_path = './Thuocl_seg.txt'
    stopword_path = './stop_word.txt'

    cut = get_cutter(dict_path=thu_path, stopword_path=stopword_path, stop_words_filtered=True)

    for i in range(len(file_list)):
        token_id = []
        segment_id = []

        law_label_lists = []
        accu_label_lists = []
        term_lists = []

        with open(args.path + 'new_data/' + '{0}.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
            total = 0
            for line in f.readlines():
                total += 1
                line = json.loads(line.strip()) #去除可能存在的空格
                fact = line['fact_cut']

                #sentence = seg_sentence(fact, cut)

                token_ids, segment_ids = tokenizers.encode(fact, maxlen = 512)
                #segment_ids = [0] * 512

                token_id.append(token_ids)
                segment_id.append(segment_ids)

                law_label_lists.append(line['law'])
                accu_label_lists.append(line['accu'])
                term_lists.append(line['term'])

                if total % 100 == 0:
                    print(total)
            f.close()

        token_id = sequence_padding(token_id, 512)
        segment_id = sequence_padding(segment_id, 512)

        data_dict = {'token_id': token_id, 'segment_id': segment_id, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists,
                     'term_lists': term_lists}
        pk.dump(data_dict, open(save_path + '{0}_processed_bert.pkl'.format(file_list[i]), 'wb'))
        print('For BERT-based models: {0}_dataset is processed over'.format(file_list[i]) + '\n')


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Structure")
    parser.add_argument('-p', dest='path', default='./data/')
    parser.add_argument('-d', dest='pkl', default='pkl/')  # or 'big/'
    #parser.add_argument('-b', dest='bert', default='Bert/' )
    args = parser.parse_args()
    main(args)