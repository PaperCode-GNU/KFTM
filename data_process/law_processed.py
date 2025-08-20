import numpy as np
import re
import jieba
import thulac
import pickle as pk

from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import load_model
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array, sequence_padding

def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)

#将法条放入一个list
def law_to_list(path, remain_new_line = False):
    with open(path, 'r', encoding='utf-8') as f:
        law = []
        for line in f:
            if line == '\n' or re.compile(r'第.*[节|章]').search(line[:10]) is not None:
                continue
            try:
                tmp = re.compile(r'第.*条').search(line.strip()[:8]).group(0)
                if remain_new_line:
                    law.append(line)
                else:
                    law.append(line.strip())
            except (TypeError, AttributeError):
                if remain_new_line:
                    law[-1] += line
                else:
                    law[-1] += line.strip()
    return law


#处理法条
def process_law(law):
    # single article
    # cut=get_cutter()
    condition_list = []
    for each in law.split('。')[:-1]:#将句尾的句号去除
        suffix = None
        if '：' in each: #找到有行为解释的法条
            each, suffix = each.split('：') #suffix：包括的行为
            #suffix = cut(suffix)#对行为进行分词
        #words = cut(each) #对所有each进行分词
        words = each
        seg_point = [-1]
        conditions = []

        for i in range(len(words)):
            if words[i] == '；' or words[i] == ';':
                seg_point.append(i)
        seg_point.append(len(words))
        for i in range(len(seg_point) - 1):
            for j in range(seg_point[i + 1] - 1, seg_point[i], -1):
                if j + 1 < len(words) and words[j] == '的' and words[j + 1] == '，':
                    conditions.append(words[seg_point[i] + 1:j + 1])
                    break
        # context=law.split('。')[:-1]
        for i in range(1, len(conditions)):
            conditions[i] = conditions[0] + conditions[i]
        # if len(condition_list)==0 and len(conditions)==0:
        #     conditions.append([])
        if suffix is not None:
            conditions = [x + suffix for x in conditions]
        condition_list += conditions
        condition_list = ''.join(condition_list)

    if condition_list == []:
        condition_list.append(law[:-1])
    n_word = [len(i) for i in condition_list] #多少个词
    return condition_list, n_word

def cut_law(law_list, order=None, cut_sentence=True, cut_penalty=False, stop_words_filtered=True):
    res = []
    #cut = get_cutter(stop_words_filtered=stop_words_filtered)
    if order is not None:
        key_list = [int(i) for i in order.keys()]
        filter = key_list
    for each in law_list:
        index, content = each.split('　')
        index = hanzi_to_num(index[1:-1])
        charge, content = content[1:].split('】')
        # if charge[-1]!='罪':
        #     continue
        if order is not None and index not in filter:
            continue
        if cut_penalty:
            context, n_words = process_law(content) #context:处理完的法条，一整条没有分行，n_words:有多少个词
        elif cut_sentence:#有分行
            context, n_words = [], []
            for i in content.split('。'):
                if i != '':
                    #context.append(cut(i))
                    context.append(i)
                    n_words.append(len(context[-1]))
        else:
            #context = cut(content)
            context = content
            n_words = len(context)
        res.append([index, charge, context, n_words])#index：第几条，charge：罪名，context：法条内容，n_words:分词完后有多少个词多少个词
    if order is not None:
        res = sorted(res, key=lambda x: order[str(x[0])])
    return res

def get_keylist(order = None):
    key_list = [int(i) for i in order.keys()]
    filter_list = key_list
    return filter_list

def seg_sentence(sentence):

    length = len(sentence)
    if length > 510:
        sentence = sentence[: 510]

    return sentence

def get_token_id(x, tokenizers):
    token_id = []
    segment_id = []

    for each in x:
        sentence = seg_sentence(each)
        token_ids, segment_ids = tokenizers.encode(sentence, maxlen=512)

        token_id.append(token_ids)
        segment_id.append(segment_ids)

    token_id = sequence_padding(token_id, 512)
    segment_id = sequence_padding(segment_id, 512)

    return token_id, segment_id

chapter_list = [[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],[114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139],
                [140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],[151, 152, 153, 154, 155, 156, 157],[158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169],
                [170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191],
                [192, 193, 194, 195, 196, 197, 198, 199, 200],[201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220],
                [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231],[232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262],
                [263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276],[277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304],
                [305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317],[318, 319, 320, 321, 322, 323],[324, 325, 326, 327, 328, 329],
                [330, 331, 332, 333, 334, 335, 336, 337],[338, 339, 340, 341, 342, 343, 344, 345, 346],[347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357],
                [358, 359, 360, 361, 362],[363, 364, 365, 366, 367],[368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381],
                [382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396],[397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419],
                [420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 451]]



def gen_law_relation(law_label2index_path):

    law_file_order = pk.load(open(law_label2index_path, 'rb'))

    law_list = law_to_list('./data/law_accu/criminal_law.txt')

    law_1 = cut_law(law_list, order=law_file_order, cut_sentence=False, cut_penalty=False, stop_words_filtered=True)
    law_1 = list(zip(*law_1))
    #token_id, segment_id = get_token_id(law_1[-2], tokenizers)
    key_list = get_keylist(order=law_file_order)


    return key_list, chapter_list

def gen_law_relation_all(law_label2index_path):

    tokenizers = Tokenizer(dict_path, do_lower_case=True, word_maxlen=512)
    law_file_order = pk.load(open(law_label2index_path, 'rb'))

    law_list = law_to_list('./data/law_accu/criminal_law.txt')

    law_1 = cut_law(law_list, order=law_file_order, cut_sentence=False, cut_penalty=False, stop_words_filtered=True)
    law_1 = list(zip(*law_1))

    token_id, segment_id = get_token_id(law_1[-2], tokenizers)
    key_list = get_keylist(order=law_file_order)

    return token_id, segment_id, key_list, chapter_list