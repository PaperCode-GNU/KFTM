import jieba
import thulac


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

#去除停用词
def get_cutter(dict_path, stopword_path, mode='thulac', stop_words_filtered=True):
    if stop_words_filtered:
        stopwords = stopwordslist(stopword_path)
    else:
        stopwords = []
    if mode == 'jieba': #加入thuocl_seg 和stop_word两个中的分词
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]


# 截断句子长度
def seg_sentence(sentence, cut):
    cut_sentence = cut(sentence)  # 去除停用词
    join_sentence = ''.join(cut_sentence)

    length = len(join_sentence)
    if length > 254:
        join_sentence = join_sentence[-254:]

    return join_sentence





