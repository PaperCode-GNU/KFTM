import argparse

from data_processed import get_law_index, get_accu_index, get_statistics_for_filter, filter_law_and_accu, filter_samples, split_seed_randomly
from utils.log import set_logger

def main(args):
    # 相关文件的路径
    save_path = args.path + args.problem
    train_path = args.path + args.problem + args.train
    valid_path = args.path + args.problem + args.valid
    test_path = args.path + args.problem + args.test

    # if args.problem == 'big/':
    #     # If dataset is CAIL-big, we need to create a validation set
    #     split_seed_randomly(save_path, 'train.json', 'test.json', args.seed, args.ratio)

    # 统计指控和法律的频率，并对其进行过滤，只生成数据中所提到的相应指控和法条
    law_file = open(args.path + 'law_accu/law.txt', 'r')
    law2num, num2law, total_law = get_law_index(law_file)

    accu_file = open(args.path + 'law_accu/accu.txt', 'r', encoding='utf-8')
    accu2num, num2accu, total_accu = get_accu_index(accu_file)

    frequency_law, frequency_accu = get_statistics_for_filter(train_path, None, total_law, law2num,
                                                              total_accu, accu2num)

    filter_law_list, filter_law2num, filter_accu_list, filter_accu2num = filter_law_and_accu(args.path, total_law,
                                                                                             num2law, frequency_law,
                                                                                             total_accu, num2accu,
                                                                                             frequency_accu)

    # 生成新的处理后的数据
    # filter_samples(args.path, train_path, valid_path, test_path, law2num, filter_law_list,
    #                filter_law2num, accu2num, filter_accu_list, filter_accu2num)


if __name__ == "__main__":
    #设置参数
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('-p', dest='path', default='./data/')
    parser.add_argument('-d', dest='problem', default='small/')  # or 'big/'
    parser.add_argument('-i', dest='is_modify', action='store_true', default=False)
    parser.add_argument('-tr', dest='train', help='train set', default='train.json')
    parser.add_argument('-v', dest='valid', help='valid set', default='data_valid.json')
    parser.add_argument('-te', dest='test', help='test set', default='data_test.json')
    parser.add_argument('-s', dest='seed', help='random seed', type=int, default=0)
    parser.add_argument('-r', dest='ratio', help='ratio for validation', type=float, default=0.1)
    args = parser.parse_args()
    main(args)