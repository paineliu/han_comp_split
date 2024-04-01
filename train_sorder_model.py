from palm_to_stroke_data import conv_to_jsonl
from make_train_data import *
from train_model import *

if __name__=='__main__':

    width = 256
    height = 256

    han_code = '4e01' # 丁 0.99
    han_code = '6728' # 木 0.98
    han_code = '79cb' # 秋 0.54
    # han_code = '8bed' # 语 0.71
    han_code = '4f60'  # 
    
    # conv_to_jsonl('./data/sd/{}'.format(han_code), './data/jsonl', 'palm_{}_'.format(han_code))
    
    # # 生成平滑和包含部件信息的数据
    # for prefix in ['palm_{}'.format(han_code)]:
    #     for item in ['test', 'val', 'train']:
    #         stroke_data_normalize('./data/jsonl/{}_{}.jsonl'.format(prefix, item), './data/jsonl/{}_256x256_{}.jsonl'.format(prefix, item), width, height)
    #         stroke_data_add_comp('./labels/han.jsonl', './labels/comp.jsonl', './data/jsonl/{}_256x256_{}.jsonl'.format(prefix, item), './data/jsonl/{}_256x256_comp_{}.jsonl'.format(prefix, item))
    #         make_sorder_train_data([], './data/jsonl/{}_256x256_comp_{}.jsonl'.format(prefix, item), './data/result/han_sorder_{}_{}.h5'.format(prefix, item), './data/result/han_sorder_{}_{}.jsonl'.format(prefix, item))

    han_filename = './labels/han.jsonl'
    comp_filename = './labels/comp.jsonl'

    han_comp = HanComp(han_filename, comp_filename)
    han_order_label = HanOrderLabel(han_comp)

    train_model(han_order_label, 'han_sorder_palm_{}'.format(han_code), epochs = 100, add_han_label_dataset=False)
