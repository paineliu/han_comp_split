from palm_to_stroke_data import conv_to_jsonl
from make_train_data import *
from train_model import *

if __name__=='__main__':

    width = 256
    height = 256

    han_code = '4e01' # 丁
    han_code = '6728' # 木
    han_code = '79cb' # 1.0
    # han_code = '8bed' # 语 0.89
    han_code = '738b' # 王 1.0
    han_code = '4f60' # 你 1.0
    han_code = '600e' # 怎 
    han_code = '597d' # 怎 

    
    
    conv_to_jsonl('./data/3rd/palm/{}'.format(han_code), './data/jsonl', 'palm_{}_'.format(han_code))
    
    # # 生成平滑和包含部件信息的数据
    for prefix in ['palm_{}'.format(han_code)]:
        for item in ['test', 'val', 'train']:
            stroke_data_normalize('./data/jsonl/{}_{}.jsonl'.format(prefix, item), './data/jsonl/{}_256x256_{}.jsonl'.format(prefix, item), width, height)
            stroke_data_add_comp('./labels/han.jsonl', './labels/comp.jsonl', './data/jsonl/{}_256x256_{}.jsonl'.format(prefix, item), './data/jsonl/{}_256x256_comp_{}.jsonl'.format(prefix, item))
            make_stroke_train_data('./labels/han.jsonl', './labels/comp.jsonl', [], './data/jsonl/{}_256x256_comp_{}.jsonl'.format(prefix, item), './data/result/han_stroke_{}_{}.h5'.format(prefix, item), './data/result/han_stroke_{}_{}.jsonl'.format(prefix, item))

    han_filename = './labels/han.jsonl'
    comp_filename = './labels/comp.jsonl'

    han_comp = HanComp(han_filename, comp_filename)
    han_stroke_label = HanStrokeLabel(han_comp)

    train_model(han_stroke_label, 'han_stroke_palm_{}'.format(han_code), epochs = 100, add_han_label_dataset=False)
