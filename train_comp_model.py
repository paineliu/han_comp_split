from palm_to_stroke_data import conv_to_jsonl
from make_train_data import *
from train_model import *
from conv_to_stroke_data import *

if __name__=='__main__':

    width = 256
    height = 256
    # conv_to_jsonl('./data/3rd/casia/Pot1.0Test', './data/jsonl/casia_test.jsonl')
    # conv_to_jsonl('./data/3rd/casia/Pot1.0Val', './data/jsonl/casia_val.jsonl')
    # conv_to_jsonl('./data/3rd/casia/Pot1.0Train', './data/jsonl/casia_train.jsonl')
    
    # # # 生成平滑和包含部件信息的数据
    # for prefix in ['casia']:
    #     for item in ['test', 'val', 'train']:
    #         stroke_data_normalize('./data/jsonl/{}_{}.jsonl'.format(prefix, item), './data/jsonl/{}_256x256_{}.jsonl'.format(prefix, item), width, height)
    #         stroke_data_add_comp('./labels/han.jsonl', './labels/comp.jsonl', './data/jsonl/{}_256x256_{}.jsonl'.format(prefix, item), './data/jsonl/{}_256x256_comp_{}.jsonl'.format(prefix, item))
    #         make_comp_train_data('./labels/han.jsonl', './labels/comp.jsonl', './data/jsonl/{}_256x256_comp_{}.jsonl'.format(prefix,item), './data/result/han_comp_{}_{}.h5'.format(prefix,item), './data/result/han_comp_{}_{}.jsonl'.format(prefix,item))    
            

    han_filename = './labels/han.jsonl'
    comp_filename = './labels/comp.jsonl'

    han_comp = HanComp(han_filename, comp_filename)
    han_label = HanCompLabel(han_comp)

    train_model(han_label, 'han_comp_casia', epochs = 100, add_han_label_dataset=False)
