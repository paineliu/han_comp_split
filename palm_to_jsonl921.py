import os
import sys
import json
import struct
import argparse
import han_comp

def parse_args():
    parser = argparse.ArgumentParser(description='casia pot and scut data to dink jsonl convert tools')
    parser.add_argument('-i', '--input_pathname', type=str, required=True, help='casia pot | scut data pathname')
    parser.add_argument('-o', '--output_filename', type=str, required=True, help='dink jsonl filename')

    return parser.parse_args()
# 6751

def hw_file_to_json_file(input_filename, f_all_json, f_train_json, f_val_json, f_test_json):
    base_filename = os.path.basename(input_filename)
    base_filename = os.path.splitext(base_filename)[0]
    line = 0
    f = open(input_filename, 'r', encoding='utf-8')
    for each in f:
        jdata = json.loads(each)
        point_list = jdata['data'].split(',')
        stroke_idx = 0
        point_data = []
        stroke_data = []
        l = sys.maxsize
        t = sys.maxsize
        r = 0
        b = 0
        for i in range(0, len(point_list), 3):
            x = int(point_list[i])
            y = int(point_list[i+1])
            t = int(point_list[i+2])

            if x == -1 and y == -1:
                continue

            if x != -1:
                l = x if x < l else l
                t = y if y < t else t
                r = x if x > r else r
                b = y if y > b else b
                point_data.append({'x': x, 'y': y, 't':t})
            else:
                stroke_data.append(point_data.copy())
                point_data.clear()
                stroke_idx += 1

        for stroke in stroke_data:
            for point in stroke:
                point['x'] -= l
                point['y'] -= t
        w = r-l
        h = b-t
        rect = {'left': l, 'top': t, 'width': w, 'height': h}

        out_data = {'label': jdata['sel'], 'rect': rect, 'strokes': stroke_data}
        f_all_json.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))
        if line % 10 <= 7:
            f_train_json.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))
        elif line % 10 <= 8:
            f_val_json.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))
        else:
            f_test_json.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))

        line += 1
    f.close()


def conv_to_jsonl(data_pathname, jsonl_pathname, prefix, cut=False):
    hanComp = han_comp.HanComp('./labels/han.jsonl', './labels/comp.jsonl')
    print('write {}'.format(jsonl_pathname))
    os.makedirs(jsonl_pathname, exist_ok=True)
    f_all_json = open(os.path.join(jsonl_pathname, prefix + 'all.jsonl'), 'w', encoding='utf-8')
    f_train_json = open(os.path.join(jsonl_pathname, prefix + 'train.jsonl'), 'w', encoding='utf-8')
    f_val_json = open(os.path.join(jsonl_pathname, prefix + 'val.jsonl'), 'w', encoding='utf-8')
    f_test_json = open(os.path.join(jsonl_pathname, prefix + 'test.jsonl'), 'w', encoding='utf-8')
    file_total = 0
    for parent, dirs, files in os.walk(data_pathname):
        files = sorted(files)
        for filename in files:
            fullname = os.path.join(parent, filename)
            if filename.endswith('.json'):
                basename = filename.split('.')[0]
                han = chr(int(basename, 16))
                han_id = hanComp.get_han_id(han)
                if cut:
                    if han_id != -1:
                        print(file_total, hanComp.get_han_total(), '{:.2%}'.format(file_total / hanComp.get_han_total()), fullname)
                        hw_file_to_json_file(fullname, f_all_json, f_train_json, f_val_json, f_test_json)
                        file_total += 1
                else:
                    print(file_total, len(files), '{:.2%}'.format(file_total / len(files)), fullname)
                    hw_file_to_json_file(fullname, f_all_json, f_train_json, f_val_json, f_test_json)
                    file_total += 1

    print("finished.")
    
if __name__=='__main__':

    # args = parse_args()

    conv_to_jsonl('./data/3rd/palm/sample_921', './data/output/palm', 'palm_921_', True)
    conv_to_jsonl('./data/3rd/palm/sample_920', './data/output/palm', 'palm_920_', True)
    conv_to_jsonl('./data/3rd/palm/sample_gbk', './data/output/palm', 'palm_gbk_', True)

    conv_to_jsonl('./data/3rd/palm/sample_921', './data/output/palm', 'palm_full_921_')
    conv_to_jsonl('./data/3rd/palm/sample_920', './data/output/palm', 'palm_full_920_')
    conv_to_jsonl('./data/3rd/palm/sample_gbk', './data/output/palm', 'palm_full_gbk_')
