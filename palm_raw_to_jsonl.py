import os
import sys
import json
import struct
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='casia pot and scut data to dink jsonl convert tools')
    parser.add_argument('-i', '--input_pathname', type=str, required=True, help='casia pot | scut data pathname')
    parser.add_argument('-o', '--output_filename', type=str, required=True, help='dink jsonl filename')

    return parser.parse_args()
# 6751

def hw_file_to_json_file(input_filename, f_all_json, f_train_json, f_val_json, f_test_json):
    print(input_filename)
    base_filename = os.path.basename(input_filename)
    base_filename = os.path.splitext(base_filename)[0]
    line = 0
    f = open(input_filename, 'r', encoding='utf-8')
    for each in f:
        jdata = json.loads(each)
        if 'cand' in jdata:
            if jdata['cand'][0] != jdata['sel']:
                continue
        if 'kid' in jdata:
            if jdata['kid'] != 'hp':
                continue

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
        if (line > 1000000):
            break
    f.close()


def conv_file_to_json_file(input_filename, output_pathname, out_perfix):
    os.makedirs(output_pathname, exist_ok=True)
    f_all_json = open(os.path.join(output_pathname, 'palm_' + out_perfix + '_all.jsonl'), 'w', encoding='utf-8')
    f_train_json = open(os.path.join(output_pathname, 'palm_' + out_perfix + '_train.jsonl'), 'w', encoding='utf-8')
    f_val_json = open(os.path.join(output_pathname, 'palm_' + out_perfix + '_val.jsonl'), 'w', encoding='utf-8')
    f_test_json = open(os.path.join(output_pathname, 'palm_' + out_perfix + '_test.jsonl'), 'w', encoding='utf-8')
        hw_file_to_json_file(input_filename, f_all_json, f_train_json, f_val_json, f_test_json)

def conv_to_jsonl(data_pathname, jsonl_pathname):
    print('write {}'.format(jsonl_pathname))
            
    os.makedirs(jsonl_pathname, exist_ok=True)
    f_all_json = open(os.path.join(jsonl_pathname, '_all.jsonl'), 'w', encoding='utf-8')
    f_train_json = open(os.path.join(jsonl_pathname, '_train.jsonl'), 'w', encoding='utf-8')
    f_val_json = open(os.path.join(jsonl_pathname, '_val.jsonl'), 'w', encoding='utf-8')
    f_test_json = open(os.path.join(jsonl_pathname, '_test.jsonl'), 'w', encoding='utf-8')
    for parent, dirs, files in os.walk(data_pathname):
        for filename in files:
            fullname = os.path.join(parent, filename)
            if filename.endswith('.json'):
                hw_file_to_json_file(fullname, f_all_json, f_train_json, f_val_json, f_test_json)
                
            if filename.endswith('.jsonl'):
                hw_file_to_json_file(fullname, f_all_json, f_train_json, f_val_json, f_test_json)


    print("finished.")
    
if __name__=='__main__':

    # args = parse_args()
    conv_file_to_json_file('./3rd/paine/4f60.json', './data/sdata/',chr(0x4f60))
    # conv_to_jsonl('E:\\corpus\\handwrite\\samples\\sample_all\\sample', 'E:\\corpus\\handwrite\\samples\\palm')
