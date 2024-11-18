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

def scut_file_to_json_file(idx_filename, dat_filename, f_json):
    print(idx_filename)
    base_filename = os.path.basename(idx_filename)
    base_filename = os.path.splitext(base_filename)[0]

    f = open(idx_filename, 'rb')
    data = f.read(4)
    (sample_sum, ) = struct.unpack('i', data)
    list_dat_offset = []
    for i in range(sample_sum):
        # SampleState
        data = f.read(1)
        (sample_state, ) = struct.unpack('B', data)

        # OswIndex IdxIndex DatOffset
        data = f.read(12)
        (osw_index, idx_index, dat_offset) = struct.unpack('III', data)
        list_dat_offset.append(dat_offset)
    f.close()

    f = open(dat_filename, 'rb')
    for dat_offset in list_dat_offset:
        f.seek(dat_offset)
        data = f.read(1)
        (word_length, ) = struct.unpack('B', data)
        data = f.read(word_length)
        try:
            code = data.decode('gbk')

            data = f.read(6)
            (point_num, line_num, get_time_point_num) = struct.unpack('HHH', data)

            get_time_point_index = []
            data = f.read(2 * get_time_point_num)

            for i in range(get_time_point_num):
                (point_index,) = struct.unpack('H', data[2*i:2*(i+1)])
                get_time_point_index.append(point_index)

            elapsed_time = []
            data = f.read(4 * get_time_point_num)

            for i in range(get_time_point_num):
                (etime,) = struct.unpack('I', data[4*i:4*(i+1)])
                elapsed_time.append(etime)
            stroke_point_list = []
            line_list = []

            for l in range(line_num):
                data = f.read(2)
                (stroke_point_num, ) = struct.unpack('H', data)
                line_list.append(stroke_point_num)

                data = f.read(4 * stroke_point_num)
                for i in range(stroke_point_num):
                    (x, y) = struct.unpack('HH', data[4*i:4*(i+1)])
                    stroke_point_list.append({'x':x, 'y': y})

            for i in range(get_time_point_num):
                if (get_time_point_index[i] < len(stroke_point_list)):
                    stroke_point_list[get_time_point_index[i]]['t'] = elapsed_time[i]
                else:
                    pass
                    # print('err', dat_offset, i, 'point_index =', get_time_point_index[i], 'point_num =', point_num)
            
            stroke_point_begin = 0
            stroke_point_end = 0
            stroke_data = []

            for l in range(line_num):
                stroke_point_end = stroke_point_begin + line_list[l]
                line_data = stroke_point_list[stroke_point_begin:stroke_point_end]
                stroke_data.append(line_data)
                stroke_point_begin = stroke_point_end

            # calc rect
            l = sys.maxsize
            t = sys.maxsize
            r = 0
            b = 0
            for stroke in stroke_data:
                for point in stroke:
                    x = point['x']
                    y = point['y']

                    l = x if x < l else l
                    t = y if y < t else t
                    r = x if x > r else r
                    b = y if y > b else b

            for stroke in stroke_data:
                for point in stroke:
                    point['x'] -= l
                    point['y'] -= t

            w = r-l
            h = b-t
            rect = {'left': l,'top': t, 'width': w, 'height': h}
           
            jdata = {'filename':base_filename, 'label': code, 'rect': rect, 'strokes': stroke_data}
            f_json.write('{}\n'.format(json.dumps(jdata, ensure_ascii=False)))

        except:
            print('err', dat_offset, 'decode:', data)
            pass
    f.close()

def hw_file_to_json_file(input_filename, f_json):
    print(input_filename)
    base_filename = os.path.basename(input_filename)
    base_filename = os.path.splitext(base_filename)[0]

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

        out_data = {'filename':base_filename, 'label': jdata['sel'], 'rect': rect, 'strokes': stroke_data}
        f_json.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))

    f.close()

def pot_file_to_json_file(pot_filename, f_json):
    print(' -> {}'.format(pot_filename))
    base_filename = os.path.basename(pot_filename)
    f = open(pot_filename, 'rb')
    while True:
        header_size = 8
        data = f.read(header_size)
        if len(data) != header_size:
            break
        (sample_size, gbk_code1, gbk_code2, _,
        stroke_num) = struct.unpack("<1H1s1s1H1H", data)
        if gbk_code1 == b'\x00':
            gbk_code = gbk_code2
        else:
            gbk_code = gbk_code2 + gbk_code1
        tagcode = gbk_code.decode('gbk')
        point_total = int((sample_size - 8) / 4)
        point_list = []
        for i in range(point_total):
            data = f.read(4)
            (x, y) = struct.unpack("<1h1h", data)
            point_list.append({'x': x, 'y': y})

        stroke_idx = 0
        point_data = []
        stroke_data = []
        l = sys.maxsize
        t = sys.maxsize
        r = 0
        b = 0
        for i in range(len(point_list)):
            x = point_list[i]['x']
            y = point_list[i]['y']

            if x == -1 and y == -1:
                continue

            if x != -1:
                l = x if x < l else l
                t = y if y < t else t
                r = x if x > r else r
                b = y if y > b else b
                point_data.append({'x': x, 'y': y})
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

        jdata = {'filename':base_filename, 'label': tagcode,  'rect': rect,'strokes': stroke_data}
        f_json.write('{}\n'.format(json.dumps(jdata, ensure_ascii=False)))


def conv_to_jsonl(data_pathname, jsonl_filename):
    print('write {}'.format(jsonl_filename))
            
    os.makedirs(os.path.dirname(jsonl_filename), exist_ok=True)
    f_json = open(jsonl_filename, 'w', encoding='utf-8')
    for parent, dirs, files in os.walk(data_pathname):
        for filename in files:
            fullname = os.path.join(parent, filename)
            if filename.endswith('.json'):
                hw_file_to_json_file(fullname, f_json)
                
            if filename.endswith('.jsonl'):
                hw_file_to_json_file(fullname, f_json)

            if filename.endswith('.pot'):
                pot_file_to_json_file(fullname, f_json)

            if filename.endswith('.idx'):
                idx_filename = os.path.join(parent, filename)
                dat_filename = os.path.join(parent, filename[:-4] + ".dat")
                scut_file_to_json_file(idx_filename, dat_filename, f_json)
            

    print("finished.")
    
if __name__=='__main__':

    # # args = parse_args()
    # conv_to_jsonl('./data/3rd/casia/Pot1.0Test', './data/jsonl/casia_test.jsonl')
    # conv_to_jsonl('./data/3rd/casia/Pot1.0Val', './data/jsonl/casia_val.jsonl')
    # conv_to_jsonl('./data/3rd/casia/Pot1.0Train', './data/jsonl/casia_train.jsonl')

    # conv_to_jsonl('./data/3rd/scut/Couch_GB1_188/test', './data/jsonl/scut_gb1_188_test.jsonl')
    # conv_to_jsonl('./data/3rd/scut/Couch_GB1_188/train', './data/jsonl/scut_gb1_188_train.jsonl')
    # conv_to_jsonl('./data/3rd/scut/Couch_GB1_188/val', './data/jsonl/scut_gb1_188_val.jsonl')

    #遍历文件夹下的子文件夹
    output_path = './data/jsonl/casia'
    os.makedirs(output_path, exist_ok=True)
    for parent, dirs, files in os.walk('./data/casia'):
        for dir in dirs:
            full_path = os.path.join(parent, dir)
            print(full_path, os.path.join(output_path, dir + '.jsonl'))
            conv_to_jsonl(full_path, os.path.join(output_path, dir + '.jsonl'))

