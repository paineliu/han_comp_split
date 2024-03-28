import os
import numpy as np
import h5py
import json
import time
import argparse
import random
from conv_to_jsonl import conv_to_jsonl

from han_comp import HanComp

def parse_args():
    parser = argparse.ArgumentParser(description='dink jsonl to radical jsonl convert tools')
    parser.add_argument('-z', '--hanzi', type=str, required=True, help='hanzi csv filename')
    parser.add_argument('-r', '--radical', type=str, required=True, help='radical csv filename')
    parser.add_argument('-i', '--input', type=str, required=True, help='dink json filename')
    parser.add_argument('-o', '--output', type=str, required=True, help='radical jsonl filename')

    return parser.parse_args()

def dink_normalize(casia_filename, dink_filename, norm_width, norm_height):

    def proc_line_data(dink_data, max_width, max_height):
        norm_data = dink_data
        
        norm_field = "normalize"
        norm_data[norm_field] = {}
        norm_data[norm_field]['size'] = {"width":max_width, "height":max_height}     
        
        norm_stroke = []
        max_len = max(norm_data['rect']['width'], norm_data['rect']['height'])
        for stroke in norm_data['strokes']:
            out_points = []
            last_x = -1
            last_y = -1
            for point in stroke:
                if 't' in point:
                    x = int(point['x'] * max_width / max_len)
                    y = int(point['y'] * max_height / max_len)
                    t = int(point['t'])
                    out_points.append({'x': x, 'y': y, 't':t})
                else:
                    x = int(point['x'] * max_width / max_len)
                    y = int(point['y'] * max_height / max_len)
                    if (last_x == x and last_y == y):
                        continue
                    else:
                        out_points.append({'x': x, 'y': y})
                        last_x = x
                        last_y = y

            norm_stroke.append(out_points)
        norm_data[norm_field]['strokes'] = norm_stroke

        return norm_data
    
    print('stat {}'.format(casia_filename))
    os.makedirs(os.path.dirname(dink_filename), exist_ok=True)
    f_casia = open(casia_filename, 'r', encoding='utf-8')
    f_dink = open(dink_filename, 'w', encoding='utf-8')
    line_total = 0
    start_time = time.time()
    begin_time = start_time
    
    with open(casia_filename, 'r', encoding='utf-8') as file:
        num_lines = sum(1 for line in file)
        end_time = time.time()
        date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
        print('{} time={:.1f}(m) line={}'.format(date_str, (end_time - start_time)/60, num_lines))
        
    begin_time = time.time()
    print('write {}'.format(dink_filename))
    for each in f_casia:
        line_total += 1
        casia_data = json.loads(each)
        dink_data = proc_line_data(casia_data, norm_width, norm_height)
        end_time = time.time()
        if line_total == 1 or end_time - begin_time > 6:
            begin_time = end_time
            date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
            print('{} time={:.1f}(m) line={:.2%} ({}/{}) label={}'.format(date_str, (end_time - start_time)/60, line_total / num_lines, line_total, num_lines, dink_data['label']))

        f_dink.write('{}\n'.format(json.dumps(dink_data, ensure_ascii=False)))
    date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
    end_time = time.time()
    print('{} time={:.1f}(m) line={:.2%} ({}/{})'.format(date_str, (end_time - start_time)/60, line_total / num_lines, line_total, num_lines))
    print('finished.')

def dink_add_comp(han_filename, comp_filename, dink_norm_filename, dink_comp_filename):
    
    map_han = {}
    # han_reco = Recognize()
    han_comp = HanComp(han_filename, comp_filename)

    print('stat {}'.format(dink_norm_filename))
    with open(dink_norm_filename, 'r', encoding='utf-8') as file:
        num_lines = sum(1 for line in file)

    os.makedirs(os.path.dirname(dink_comp_filename), exist_ok=True)
    f_d2 = open(dink_norm_filename, 'r', encoding='utf-8')
    f_d3 = open(dink_comp_filename, 'w', encoding='utf-8')
    line_total = 0
    valid_line_total = 0
    start_time = time.time()
    begin_time = start_time

    def get_stroke_list(strokes):
        lst_stroke = []
        for stroke in strokes:
            points = [] 
            for point in stroke:
                points.append([point['x'], point['y']])
            lst_stroke.append(points)
        return lst_stroke
    
    def get_point_list(strokes):
        lst_point = []
        for s_id, stroke in enumerate(strokes):
            for p_id, point in enumerate(stroke):
                x = point['x']
                y = point['y']
                t = point.get('t', 0)
                lst_point.append([s_id, p_id, x, y, t])
        return lst_point
    
    print('write {}'.format(dink_comp_filename))
    for each in f_d2:
        line_total += 1
        dink_data = json.loads(each)
        han = dink_data['label']
        han_stroke_num = han_comp.get_han_stroke_num(han)
        if (han_stroke_num == len(dink_data['normalize']['strokes'])):
            comp_data = {}
            comp_data['han'] = han
            compoments = han_comp.get_han_comp(han)
            comp_lst = []
            comp_begin = 0
            for comp in compoments:
                comp_len = han_comp.get_comp_stroke_num(comp)
                comp_lst.append({'name':comp, 'range':[comp_begin, comp_begin + comp_len]})
                comp_begin += comp_len

            comp_data['compoments'] = comp_lst
            comp_data['strokes'] = dink_data['normalize']['strokes']
            # comp_data['points'] = get_point_list(comp_data['strokes'])
            if len(comp_data['compoments']) > 1:
                map_han[han] = map_han.get(han, 0) + 1
            #     print(comp_data['han'], comp_data['compoment'])
            #     begin = 0
            #     for comp in comp_data['compoment']:
            #         comp_stroke_num = han_comp.get_comp_stroke_num(comp)
            #         end = begin + comp_stroke_num
            #         strokes = get_stroke_list(comp_data['strokes'][begin: end])
            #         cands = han_reco.Single(strokes, 10)
            #         print(cands)
            #         begin = end
                
            f_d3.write('{}\n'.format(json.dumps(comp_data, ensure_ascii=False)))
            valid_line_total += 1
        else:
            # print(dink_data['label'], han_stroke_num, len(dink_data['normalize']['strokes']))
            pass

        end_time = time.time()
        if line_total == 1 or end_time - begin_time > 6:
            begin_time = end_time
            date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
            print('{} time={:.1f}(m) valid={} line={:.2%}({}/{}) label={}'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines, han))


    date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
    end_time = time.time()
    print('{} time={:.1f}(m) vaild={} line={:.2%}({}/{})'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines))
    print('finished.')
    # print(len(map_han))
    # print(map_han)

class H5HanDataset():

    def __init__(self, h5_filename):
        
        self.h5file = h5py.File(h5_filename, 'r') 
        self.samples_han = self.h5file["han"]
        self.samples_labels = self.h5file["labels"]
        self.samples_points = self.h5file["points"]
        self.num_samples = len(self.samples_han)
            
    def __getitem__(self, idx):
        han = self.samples_han[idx].decode('utf-8')
        points = self.samples_points[idx].reshape(-1, 4)
        labels = self.samples_labels[idx]
        return {"han":han, "labels":labels, "points":points}
    
    def __len__(self):
        return self.num_samples

def dink_comp_extr(han_filename, comp_filename, input_filename, h5_filename, output_filename):
    
    han_comp = HanComp(han_filename, comp_filename)

    print('stat {}'.format(input_filename))
    with open(input_filename, 'r', encoding='utf-8') as file:
        num_lines = sum(1 for line in file)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    lst_han = []
    lst_labels = []
    lst_points = []

    f_d2 = open(input_filename, 'r', encoding='utf-8')
    f_d3 = open(output_filename, 'w', encoding='utf-8')
    line_total = 0
    valid_line_total = 0
    start_time = time.time()
    begin_time = start_time
    
    print('write {}'.format(output_filename))
    for each in f_d2:
        line_total += 1
        dink_data = json.loads(each)
        
        han = dink_data['han']
        compoments = dink_data['compoments']
        strokes = dink_data['strokes']
        comp_ids = []
        for comp in compoments:
            comp_ids.append(han_comp.get_comp_id(comp['name']) + 1)

        valid_line_total += 1
        points = []
        for sid, stroke in enumerate(strokes):
            for pid, point in enumerate(stroke):
                points.append([sid, pid, point['x'], point['y']])

        out_data = {}
        out_data['han'] = han
        out_data['labels'] = comp_ids
        out_data['points'] = points
        point_items = []
        for point in points:
            for item in point:
                point_items.append(item)

        lst_han.append(han)
        lst_labels.append(np.asarray(comp_ids, dtype=np.int32))
        lst_points.append(np.asarray(point_items, dtype=np.int32))
        f_d3.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))

        end_time = time.time()
        if line_total == 1 or end_time - begin_time > 6:
            begin_time = end_time
            date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
            print('{} time={:.1f}(m) valid={} line={:.2%} ({}/{}) label={}'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines, han))


    date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
    end_time = time.time()
    print('{} time={:.1f}(m) vaild={} line={:.2%} ({}/{})'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines))
    print('finished.')

    f_h5 = h5py.File(h5_filename, 'w')     

    data_han = np.array(lst_han)
    dtype_han = h5py.special_dtype(vlen=str)
    ds = f_h5.create_dataset('han', data_han.shape , dtype=dtype_han)
    ds[:] = data_han

    data_point = np.asarray(lst_points, dtype=object)
    dtype_point = h5py.special_dtype(vlen=np.int32)
    f_h5.create_dataset('points', data=data_point, dtype=dtype_point)

    data_label = np.asarray(lst_labels, dtype=object)
    dtype_label = h5py.special_dtype(vlen=np.int32)
    try:
        f_h5.create_dataset('labels', data=data_label, dtype=dtype_label)
    except:
        data_label = np.asarray(lst_labels, dtype=np.int32)
        f_h5.__delitem__('labels')
        f_h5.create_dataset('labels', data=data_label)

    f_h5.close()

    hdf = H5HanDataset(h5_filename)
    print(len(hdf), hdf[-2])
    

def dink2sorder(input_filename, h5_filename, output_filename):
    
    map_han = {}

    print('stat {}'.format(input_filename))
    with open(input_filename, 'r', encoding='utf-8') as file:
        num_lines = sum(1 for line in file)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    lst_han = []
    lst_labels = []
    lst_points = []

    f_d2 = open(input_filename, 'r', encoding='utf-8')
    f_d3 = open(output_filename, 'w', encoding='utf-8')
    line_total = 0
    valid_line_total = 0
    start_time = time.time()
    begin_time = start_time
    
    print('write {}'.format(output_filename))
    for each in f_d2:
        line_total += 1
        dink_data = json.loads(each)
        
        han = dink_data['han']

        compoments = dink_data['compoments']
        strokes = dink_data['strokes']

        for i in range(int(len(strokes))):
            valid_line_total += 1
            stroke_all_ids = list(range(len(strokes)))
            if i != 0:
                random.shuffle(stroke_all_ids)
            for j in range(len(strokes)):
                stroke_ids = stroke_all_ids[:j+1]
                points = []
                for nid, sid in enumerate(stroke_ids):
                    for pid, point in enumerate(strokes[sid]):
                        points.append([nid, pid, point['x'], point['y']])
                        # print(nid, sid, pid, point)

                out_ids = []
                for id in stroke_ids:
                    out_ids.append(id + 1)
                out_data = {}
                out_data['han'] = han
                out_data['labels'] = out_ids
                out_data['points'] = points
                point_items = []
                for point in points:
                    for item in point:
                        point_items.append(item)

                lst_han.append(han)
                lst_labels.append(np.asarray(out_ids, dtype=np.int32))
                lst_points.append(np.asarray(point_items, dtype=np.int32))
                f_d3.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))

        end_time = time.time()
        if line_total == 1 or end_time - begin_time > 6:
            begin_time = end_time
            date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
            print('{} time={:.1f}(m) valid={} line={:.2%} ({}/{}) label={}'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines, han))


    date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
    end_time = time.time()
    print('{} time={:.1f}(m) vaild={} line={:.2%} ({}/{})'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines))
    print('finished.')

    f_h5 = h5py.File(h5_filename, 'w')     

    data_han = np.array(lst_han)
    dtype_han = h5py.special_dtype(vlen=str)
    ds = f_h5.create_dataset('han', data_han.shape , dtype=dtype_han)
    ds[:] = data_han

    data_label = np.asarray(lst_labels, dtype=object)
    dtype_label = h5py.special_dtype(vlen=np.int32)
    try:
        f_h5.create_dataset('labels', data=data_label, dtype=dtype_label)
    except:
        data_label = np.asarray(lst_labels, dtype=np.int32)
        f_h5.__delitem__('labels')
        f_h5.create_dataset('labels', data=data_label)


    data_point = np.asarray(lst_points, dtype=object)
    dtype_point = h5py.special_dtype(vlen=np.int32)
    f_h5.create_dataset('points', data=data_point, dtype=dtype_point)
    
    f_h5.close()

    hdf = H5HanDataset(h5_filename)
    print(len(hdf), hdf[0])

def dink2stroke(han_filename, comp_filename, input_filename, h5_filename, output_filename):
    
    han_comp = HanComp(han_filename, comp_filename)

    print('stat {}'.format(input_filename))
    with open(input_filename, 'r', encoding='utf-8') as file:
        num_lines = sum(1 for line in file)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    lst_han = []
    lst_labels = []
    lst_points = []

    f_d2 = open(input_filename, 'r', encoding='utf-8')
    f_d3 = open(output_filename, 'w', encoding='utf-8')
    line_total = 0
    valid_line_total = 0
    start_time = time.time()
    begin_time = start_time
    
    print('write {}'.format(output_filename))
    for each in f_d2:
        line_total += 1
        dink_data = json.loads(each)
        
        han = dink_data['han']
        
        compoments = dink_data['compoments']
        strokes = dink_data['strokes']
        han_stroke = han_comp.get_han_stroke(han)

        for i in range(1):
            valid_line_total += 1
            stroke_ids = list(range(len(strokes)))
            # random.shuffle(stroke_ids)
            points = []
            for nid, sid in enumerate(stroke_ids):
                for pid, point in enumerate(strokes[sid]):
                    points.append([nid, pid, point['x'], point['y']])
                    # print(nid, sid, pid, point)

            han_stroke_ids = []
            for id in stroke_ids:
                han_stroke_ids.append(int(han_stroke[id]))
            out_data = {}
            out_data['han'] = han
            out_data['labels'] = han_stroke_ids
            out_data['points'] = points
            point_items = []
            for point in points:
                for item in point:
                    point_items.append(item)

            lst_han.append(han)
            lst_labels.append(np.asarray(han_stroke_ids, dtype=np.int32))
            lst_points.append(np.asarray(point_items, dtype=np.int32))
            f_d3.write('{}\n'.format(json.dumps(out_data, ensure_ascii=False)))

        end_time = time.time()
        if line_total == 1 or end_time - begin_time > 6:
            begin_time = end_time
            date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
            print('{} time={:.1f}(m) valid={} line={:.2%} ({}/{}) label={}'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines, han))


    date_str = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
    end_time = time.time()
    print('{} time={:.1f}(m) vaild={} line={:.2%} ({}/{})'.format(date_str, (end_time - start_time)/60, valid_line_total, line_total / num_lines, line_total, num_lines))
    print('finished.')

    f_h5 = h5py.File(h5_filename, 'w')     

    data_han = np.array(lst_han)
    dtype_han = h5py.special_dtype(vlen=str)
    ds = f_h5.create_dataset('han', data_han.shape , dtype=dtype_han)
    ds[:] = data_han

    data_label = np.asarray(lst_labels, dtype=object)
    dtype_label = h5py.special_dtype(vlen=np.int32)
    try:
        f_h5.create_dataset('labels', data=data_label, dtype=dtype_label)
    except:
        data_label = np.asarray(lst_labels, dtype=np.int32)
        f_h5.__delitem__('labels')
        f_h5.create_dataset('labels', data=data_label)

    data_point = np.asarray(lst_points, dtype=object)
    dtype_point = h5py.special_dtype(vlen=np.int32)
    f_h5.create_dataset('points', data=data_point, dtype=dtype_point)
    
    f_h5.close()

    hdf = H5HanDataset(h5_filename)
    print(len(hdf), hdf[0])



if __name__=='__main__':

    # args = parse_args()

    width = 256
    height = 256

    for prefix in ['palm_gbk']:
        for item in ['test', 'val', 'train']:
            dink_normalize('./data/jsonl/{}_{}.jsonl'.format(prefix, item), './data/dink/{}_256x256_{}.jsonl'.format(prefix, item), width, height)
            dink_add_comp('./labels/han.jsonl', './labels/comp.jsonl', './data/dink/{}_256x256_{}.jsonl'.format(prefix, item), './data/dink/{}_256x256_comp_{}.jsonl'.format(prefix, item))
            dink_comp_extr('./labels/han.jsonl', './labels/comp.jsonl', './data/dink/{}_256x256_comp_{}.jsonl'.format(prefix,item), './data/result/han_comp_extr_{}_{}.h5'.format(prefix,item), './data/result/han_comp_extr_{}_{}.jsonl'.format(prefix,item))    
            dink2stroke('./labels/han.jsonl', './labels/comp.jsonl', './data/dink/{}_256x256_comp_{}.jsonl'.format(prefix,item), './data/result/han_stroke_{}_{}.h5'.format(prefix, item), './data/result/han_stroke_{}_{}.jsonl'.format(prefix, item))
            # dink2sorder('./data/dink/{}_256x256_comp_{}.jsonl'.format(prefix, item), './data/result/han_sorder_{}_{}.h5'.format(prefix, item), './data/result/han_sorder_{}_{}.jsonl'.format(prefix, item))

    