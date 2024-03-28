import os
import sys
import json
import time

class HanComp:
    def __init__(self, han_filename, comp_filename, print_log = True):
        self.print_log = print_log
        self.map_han = {}
        self.map_han_id = {}
        
        self.map_comp = {}
        self.map_comp_id = {}
        self.stroke_max_num = 0
        self.comp_max_num = 0
        self.load_han(han_filename, self.map_han, self.map_han_id)
        self.load_comp(comp_filename, self.map_comp, self.map_comp_id)

    def load_han(self, filename, map_han, map_han_id):
        if self.print_log:
            print('load', filename)
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                id = len(map_han)
                data['id'] = id
                map_han[data['han']] = data
                map_han_id[id] = data['han']
                if self.stroke_max_num < len(data['stroke']):
                    self.stroke_max_num = len(data['stroke'])
                if self.comp_max_num < len(data['component']):
                    self.comp_max_num = len(data['component'])

            return True
        return False

    def load_comp(self, filename, map_comp, map_comp_id):
        if self.print_log:
            print('load', filename)
 
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                id = len(map_comp)
                data['id'] = id
                map_comp[data['component']] = data
                map_comp_id[id] = data['component']
            return True
        return False
    
    def get_han_comp(self, han):
        if han in self.map_han:
            return self.map_han[han]['component']
        return []

    def get_comp_id(self, comp):
        if comp in self.map_comp:
            return self.map_comp[comp]['id']
        return -1

    def get_comp_total(self):
        return len(self.map_comp)

    def get_stroke_total(self):
        return 5
    
    def get_comp_name(self, id):
        return self.map_comp_id.get(id, '')

    def get_comp_max_num(self):
        return self.comp_max_num
    
    def get_comp_stroke_num(self, comp):
        if comp in self.map_comp:
            return len(self.map_comp[comp]['stroke'])
        return -1

    def get_han_total(self):
        return len(self.map_han)
    
    def get_han_stroke_num(self, han):
        if han in self.map_han:
            return len(self.map_han[han]['stroke'])
        return -1
    
    def get_han_stroke(self, han):
        if han in self.map_han:
            return self.map_han[han]['stroke']
        return ''

    def get_stroke_max_num(self):
        return self.stroke_max_num
    
    def get_han_name(self, id):
        return self.map_han_id.get(id, '')
    
    def get_han_id(self, han):
        if han in self.map_han:
            return self.map_han[han]['id']
        return -1


if __name__=='__main__':

    han_comp = HanComp('./labels/han.jsonl', './labels/comp.jsonl')
    
    map_add_han = set()
    for i in range(han_comp.get_comp_total()):
        comp = han_comp.get_comp_name(i)
        han = comp[0]
        if han_comp.get_han_id(han) == -1:
            map_add_han.add(han)
            
    print(len(map_add_han))

    
