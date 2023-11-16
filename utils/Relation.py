import json
IMAGE_ID = "image_id"
RELATIONSHIPS = "relationships"
PREDICATE = "predicate"
RELATIONSHIP_ID = "relationship_id"
HEIGHT = "h"
WIDTH = "w"
X_CORD = "x"
Y_CORD = "y"
NAMES = "names"
NAME = "name"
PICK_NAME_IDX = 0
OBJECT_ID = "object_id"
SYNSETS = "synsets"
OBJECT = "object"
SUBJECT = "subject"

class Object:
    def __init__(self, object):
        self.height = object[HEIGHT]
        self.width = object[WIDTH]
        self.x = object[X_CORD]
        self.y = object[Y_CORD]
        self.object_id = object[OBJECT_ID]
        self.sysnets = object[SYNSETS]
        if NAMES in object:
            self.name = object[NAMES][PICK_NAME_IDX]
        else:
            self.name = object[NAME]

class Subject:
    def __init__(self, object):
        self.height = object[HEIGHT]
        self.width = object[WIDTH]
        self.x = object[X_CORD]
        self.y = object[Y_CORD]
        self.object_id = object[OBJECT_ID]
        self.sysnets = object[SYNSETS]
        if NAMES in object:
            self.name = object[NAMES][PICK_NAME_IDX]
        else:
            self.name = object[NAME]

class Relation:
    ## relationships: directly read json file 
    ## image_id: the image idx, start from 1, use to access image file
    ## idx_in_json, index of a single action in this json file, start with 0
    def __init__(self, relationships, image_id, idx_in_json):
        dict = relationships["relationships"][idx_in_json]
        self.image_id = image_id
        self.idx_in_json = idx_in_json
        self.relationship_id = dict[RELATIONSHIP_ID]
        self.predicate = dict[PREDICATE]
        self.object = Object(dict[OBJECT])
        self.subject = Object(dict[SUBJECT])
    
    def get_sub_location(self):
        x, y = self.subject.x, self.subject.y
        w, h = self.subject.width, self.subject.height
        return x, y, w, h
    
    def get_obj_location(self):
        x, y = self.object.x, self.object.y
        w, h = self.object.width, self.object.height
        return x, y, w, h
    
    def get_predicate(self):
        return self.predicate

    def get_idx_in_json(self):
        return self.idx_in_json
    
    def get_obj_name(self):
        return self.object.name
    
    def get_sub_name(self):
        return self.subject.name
    
    def set_sub_name(self, name):
        self.subject.name = name
    
    def set_obj_name(self, name):
        self.object.name = name
