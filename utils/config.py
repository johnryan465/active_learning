from dataclasses import fields
from typing import Any
from params.registered import Registered
from params.params import Params
import os.path
import json
import csv

class IO:
    def parseParams(cls : type, dic : dict) -> Any:
        kwds = {}
        for field in fields(cls):
            a = dic.get(field.name)
            if issubclass(field.type, Params):
                for option in Registered.types[field.type.name()]:
                    if option.name() == dic[field.name]['__name__']:
                        a = IO.parseParams(option, dic[field.name])

            kwds[field.name] = a

        return cls(**kwds)

    def create_directory(path : str) -> None:
        # remove the file name
        directory = "/".join(path.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def dict_to_file(dic : dict, file_name : str) -> None:
        IO.create_directory(file_name)
        with open(file_name, 'w') as outfile:
            json.dump(dic, outfile, indent=4)

    def load_dict_from_file(file_name : str) -> dict:
        with open(file_name) as json_file:
            data = json.load(json_file)
        return data

    def file_exists(file_name : str) -> bool:
        return os.path.isfile(file_name) 

    def dict_to_csv(dic : dict, file_name : str) -> None:
        IO.create_directory(file_name)
        keys = dic[0].keys()
        with open(file_name, 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(dic)