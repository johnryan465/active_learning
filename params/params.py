from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod
from typing import List
import json
from dataclasses import fields, asdict

class Params(ABC):
    def export(self) -> 'cls':
        kwds = {"__name__": self.name()}
        cls = self.__class__
        dic = self.__dict__
        for field in fields(cls):
            a = dic.get(field.name)
            if issubclass(field.type, Params):
                a = a.export()

            kwds[field.name] = a
        
        return kwds

    @classmethod
    def options(cls : type) -> List[type]:
        return [cls]
    
    @classmethod
    def name(cls : type) -> str:
        return cls.__name__

    
