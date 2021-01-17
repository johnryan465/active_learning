from dataclasses import fields
from params.registered import Registered
from params.params import Params


class IO:
    def parseParams(cls : type, dic : dict) -> 'cls':
        kwds = {}
        for field in fields(cls):
            a = dic.get(field.name)
            if issubclass(field.type, Params):
                for option in Registered.types[field.type.name()]:
                    if option.name() == dic[field.name]['__name__']:
                        a = IO.parseParams(option, dic[field.name])

            kwds[field.name] = a

        return cls(**kwds)

    def dict_to_file(dic : dict, file_name : str) -> None:
        pass

    def load_dict_from_file(file_name : str) -> dict:
        pass
