#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Fri Aug  9 11:20:32 2019

@author: Macrobull
"""

from __future__ import absolute_import, division, unicode_literals

import argparse, logging, shutil
import yaml

from collections import namedtuple
from functools import wraps
from pathlib import Path
from yaml import Node
from yaml.constructor import BaseConstructor, SafeConstructor


ConfigDict       :type = 'Mapping[str, Any]'
ConfigTransform  :type = 'Callable[[ConfigDict, ...], ConfigDict]'
ConfigTransforms :type = 'Collection[ConfigTransform]'

DELIMITER_ARG  :str = '='
DELIMITER_PATH :str = '.'

PROGRAM_NAME           :str = shutil.os.path.basename(shutil.sys.argv[0])
DEFAULT_LOGGING_FORMAT :str = (
        '[%(levelname)8s](%(asctime)8s)%(name)s::%(funcName)s:%(lineno)4d: %(message)s')

state = dict()


def is_listy(obj:'Any')->bool:
    r"""is `obj` 'list'-like"""

    return isinstance(obj, (tuple, list, set))


def is_dict(obj:'Any')->bool:
    r"""is `obj` 'dict'"""

    return isinstance(obj, dict)


def recurse(func:'Callable', obj:'Any',
            with_list:bool=True, with_dict:bool=True)->'Any':
    r"""recursive map"""

    if with_list and is_listy(obj):
        return type(obj)(recurse(func, item, with_dict=with_dict) for item in obj)
    if with_dict and is_dict(obj):
        return type(obj)({k: recurse(func, v, with_list=with_list) for k, v in obj.items()})
    return func(obj)


def get_func_name(func:'Callable')->str:
    r"""get name of `func`, a 'function', 'partial' or 'callable' class"""

    return getattr(func, '__name__', getattr(getattr(func, 'func', type(func)), '__name__'))


def dict2obj(d:'Mapping[str, Any]',
             cls_name:str='Namespace')->'Any':
    r"""
        'dict' to read-only fields recursively
        if mutable keys wanted, try EasyDict
    """

    d = d.copy()
    for k, v in d.items():
        if is_dict(v):
            d[k] = dict2obj(v, cls_name=k)
    cls = namedtuple(cls_name, d.keys())
    return cls(**d)


def parse_args(
        args:'Optional[Sequence[str]]'=None,
        default_config:str='configs/default')->argparse.Namespace:
    r"""the default parse_args"""

    parser = argparse.ArgumentParser(
            description=PROGRAM_NAME,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
    parser.add_argument(
            '--config', '-c', default=f'{default_config}.yaml',
            help='path to config.yaml',
            )
    parser.add_argument(
            '--default', '-f', default=f'{default_config}.yaml',
            help='path to fallback default.yaml',
            )
    parser.add_argument(
            '--debug', '-d', action='store_true',
            help='enable debug logging and checking',
            )
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(args)
    return args


class ConfigConstructor(SafeConstructor):
    r"""YAML Constructor extension overload"""


class ConfigLoader(
        yaml.reader.Reader, yaml.scanner.Scanner, yaml.parser.Parser, yaml.composer.Composer,
        ConfigConstructor,
        yaml.resolver.Resolver):
    r"""YAML Loader for config"""

    def __init__(self, stream):
        yaml.reader.Reader.__init__(self, stream)
        yaml.scanner.Scanner.__init__(self)
        yaml.parser.Parser.__init__(self)
        yaml.composer.Composer.__init__(self)
        ConfigConstructor.__init__(self)
        yaml.resolver.Resolver.__init__(self)


def construct_path(constructor:BaseConstructor, node:Node)->Path:
    r"""construct path scalar"""

    return Path(constructor.construct_scalar(node)).expanduser().resolve()


def construct_tensor(
        constructor:BaseConstructor, node:Node,
        tensor_cls:type=list)->'Any':
    r"""construct tensor scalar"""

    return tensor_cls(yaml.load(constructor.construct_scalar(node), Loader=ConfigLoader))


ConfigConstructor.add_constructor('!path', construct_path)
ConfigConstructor.add_constructor('!tensor', construct_tensor)


def on_path(func:ConfigTransform, path:str,
            *args, **kwargs)->ConfigTransform:
    r"""call `func` at `path` in `config`"""

    @wraps(func)
    def _wrap(config, **kwargs):
        config = config.copy()
        obj, po, pk = config, None, None
        if path:
            for key in path.split(DELIMITER_PATH):
                try:
                    if is_listy(obj):
                        key = int(key) # throw ValueError
                    if is_dict(obj[key]): # throw KeyError, IndexError
                        obj[key] = obj[key].copy()
                    po, pk = obj, key
                    obj = obj[key]
                except (ValueError, KeyError, IndexError):
                    logger = state.get('logger', None)
                    if logger is not None:
                        logger.warning('bad index %s for %s, transform skipped',
                                       key, obj)
                    return config
        ret = func(obj, *args, **kwargs)
        if po is None:
            config = ret
        else:
            po[pk] = ret
        return config

    func_name = get_func_name(func)
    _wrap.__name__ = f'{func_name}(@path={path})'
    return _wrap


def setup_global_logging(
        config:ConfigDict,
        *args, **kwargs)->ConfigDict:
    r"""setup global logging level according to "debug" option in `config`"""

    try:
        import tqdm_color_logging # throw ImportError
    except ImportError:
        pass

    logging_format = DEFAULT_LOGGING_FORMAT
    logging_level = logging.DEBUG if config['debug'] else logging.INFO
    logging.basicConfig(format=logging_format, level=logging_level)

    state['logger'] = logging.getLogger(__name__)

    return config


def mkdirs(config:ConfigDict,
           *args, **kwargs)->ConfigDict:
    r"""create directories for 'Path's in `config`"""

    def _inner(d):
        for k, v in d.items():
            if is_dict(v):
                _inner(v)
                continue
            if isinstance(v, (Path, str)) and k.endswith('_dir'):
                v = Path(v)
                if v.is_absolute(): # only resolved
                    shutil.os.makedirs(v, exist_ok=True)

    _inner(config)
    return config


def convert_rel_path(
        config:ConfigDict,
        *args,
        base_key:str='train_dir',
        **kwargs)->ConfigDict:
    r"""convert 'Path's in `config` if it's relative(with _rel_)"""

    base_dir = config[base_key]
    assert isinstance(base_dir, Path), f'base_dir({base_dir}) is not a Path'

    def _inner(d):
        r = type(d)()
        for k, v in d.items():
            if is_dict(v):
                r[k] = _inner(v)
                continue
            if k.count('_rel_') != 1 or not isinstance(v, (Path, str)):
                r[k] = v
                continue
            v_ = Path(v)
            if v_.is_absolute():
                r[k] = v
                continue
            r[k.replace('_rel_', '_')] = base_dir / v
        return r

    return _inner(config)


def resolve_args(
        args:argparse.Namespace,
        tfms:ConfigTransforms=(setup_global_logging, ),
        config_cls:'Optional[type]'=None,
        )->'Uniont[config, EasyDict]':
    r"""
        load, resolve and merge from YAML config file
        `tfms`(transforms): convert_rel_path, mkdirs, setup_global_logging ...
        `config_cls` can be other types like EasyDict to be used instead of namedtuple as config
    """

    def _recurse_update(d, s):
        for k, v in s.items():
            if is_dict(v):
                v_ = d.get(k, None)
                if is_dict(v_):
                    _recurse_update(v_, v)
                    continue
            d[k] = v

    config = dict()
    if args.default and shutil.os.path.isfile(args.default):
        config = yaml.load(open(args.default), Loader=ConfigLoader) or dict()

    config_ = yaml.load(open(args.config), Loader=ConfigLoader) or dict()
    _recurse_update(config, config_)

    for k, v in args.__dict__.items():
        if k in ('args', ):
            continue
        config[k] = v

    for arg in args.args:
        path, _, value = arg.partition(DELIMITER_ARG)
        value = yaml.load(value, Loader=ConfigLoader) or True # HINT: true for switch-on
        keys = path.split('.')
        obj = config
        for key in keys[:-1]:
            try:
                if is_listy(obj):
                    obj = obj[int(key)] # throw ValueError, IndexError
                else:
                    obj_ = obj
                    obj = obj_.get(key, None) # throw AttributeError
                    if obj is None:
                        obj = obj_[key] = dict() # throw TypeError
            except (ValueError, IndexError, AttributeError, TypeError):
                raise ValueError(f'bad index {key} for {obj} in arg {path}')
        key = keys[-1]
        try:
            if is_listy(obj):
                obj[int(key)] = value # throw ValueError
            else:
                obj[key] = value # throw TypeError
        except (ValueError, TypeError):
            raise ValueError(f'bad index {key} for {obj} in arg {path}')

    for tfm in tfms:
        config = tfm(config)

    if config_cls is None:
        return dict2obj(config, 'config')

    return config_cls(config)


def get_transforms()->ConfigTransforms:
    r"""get the default tfms for config"""

    return (setup_global_logging,
            on_path(convert_rel_path, 'path'),
            on_path(mkdirs, 'path'),
            )


def get_config(
        tfms:'Optional[ConfigTransforms]'=None,
        **kwargs)->'config':
    r"""get the default config"""

    if tfms is None:
        tfms = (setup_global_logging, )

    args = parse_args()
    config = resolve_args(args, tfms=tfms, **kwargs)
    return config


if __name__ == '__main__':
    from easydict import EasyDict

    config = get_config(config_cls=EasyDict)
    print(config)
