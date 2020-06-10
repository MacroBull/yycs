# yycs
Yum Yaml Config System

YUMMY! Config from YAML file and then override with command line is so easy!

# Usage
Load and parse config with the following:
```python
from yycs import get_config
config = get_config()
```

By launch your program with:
```shell
python main.py -c config.yaml scope.key=value
```

You will get `config` as:
```python
config(config='config.yaml', default='configs/default.yaml', debug=False, ...(things from config files), scope=scope(key='value'))
```
