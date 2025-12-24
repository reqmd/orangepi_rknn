import yaml

def save_yaml(name = 'config.yaml', params = None):
    with open(name, 'w') as file:
        yaml.dump(params, file, default_flow_style=False, sort_keys=False)

def load_yaml(name = 'config.yaml'):
    with open(name, 'r') as file:
        return yaml.safe_load(file)