import json


class Recorder():
    def __init__(self, config):
        self.global_config = config
        self.meta_info = {}
        self.main_info = []

    def add_record(self, type, record):
        if type == 'META':
            self.meta_info.update(record)
        elif type == 'TRAJ':
            self.main_info.append(record)
        else:
            raise "Record Type Error!"

    def write_prompt_level(self):
        path = self.global_config['data_path']['record']
        output = {
            'meta_info': self.meta_info,
            'main_info': self.main_info
        }
        with open(path, 'w') as f:
            json.dump(output, f, ensure_ascii=False)
