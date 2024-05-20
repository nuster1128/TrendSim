import csv
import datetime
import heapq
import pickle
import random
import time
from copy import deepcopy
from LLM import create_LLM
import os

import numpy as np

DATA_AND_BASELINE_CODE = {
    'mild': {
        'clean': (1000, 0, 0, 0),
        'full': (900, 33, 33, 34)
    },
    'medium': {
        'clean': (1000, 0, 0, 0),
        'full': (700, 100, 100, 100)
    },
    'severe': {
        'clean': (1000, 0, 0, 0),
        'full': (500, 166, 167, 167)
    },
    'debug': {
        'clean': (1000, 0, 0, 0),
        'full': (700,100,100,100)
    }
}

AGENT_CLASS = {
    'normal': 'Agent',
    'S': 'SocialAttacker',
    'T': 'TrollingAttacker',
    'F': 'FactAttacker'
}


def update_config(cfg, extra_args):
    if len(extra_args) % 2 != 0:
        raise "Command argparse number is not correct!"
    for i in range(0, len(extra_args), 2):
        name, par = extra_args[i], extra_args[i + 1]
        if name == '-port':
            cfg['LLM_config']['port'] = int(par)
        if name == '-tweet_index':
            cfg['media_config']['tweet_index'] = int(par)
        if name == '-degree':
            cfg['simulation_config']['degree'] = par
        if name == '-baseline':
            cfg['simulation_config']['baseline'] = par
        if name == '-record_path':
            cfg['data_path']['record'] = par
        if name == '-load_checkpoint':
            cfg['data_path']['load_checkpoint'] = par
        if name == '-save_checkpoint':
            cfg['data_path']['save_checkpoint'] = par
        if name == '-defense_time':
            cfg['intervention']['comment_poisoning_detect']['split_time'] = float(par)
    return cfg


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(obj, path):
    print(obj)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def binary_search_insert(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left


def load_user(config):
    user_path = config['data_path']['normal_user']
    degree = config['simulation_config']['degree']
    baseline = config['simulation_config']['baseline']

    code = DATA_AND_BASELINE_CODE[degree][baseline]

    normal_user_list = load_normal_user(user_path, code)
    attacker_list = load_attacker(code)

    all_user = [u for u in attacker_list]
    for u in normal_user_list:
        all_user.append(u)
    return all_user


def load_normal_user(path, num_tuple):
    user_list = []
    with open(path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for index, line in enumerate(reader):
            if index >= num_tuple[0]:
                break
            user_list.append({
                'type': 'normal',
                'name': line[2],
                'profile': {
                    'uid': line[1],
                    'followed_num': line[3],
                    'following_num': line[4],
                    'description': line[5]
                }
            })

    return user_list


def load_attacker(num_tuple):
    attacker_list = []
    for i in range(num_tuple[1]):
        attacker_list.append({
            'type': 'S',
            'name': '社会先锋%d' % i,
            'profile': {
                'uid': str(num_tuple[0] + i),
                'followed_num': str(0),
                'following_num': str(0),
                'description': '这个社会太黑暗了！贫富差距太大！打工人不配活着！'
            }
        })

    for i in range(num_tuple[2]):
        attacker_list.append({
            'type': 'T',
            'name': '狂热粉丝%d' % i,
            'profile': {
                'uid': str(num_tuple[0] + num_tuple[1] + i),
                'followed_num': str(0),
                'following_num': str(0),
                'description': '只有坤坤是最棒的！表演天赋最佳！其他人都太弱了！'
            }
        })

    for i in range(num_tuple[3]):
        attacker_list.append({
            'type': 'F',
            'name': '小道预言家%d' % i,
            'profile': {
                'uid': str(num_tuple[0] + num_tuple[1] + num_tuple[2] + i),
                'followed_num': str(0),
                'following_num': str(0),
                'description': '我有一个朋友和当事人很熟！他知道内情！'
            }
        })
    return attacker_list


def load_tweet(config):
    tweet_list = []
    with open(config['data_path']['tweets'], 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for line in reader:
            tweet_list.append({
                'title': line[1],
                'brief_intro': line[2],
                'content': line[3],
                'source': line[4],
                'publish_time': line[5]
            })
    return tweet_list[config['media_config']['tweet_index']]


class Event():
    def __init__(self, time, info):
        self.time = time
        self.info = info

    def __lt__(self, other):
        if isinstance(other, Event):
            return self.time < other.time


class Heap():
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        if len(self.heap) <= 0:
            raise "Heap already empty but requires pop."
        return heapq.heappop(self.heap)

    def top(self):
        return heapq.nsmallest(1, self.heap)[0]

    def empty(self):
        return len(self.heap) == 0

    def print(self):
        print(self.heap)


class SortedList():
    def __init__(self, classType):
        self.loc = []
        self.sorted = []
        self.classType = classType

    def __len__(self):
        return len(self.sorted)

    def append(self, cid, text):
        self.loc.append(len(self.sorted))
        self.sorted.append(self.classType(cid, text))

    def add_like(self, cid):
        try :
            location = self.loc[cid]
            self.sorted[location].like_num += 1
            self.update_since(location)
        except Exception as e:
            print('Add like fails: %s' % e)

    def update_since(self, location):
        cmt = self.sorted[location]
        cid = cmt.cid
        like_num = cmt.like_num
        pt = location - 1
        while pt >= 0 and self.sorted[pt].like_num < like_num:
            other_cid = self.sorted[pt].cid
            self.loc[other_cid], self.loc[cid] = self.loc[cid], self.loc[other_cid]
            self.sorted[location], self.sorted[pt] = self.sorted[pt], self.sorted[location]
            location = pt
            pt = location - 1


def timestring_to_timestamp(time_string):
    return datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S').timestamp()


def timestamp_to_timestring(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)


def get_sample_prob(f, value_range, inter, num=101):
    delta = (value_range[1] - value_range[0]) / (num - 1)
    x_list = np.linspace(value_range[0], value_range[1], num)
    prob_list = [f(x) / inter for x in x_list]
    cdf_list = [0.0]
    for i in range(1, num):
        cdf_list.append(prob_list[i - 1] * delta + cdf_list[i - 1])
    y = random.random()
    idx = binary_search_insert(cdf_list, y)
    return idx / (num - 1) * (value_range[1] - value_range[0])


class CheckPoint():
    def __init__(self, config, start_time):
        self.config = config
        self.start_time = start_time

        self.checkpoint_time = None
        self.simulator = None

    def save_checkpoint(self, simulator):
        
        pass
        # self.simulator = deepcopy(simulator)
        # self.checkpoint_time = time.time()
        
        # write_pickle(self, self.config['data_path']['save_checkpoint'])

    def load_checkpoint(self):
        # if not os.path.exists(self.config['data_path']['load_checkpoint']):
        #     return

        # cp = read_pickle(self.config['data_path']['load_checkpoint'])
        # llm = create_LLM(self.config)
        # self.simulator.llm = llm
        # self.simulator.social_media.llm = llm
        # for a in self.simulator.agents:
        #     a.llm = llm

        # self.config, self.simulator = cp.config, cp.simulator
        # self.start_time, self.checkpoint_time = cp.start_time, cp.checkpoint_time
        pass

