import json
import time

import openai
import requests

from zhipuai import ZhipuAI

LLMMODEL_TO_TAG = {
    'ERNIE_Bot': 'completions',
    'ERNIE_Bot_turbo': 'eb-instant'
}


def set_openai_key(key):
    openai.api_key = key


def create_LLM(config):
    if config['LLM_config']['model_name'] in {'gpt-3.5-turbo'}:
        return GPT_LLM(config)
    if config['LLM_config']['model_name'] in {'ERNIE_Bot', 'ERNIE_Bot_turbo'}:
        return ERNIE_LLM(config)
    if config['LLM_config']['model_name'] in {'chatglm2-6b', 'Baichuan2-7B-Chat'}:
        return Local_LLM(config)
    if config['LLM_config']['model_name'] in {'glm-3-turbo', 'glm-4'}:
        return ZhipuLLM(config)


class GPT_LLM():
    def __init__(self, config):
        self.global_config = config
        self.config = config['LLM_config']

        self.key_list = self.config['API_KEYs']

        self.call_time = 0

        self.total_consumption = 0
        self.input_consumption = 0
        self.ouput_consumption = 0

    def parse_response(self, response):
        return {
            'run_id': response['id'],
            'time_stamp': response['created'],
            'result': response['choices'][0]['message']['content'],
            'input_token': response['usage']['prompt_tokens'],
            'output_token': response['usage']['completion_tokens'],
            'total_token': response['usage']['total_tokens']
        }

    def run(self, message_list, temperature=1.0, penalty_score=0.0):
        time.sleep(2)
        set_openai_key(self.key_list[self.call_time % len(self.key_list)])
        response = openai.ChatCompletion.create(
            model=self.config['model_name'],
            messages=message_list,
            temperature=temperature,
            frequency_penalty=penalty_score,
            presence_penalty=penalty_score
        )
        response = self.parse_response(response)
        self.add_consumption(response)
        self.call_time += 1

        return response

    def fast_run(self, query, temperature=1.0, penalty_score=0.0):
        response = self.run([{"role": "user", "content": query}], temperature, penalty_score)
        return response['result']

    def add_consumption(self, response):
        self.input_consumption += response['input_token']
        self.ouput_consumption += response['output_token']
        self.total_consumption += response['total_token']

    def print_total_consumption(self):
        if self.config['model_name'] == 'gpt-3.5-turbo':
            print('The input consumption is %d tokens, output consumption is %s, which is %f CNY.' % (
                self.input_consumption, self.ouput_consumption,
                (0.0015 * self.input_consumption + 0.002 * self.ouput_consumption) / 1000 * 8))


class ERNIE_LLM():
    def __init__(self, config):
        self.global_config = config
        self.config = config['LLM_config']

        if not self.config['access_token']:
            self.config['access_token'] = self.get_access_token()

        self.total_consumption = 0
        self.input_consumption = 0
        self.ouput_consumption = 0

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.config['API_KEY'],
                  "client_secret": self.config['SECRET_KEY']}
        return str(requests.post(url, params=params).json().get("access_token"))

    def print_total_consumption(self):
        if self.config['model_name'] == 'ERNIE_Bot':
            print('The total consumption is %d tokens, which is %f CNY.' % (
                self.total_consumption, self.total_consumption * 0.012 / 1000))
        if self.config['model_name'] == 'ERNIE_Bot_turbo':
            print('The input consumption is %d tokens, output consumption is %s, which is %f CNY.' % (
                self.input_consumption, self.ouput_consumption, self.total_consumption * 0.016 / 1000))

    def add_consumption(self, response):
        self.input_consumption += response['input_token']
        self.ouput_consumption += response['output_token']
        self.total_consumption += response['total_token']

    def parse_response(self, response):
        return {
            'run_id': response['id'],
            'time_stamp': response['created'],
            'result': response['result'],
            'risk': response['need_clear_history'],
            'input_token': response['usage']['prompt_tokens'],
            'output_token': response['usage']['completion_tokens'],
            'total_token': response['usage']['total_tokens']
        }

    def run(self, message_list, temperature=0.95, penalty_score=1.0):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/%s?access_token=" % LLMMODEL_TO_TAG[
            self.config['model_name']] + self.config['access_token']
        headers = {
            'Content-Type': 'application/json'
        }
        data = json.dumps({
            "messages": message_list,
            "temperature": temperature,
            "penalty_score": penalty_score
        })

        response = requests.request("POST", url, headers=headers, data=data).json()

        response = self.parse_response(response)
        self.add_consumption(response)

        return response

    def fast_run(self, query, temperature=0.95, penalty_score=1.0):
        response = self.run([{"role": "user", "content": query}], temperature, penalty_score)
        return response['result']

class ZhipuLLM():
    def __init__(self, config):
        self.global_config = config
        self.config = config['LLM_config']

        self.client = ZhipuAI(api_key=self.config['API_KEY'])

        self.call_time = 0

    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': response.choices[0].message.content
        }

    def run(self, message_list, temperature=1.0, penalty_score=0.0):
        response = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=message_list,
            temperature=temperature,
        )
        response = self.parse_response(response)
        self.call_time += 1

        return response

    def fast_run(self, query, temperature=0.95, penalty_score=1.0, exception_times = 10):
        response = None
        for et in range(exception_times):
            try:
                response = self.run([{"role": "user", "content": query}], temperature, penalty_score)
                break
            except Exception as e:
                print('[%d] LLM inference fails: %s' % (et,e))
        if not response:
            return ' '

        return response['result']
    
    def __deepcopy__(self, memo):
        return None

class Local_LLM():
    def __init__(self, config):
        self.global_config = config
        self.config = config['LLM_config']

        self.model_path = self.config['model_path']
        self.port = self.config['port']

        self.call_time = 0

        self.total_consumption = 0
        self.input_consumption = 0
        self.ouput_consumption = 0

    def print_total_consumption(self):
        print('The input consumption is %d tokens, output consumption is %s, which is 0.0 CNY.' % (
            self.input_consumption, self.ouput_consumption))

    def add_consumption(self, response):
        self.input_consumption += response['input_token']
        self.ouput_consumption += response['output_token']
        self.total_consumption += response['total_token']

    def parse_response(self, response):
        return {
            'time_stamp': response['created'],
            'result': response['choices'][0]['message']['content'],
            'input_token': 0,
            'output_token': 0,
            'total_token': 0
        }

    def run(self, message_list, temperature=1.0, penalty_score=0.0):
        openai.api_base = "http://localhost:%d/v1" % self.port
        openai.api_key = "none"

        response = openai.ChatCompletion.create(
            model=self.config['model_name'],
            messages=message_list,
        )

        response = self.parse_response(response)
        self.call_time += 1
        self.add_consumption(response)

        return response

    def fast_run(self, query, temperature=0.95, penalty_score=1.0):
        response = self.run([{"role": "user", "content": query}], temperature, penalty_score)
        return response['result']
