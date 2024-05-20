import re
import warnings

from mechanism import *
from scipy import integrate
from utils import *
from Exception import *


class Agent():
    def __init__(self, config, aid, feature, llm):
        self.global_config = config
        self.config = config['agent_config']
        self.llm = llm

        # Static Part
        self.aid = aid
        self.name = feature['name']
        self.type = feature['type']
        self.profile = feature['profile']

        # Dynamic Part
        self.memory = self.config['default_score']['memory']
        self.emotion = self.config['default_score']['emotion']
        self.social_confidence = self.config['default_score']['social_confidence']
        self.opinion = self.config['default_score']['opinion']

    def recover_llm(self, llm):
        self.llm = llm

    def get_next_execute_time(self, current_time, social_media, simulation):
        alpha, beta, T = self.config['execute_gap_params']['alpha'], self.config['execute_gap_params']['beta'], \
                         self.config['execute_gap_params']['T']
        A = 0.5 + social_media.like_num / len(simulation.agents) + float(self.emotion)
        r = [0.0, 10.0]
        inter = integrate.quad(time_func, r[0], r[1], args=(alpha, A, T))[0]
        x = get_sample_prob(lambda x: time_func(x, alpha, A, T), r, inter)
        ed = timestring_to_timestamp(simulation.global_config['time_config']['end_time'])
        st = timestring_to_timestamp(simulation.global_config['time_config']['start_time'])
        beta = (ed - st) / 2 / simulation.global_config['time_config']['epoch_num']
        gap_time = int(x * beta / 10)
        if current_time <= st + gap_time <= ed:
            return st + gap_time
        else:
            return None

    def sensory_process(self, full_obs, sensory_info):
        sp_prompt = get_sp_prompt(self, full_obs)
        sensory_info['prompt'] = sp_prompt
        result = self.llm.fast_run(sp_prompt)
        sensory_info['response'] = result
        return result

    def decision_process(self, full_obs, impression, current_state, decision_info):
        action_info_dict = {}
        dp_prompt = get_dp_prompt(self, full_obs, impression, current_state)
        action_info_dict['prompt'] = dp_prompt
        result = self.llm.fast_run(dp_prompt)
        # print('(prompt of desicion result):', dp_prompt)
        print('(result):',result)
        action_info_dict['response'] = result
        if len(re.findall('\d', result)) == 0:
            e_tmp = LLMException('Agent-Decision-Fail-Action-Code', dp_prompt, result)
            warnings.warn(e_tmp.__str__())
            result = '0'
        else:
            result = re.findall('\d', result)[0]
        try:
            action = STATE_CODE_TO_ACTION[current_state][result]
        except Exception as e:
            warnings.warn(e.__str__())
            e_tmp = LLMException('Agent-Decision-Fail', dp_prompt, result)
            warnings.warn(e_tmp.__str__())
            action_info_dict['warning'] = '[FAIL]'
            if current_state == 'CommentState':
                return 'Back', '', get_action_info('Back', '')
            else:
                return 'Leave', '', get_action_info('Leave', '')

        extrares_info = {}
        if action in {'Comment', 'Reply'}:
            if action == 'Comment':
                comment_prompt = get_comment_prompt(self, impression)
                extrares_info['prompt'] = comment_prompt
                extra_response = self.llm.fast_run(comment_prompt)
                extrares_info['response'] = extra_response
            elif action == 'Reply':
                comment_prompt = get_reply_prompt(self, impression)
                extrares_info['prompt'] = comment_prompt
                extra_response = self.llm.fast_run(comment_prompt)
                extrares_info['response'] = extra_response
            else:
                raise "Error Occurs in DP of Agent."

        elif action in {'Detailed Comment'}:
            prompt = get_comment_id(self, impression)
            extrares_info['prompt'] = prompt
            result = self.llm.fast_run(prompt)
            extrares_info['response'] = result
            if result.isdigit():
                extra_response = int(result)
            else:
                e_tmp = LLMException('Detailed-Comment-Index-None-Int', prompt, result)
                extra_response = 0
        else:
            extra_response = ''
            extrares_info['response'] = ''

        action_info = get_action_info(action, extra_response)
        decision_info['action'] = action_info_dict
        decision_info['extra_response'] = extrares_info
        decision_info['action_info'] = action_info

        return action, extra_response, action_info

    def rp_parser(self, prompt_dict, result_dict):
        data = []

        if len(re.findall('\d+', result_dict['result_emotion'])) == 0:
            e_tmp = LLMException('Agent-Reflection-Fail-Emotion-Non-Format', prompt_dict['prompt_emotion'],
                                 result_dict['result_emotion'])
            warnings.warn(e_tmp.__str__())
            emotion = 0.5
        else:
            emotion = min(float(re.findall('\d+', result_dict['result_emotion'])[0]) / 100.0, 1.0)

        if len(re.findall('\d+', result_dict['result_socialconf'])) == 0:
            e_tmp = LLMException('Agent-Reflection-Fail-Socialconf-Non-Format', prompt_dict['prompt_socialconf'],
                                 result_dict['result_socialconf'])
            warnings.warn(e_tmp.__str__())
            socialconf = 0.5
        else:
            socialconf = min(float(re.findall('\d+', result_dict['result_socialconf'])[0]) / 100.0, 1.0)

        data.append(result_dict['result_summary'])
        data.append(result_dict['result_opinion'])
        data.append(emotion)
        data.append(socialconf)
        return data

    def reflection_process(self, full_obs, impression, action_info, reflection_info):
        rp_prompt_summary = get_rp_prompt_summary(self, impression, action_info)
        rp_prompt_opinion = get_rp_prompt_opinion(self, impression, action_info)
        rp_prompt_emotion = get_rp_prompt_emotion(self, impression, action_info)
        rp_prompt_socialconf = get_rp_prompt_socialconf(self, impression, action_info)
        reflection_info['prompt'] = {
            'prompt_summary': rp_prompt_summary,
            'prompt_opinion': rp_prompt_opinion,
            'prompt_emotion': rp_prompt_emotion,
            'prompt_socialconf': rp_prompt_socialconf
        }
        result_summary = self.llm.fast_run(rp_prompt_summary)
        print('(memory):',result_summary)
        result_opinion = self.llm.fast_run(rp_prompt_opinion)
        print('(opinion):',result_opinion)
        result_emotion = self.llm.fast_run(rp_prompt_emotion)
        print('(emotion):',result_emotion)
        result_socialconf = self.llm.fast_run(rp_prompt_socialconf)
        print('(sc):',result_socialconf)
        reflection_info['response'] = {
            'result_summary': result_summary,
            'result_opinion': result_opinion,
            'result_emotion': result_emotion,
            'result_socialconf': result_socialconf
        }
        reflection_info['previous_condition'] = {
            'memory': self.memory,
            'emotion': self.emotion,
            'social_confidence': self.social_confidence,
            'opinion': self.opinion
        }
        try:
            self.memory, self.opinion, self.emotion, self.social_confidence = self.rp_parser(reflection_info['prompt'],
                                                                                             reflection_info[
                                                                                                 'response'])
            reflection_info['after_condition'] = {
                'memory': self.memory,
                'emotion': self.emotion,
                'social_confidence': self.social_confidence,
                'opinion': self.opinion
            }
        except LLMException as e:
            reflection_info['after'] = '[LLM_EXCEPTION]'
            warnings.warn(e.__str__())

    def take_action(self, full_obs, current_state, agent_action_info):
        sensory_info = {}
        impression = self.sensory_process(full_obs, sensory_info)
        print('(impression):',impression)
        decision_info = {}
        action, text_response, action_info = self.decision_process(full_obs, impression, current_state, decision_info)
        reflection_info = {}
        self.reflection_process(full_obs, impression, action_info, reflection_info)
        agent_action_info['sensory_info'] = sensory_info
        agent_action_info['decision_info'] = decision_info
        agent_action_info['reflection_info'] = reflection_info
        return action, text_response, action_info
