import re

from mechanism import *
from utils import *
from Attacker import SocialAttacker, TrollingAttacker, FactAttacker
from Agent import Agent


class TimeSystem():
    def __init__(self, config):
        self.global_config = config
        self.config = config['time_config']

        self.start_timestamp = timestring_to_timestamp(config['time_config']['start_time'])
        self.end_timestamp = timestring_to_timestamp(config['time_config']['end_time'])
        self.priority_queue = Heap()

    def is_finish(self):
        if self.priority_queue.empty():
            return True
        if self.priority_queue.top().time > self.end_timestamp:
            return True
        return False

    def add_event(self, time, info):
        self.priority_queue.push(Event(time, info))

    def execute_event(self):
        event = self.priority_queue.pop()
        return event.time, event.info


class Reply():
    def __init__(self, rid, cid, reply_text):
        self.rid = rid
        self.cid = cid
        self.reply_text = reply_text

    def __str__(self):
        return self.reply_text


class Comment():
    def __init__(self, cid, comment_text):
        self.cid = cid
        self.comment_text = comment_text

        self.like_num = 0
        self.reply_list = []

    def __str__(self):
        return self.comment_text

    def __lt__(self, other):
        if isinstance(other, Comment):
            return self.like_num < other.like_num


class SocialMedia():
    def __init__(self, config, llm):
        self.global_config = config
        self.config = config['media_config']
        self.llm = llm

        self.tweet_info = load_tweet(config)
        self.comment_list = SortedList(Comment)
        self.detect_list = []

        self.view_num = 0
        self.like_num = 0
        self.repost_num = 0
    
    def recover_llm(self, llm):
        self.llm = llm

    def print_comment(self):
        for comment in self.comment_list.sorted:
            print('[%d] %s' % (comment.cid, comment.comment_text))

    def action_response(self, action, text_response, action_info, current_comment_id, current_time):
        if action == 'More Comments':
            action = 'More_Comments'
        if action == 'Detailed Comment':
            action = 'Detailed_Comment'
        return eval('self.%s_action' % action)(text_response, action_info, current_comment_id, current_time)

    def Leave_action(self, text_response, action_info, current_comment_id, current_time):
        return 'EndState'

    def Enter_action(self, text_response, action_info, current_comment_id, current_time):
        self.view_num += 1
        return 'MainState'

    def Like_tweet_action(self, text_response, action_info, current_comment_id, current_time):
        self.like_num += 1
        return 'MainState'

    def Comment_action(self, text_response, action_info, current_comment_id, current_time):
        split_time = self.global_config['intervention']['comment_poisoning_detect']['split_time']
        threshold = (1.0 - split_time) * timestring_to_timestamp(
            self.global_config['time_config']['start_time']) + split_time * \
                    timestring_to_timestamp(self.global_config['time_config']['end_time'])
        if self.global_config['intervention']['comment_poisoning_detect'][
            'type'] == 'LLM' and current_time >= threshold:
            prompt = '评论: %s\n' % text_response
            prompt += '请你判断上面这句评论有无恶意信息。如果没有，请输出0；如果有，请输出1。输出只能包含一个整数。'
            result = self.llm.fast_run(prompt)
            code_list = re.findall('\d', result)
            if len(code_list) == 0:
                self.comment_list.append(len(self.comment_list), text_response)
            elif code_list[0] == '1':
                self.detect_list.append(text_response)
            else:
                self.comment_list.append(len(self.comment_list), text_response)
        else:
            self.comment_list.append(len(self.comment_list), text_response)

        return 'MainState'

    def Repost_action(self, text_response, action_info, current_comment_id, current_time):
        self.repost_num += 1
        return 'MainState'

    def More_Comments_action(self, text_response, action_info, current_comment_id, current_time):
        return 'MainState'

    def Detailed_Comment_action(self, text_response, action_info, current_comment_id, current_time):
        return 'CommentState'

    def Like_comment_action(self, text_response, action_info, current_comment_id, current_time):
        self.comment_list.add_like(current_comment_id)
        return 'CommentState'

    def Reply_action(self, text_response, action_info, current_comment_id, current_time):
        try :
            self.comment_list.sorted[current_comment_id].reply_list.append(text_response)
        except Exception as e:
            print('Add like fails: %s' % e)
        return 'CommentState'

    def Back_action(self, text_response, action_info, current_comment_id, current_time):
        return 'MainState'

    def get_current_info(self):
        info_dict = {
            'basic_info': self.tweet_info,
            'comment_list': [{
                'cid': cmt.cid,
                'content': cmt.comment_text,
                'reply_list': cmt.reply_list
            } for cmt in self.comment_list.sorted],
            'detect_list': self.detect_list,
            'view_num': self.view_num,
            'like_num': self.like_num,
            'repost_num': self.repost_num
        }
        return info_dict


class Simulator():
    def __init__(self, config, llm, recorder, checkpoint):
        self.global_config = config
        self.config = config['simulation_config']
        self.llm = llm
        self.recorder = recorder
        self.checkpoint = checkpoint

        self.time_system = TimeSystem(config)
        self.social_media = SocialMedia(config, llm)
        self.agents = []

        self.initialize()
    
    def recover_llm(self, llm):
        self.llm = llm

    def initialize(self):
        # Initialize users.
        user_list = load_user(self.global_config)
        self.recorder.add_record('META', {'user_list': user_list})
        for index, user in enumerate(user_list):
            agent = eval(AGENT_CLASS[user['type']])(self.global_config, index, user, self.llm)
            self.agents.append(agent)
            event_time = agent.get_next_execute_time(self.time_system.start_timestamp, self.social_media, self)
            if event_time:
                self.time_system.add_event(event_time, index)

    def agent_workflow(self, current_time, aid):
        print('----- %d START at %s -----' % (aid, timestamp_to_timestring(current_time)))
        agent = self.agents[aid]
        current_state = 'EntryState'
        time_usage = 0
        comment_page = 0
        current_comment_id = None
        action_list = []
        record_info = {
            'user_info': {
                'aid': agent.aid,
                'name': agent.name,
                'type': agent.type,
                'profile': agent.profile
            }
        }
        trajectory_list = []

        while current_state != 'EndState' and len(action_list) <= self.config['max_step']:
            trajectory_info = {'State': current_state}
            view_info = get_view_info(self.social_media, current_state, comment_page, current_comment_id)
            trajectory_info['view_info'] = view_info
            agent_action_info = {}
            action, text_response, action_info = agent.take_action(view_info, current_state, agent_action_info)
            trajectory_info['action'], trajectory_info['text_response'], trajectory_info[
                'action_info'] = action, text_response, action_info
            trajectory_info['action_detail'] = agent_action_info
            action_list.append(action)
            current_state = self.social_media.action_response(action, text_response, action_info, current_comment_id,
                                                              current_time + time_usage)

            time_usage += int(len(view_info) / 10)
            trajectory_info['time_consume'] = int(len(view_info) / 10)
            comment_page += int(action == 'More Comments')
            if action == 'Detailed Comment':
                current_comment_id = text_response

            print('Action:', action_info)
            trajectory_list.append(trajectory_info)

        record_info['trajectory'] = trajectory_list
        record_info['meta'] = {
            'start_time': timestamp_to_timestring(current_time).__str__(),
            'end_time': timestamp_to_timestring(current_time + time_usage).__str__(),
            'total_consume_time': time_usage
        }
        self.recorder.add_record('TRAJ', record_info)
        print('Finish this workflow.')
        return current_time + time_usage

    def run(self):
        cmt = 0
        while not self.time_system.is_finish():
            self.checkpoint.save_checkpoint(self)
            current_time, aid = self.time_system.execute_event()
            end_time = self.agent_workflow(current_time, aid)
            event_time = self.agents[aid].get_next_execute_time(end_time, self.social_media, self)
            if event_time:
                self.time_system.add_event(event_time, aid)
            print('### Current %d ###' % cmt)
            cmt += 1
            # self.llm.print_total_consumption()
        # self.recorder.add_record('META', {'total_consumption': self.llm.print_total_consumption()})
        self.recorder.add_record('META', {'final_social_media': self.social_media.get_current_info()})
