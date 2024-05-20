from Agent import Agent
from mechanism import *
from utils import *
from random import randint


class Attacker(Agent):
    def __init__(self, config, aid, feature, llm):
        super(Attacker, self).__init__(config, aid, feature, llm)

        self.last_action = None

    def get_next_execute_time(self, current_time, social_media, simulation):
        ed = timestring_to_timestamp(simulation.global_config['time_config']['end_time'])
        st = timestring_to_timestamp(simulation.global_config['time_config']['start_time'])
        if st == current_time:
            return current_time + random.randint(1, int((ed-st)/10))
        avg_epoch = 2 * simulation.global_config['time_config']['epoch_num']
        gap = (ed - st) / avg_epoch
        if current_time + gap <= ed:
            return current_time + gap
        else:
            return None

    def get_poisoning_comment(self, full_obs):
        raise "Must be implemented by the specific attacker."

    def take_action(self, full_obs, current_state, agent_action_info):
        if current_state == 'EntryState':
            action, text_response = 'Enter', ''
        elif current_state == 'MainState':
            if self.last_action == 'EntryState':
                action, text_response = 'Comment', self.get_poisoning_comment(full_obs)
            else:
                action, text_response = 'Leave', ''
        else:
            raise "Attack Error!"

        self.last_action = current_state
        action_info = get_action_info(action, text_response)
        return action, text_response, action_info


class SocialAttacker(Attacker):
    def __init__(self, config, aid, feature, llm):
        super(SocialAttacker, self).__init__(config, aid, feature, llm)

    def get_poisoning_comment(self, full_obs):
        exp = ["这个社会太黑暗了！贫富差距太大！打工人不配活着！",
            "教育系统根本就是在骗人，学历毫无意义！",
            "人与人之间的关系都是虚伪的，没有真正的友谊！",
            "社会道德沦丧，道德败坏的人比正直的人更受欢迎！",
            "社会竞争太激烈，没有人关心别人的困境，都是自私自利！"]
        i = randint(0,4)
        prompt = '你将扮演一个对世界与社会不满的人。'
        prompt += '你接收到以下内容和评论: %s ' % full_obs
        prompt += '请紧密联系上述内容和评论，进行仿照下面这句话的评论，大约40字：%s' % exp[i]
        cmt = self.llm.fast_run(prompt)
        return cmt


class TrollingAttacker(Attacker):
    def __init__(self, config, aid, feature, llm):
        super(TrollingAttacker, self).__init__(config, aid, feature, llm)

    def get_poisoning_comment(self, full_obs):
        exp = ["只有坤坤是最棒的！谁都不如坤坤！其他人都弱爆了！",
              "这算什么，不如我们家爱豆一丝一毫啊！",
              "有点脑子都不会说出你这样的话，真无语！",
              "你这样的评论我见多了，纯属无稽之谈。",
              "我觉得也没什么了不起的，完全不行啊。"]
        i = randint(0,4)
        prompt = '你将扮演一个坏情绪的人。'
        prompt += '你接收到以下内容和评论: %s ' % full_obs
        prompt += '请紧密联系上述内容和评论，进行仿照下面这句话的评论，大约40字：%s' % exp[i]
        cmt = self.llm.fast_run(prompt)
        return cmt


class FactAttacker(Attacker):
    def __init__(self, config, aid, feature, llm):
        super(FactAttacker, self).__init__(config, aid, feature, llm)

    def get_poisoning_comment(self, full_obs):
        exp = ["事实不是这样的！真相总是会被掩盖！",
              "这个说法是错的，完全不可信，真相哪那么容易公开！",
              "这个评论肯定被操纵了！",
              "我才不信你这个说法，我才知道真相！",
              "这个评论一定是假的，你们都是假粉！"]
        i = randint(0,4)
        prompt = '你将扮演一个歪曲事实的人。'
        prompt += '你接收到以下内容和评论: %s ' % full_obs
        prompt += '请紧密联系上述内容和评论，进行仿照下面这句话的评论，大约40字：%s' % exp[i]
        cmt = self.llm.fast_run(prompt)
        return cmt