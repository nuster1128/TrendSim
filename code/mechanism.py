import numpy as np


# ----- Time Function -----
def time_func(x, alpha, A, T):
    if x <= T:
        return np.exp(A * (x - T))
    elif x <= T + 1 / alpha:
        return -alpha * A * (x - T - 1 / 2 / alpha) ** 2 + 1 + A / 4 / alpha
    else:
        return (x - T - 1 / alpha + 1) ** (-A)


STATE_CODE_TO_ACTION = {
    'EntryState': {
        '0': 'Enter',
        '1': 'Leave'
    },
    'MainState': {
        '0': 'Like_tweet',
        '1': 'Comment',
        '2': 'Repost',
        '3': 'More Comments',
        '4': 'Detailed Comment',
        '5': 'Leave'
    },
    'CommentState': {
        '0': 'Like_comment',
        '1': 'Reply',
        '2': 'Back'
    }
}


# ----- View Info -----

def get_view_info(social_media, current_state, comment_page, current_comment_id):
    if current_state == 'EntryState':
        return show_EntryState(social_media)
    elif current_state == 'MainState':
        return show_MainState(social_media, comment_page)
    elif current_state == 'CommentState':
        return show_CommentState(social_media, current_comment_id)


def show_EntryState(social_media):
    info = '标题: %s;简介: %s.' % (social_media.tweet_info['title'], social_media.tweet_info['brief_intro'])
    return info


def show_MainState(social_media, comment_page):
    comment_start = comment_page * social_media.config['comment_pagesize']
    info = '内容: %s;' % social_media.tweet_info['content']
    info += '评论: '
    if comment_start >= len(social_media.comment_list):
        info += '-;'
    else:
        for index, comment in enumerate(social_media.comment_list.sorted[
                                        comment_start:min(comment_start + social_media.config['comment_pagesize'],
                                                          len(social_media.comment_list))]):
            info += ';[%d] %s' % (index + comment_start, comment)
        info += ';'
    return info


def show_CommentState(social_media, current_comment_id):
    if current_comment_id >= len(social_media.comment_list):
        return '该评论不存在.'
        # raise LLMException('Comment Index Out-of-range','Non-prompt',str(current_comment_id))
    comment = social_media.comment_list.sorted[current_comment_id]
    info = '评论: %s;' % comment
    info += '回复: '
    if len(comment.reply_list) == 0:
        info += '-;'
    else:
        for index, reply in enumerate(
                comment.reply_list[:min(social_media.config['reply_pagesize'], len(comment.reply_list))]):
            info += ';[%d] %s' % (index, reply)
        info += ';'
    return info


# ----- Action Info -----

def get_action_info(action, extra_response):
    if action == 'Enter':
        return '查看了微博详情。\n'
    elif action == 'Leave':
        return '离开了该微博。\n'
    elif action == 'Like_tweet':
        return '点赞了该微博。\n'
    elif action == 'Comment':
        return '评论了该微博: %s \n' % extra_response
    elif action == 'Repost':
        return '转发了该微博。\n'
    elif action == 'More Comments':
        return '查看了更多评论。\n'
    elif action == 'Detailed Comment':
        return '查看了评论[%d]详情。\n' % extra_response
    elif action == 'Like_comment':
        return '点赞了该评论。\n'
    elif action == 'Reply':
        return '回复了该评论: %s\n' % extra_response
    elif action == 'Back':
        return '回到了该微博。\n'


# ----- Prompt -----

def get_sp_prompt(agent, view_info):
    prompt ="""
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，刚刚浏览的内容是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, view_info)
    prompt += """
请你针对这一浏览内容，以第一人称，输出约40字的浏览印象。
输出样例：
今年上半年，A股市场虽然人均盈利，但整体赚钱效应并不明显，只有少数人真正获利。我国居民投资股市的比例相对较低，更多倾向于房产投资。未来可能会有更多资金从楼市转向股市，为市场提供新的活力。
"""
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览信息]%s\n' % view_info
    # prompt += '请从此人的角度，输出约40字的对[浏览信息]的印象。'
    return prompt


def get_dp_prompt(agent, full_obs, impression, current_state):
    if current_state == 'EntryState':
        return prompt_EntryState(agent, impression)

    if current_state == 'MainState':
        return prompt_MainState(agent, impression)

    if current_state == 'CommentState':
        return prompt_CommentState(agent, impression)


def prompt_EntryState(agent, impression):
    prompt ="""
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点，选择要执行的动作:
[0] 查看详情
[1] 离开
请用序号表示所选的动作，输出只包括一个数字。
输出样例:
0
"""
    return prompt
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '对社会热点的浏览印象: %s\n' % impression
    # prompt += '你经常查看社会热点，请从此人的角度选择动作: '
    # prompt += '[0] 查看该社会热点的详细信息;'
    # prompt += '[1] 离开.\n'
    # prompt += '用序号代表所选动作，输出必须只包含1或者0.'
    # return prompt


def prompt_MainState(agent, impression):
    prompt ="""
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点，选择要执行的动作:
[0] 点赞
[1] 评论
[2] 转发
[3] 查看更多评论
[4] 查看评论细节
[5] 离开
请用序号表示所选的动作，输出只包括一个数字。
输出样例:
1
"""
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '此人正在浏览微博热搜，请从此人的角度选择以下动作: '
    # prompt += '[0] 点赞;'
    # prompt += '[1] 评论;'
    # prompt += '[2] 转发;'
    # prompt += '[3] 查看更多评论;'
    # prompt += '[4] 查看评论细节;'
    # prompt += '[5] 离开.\n'
    # prompt += '用序号代表所选动作，输出必须只包含1个0~5的整数。'
    return prompt


def prompt_CommentState(agent, impression):
    prompt ="""
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点的一些评论，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点的评论，选择要执行的动作:
[0] 点赞
[1] 回复评论
[2] 退出评论
请用序号表示所选的动作，输出只包括一个数字。
输出样例:
1
"""
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '此人正在浏览微博评论，请从此人的角度选择以下动作: '
    # prompt += '[0] 点赞;'
    # prompt += '[1] 回复;'
    # prompt += '[2] 返回.\n'
    # prompt += '用序号代表所选动作，输出必须只包含1个0~2的整数。'
    return prompt


def get_rp_prompt_summary(agent, impression, action_info):
    prompt ="""
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点，你执行了动作：%s
请你针对印象与动作，以第一人称，输出一个约40字的总结。
输出样例: 
这条关于财经的新闻很有价值，它揭示了A股市场的盈利状况和居民投资偏好。我点赞这条新闻，期待未来资金从楼市转向股市，为市场注入新的活力。
""" % action_info
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '[当前动作]%s\n' % action_info
    # prompt += '请从此人角度，输出约40字的总结.'
    return prompt


def get_rp_prompt_opinion(agent, impression, action_info):
    prompt ="""
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点，你执行了动作：%s
请你针对印象与动作，以第一人称，输出一个约20字的个人观点。
输出样例: 
我认为未来资金会从楼市转向股市，为市场注入新的活力。
""" % action_info
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '[当前动作]%s\n' % action_info
    # prompt += '请从此人角度，输出约20字的对事件的看法.'
    return prompt


def get_rp_prompt_emotion(agent, impression, action_info):
    prompt ="""
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点，你执行了动作：%s
请你基于之前的心理状况，结合当前印象与动作，输出一个百分数，客观地表示当前心理状况的积极程度，应该体现出其中的前后变化，因为角色的心理状况会受到浏览信息的影响，积极则会升高，消极则会降低。
输出应该只包括这个百分数，禁止给出任何解释和描述。
输出样例: 
35%%
""" % action_info
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '[当前动作]%s\n' % action_info
    # prompt += '请从此人角度，输出一个百分数，表示情绪积极程度.'
    return prompt


def get_rp_prompt_socialconf(agent, impression, action_info):
    prompt = """
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点，你执行了动作：%s
请你基于之前的心理状况，结合当前印象与动作，输出一个百分数，客观地表示当前心理状况的社会信心，应该体现出其中的前后变化，因为角色的心理状况会受到浏览信息的影响，积极则会升高，消极则会降低。

输出应该只包括这个百分数，禁止给出任何解释和描述。
输出样例: 
35%%
""" % action_info
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '[当前动作]%s\n' % action_info
    # prompt += '请从此人角度，输出一个百分数，表示社会信心程度.\n'
    return prompt


def get_comment_prompt(agent, impression):
    prompt = """
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个社会热点，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一社会热点，以第一视角，输出一条约30字的评论。
输出样例: 
只有未来资金会从楼市转向股市，才能为市场注入新的活力。希望企业能够齐心协力，共渡难关。
"""
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '请从此人的角度发表一条约30字的评论。'
    return prompt


def get_reply_prompt(agent, impression):
    prompt = """
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一个评论，对它的印象是: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这一评论，以第一视角，输出一条约30字的回复。
输出样例: 
我不同意你的观点，我认为只有未来资金会从楼市转向股市，才能为市场注入新的活力。
"""
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '请从此人的角度发表一条约30字的回复。'
    return prompt


def get_comment_id(agent, impression):
    prompt = """
请你扮演以下角色:
[人物特点] %s
[人物记忆] %s
[人物观点] %s
[心理状况] 积极程度为%.0f/1.0，社会信心为%.0f/1.0
你喜欢浏览社会热点，你刚刚浏览了一些评论: %s 
""" % (agent.profile['description'], agent.memory, agent.opinion,agent.emotion * 100,agent.social_confidence * 100, impression)
    prompt += """
请你针对这些评论，选择想要回复的一条评论，只需要输出该评论的编号，禁止给出任何解释和描述。
输出样例: 
0
"""
    # prompt = '请你扮演此人: %s %s %s' % (agent.profile['description'], agent.memory, agent.opinion)
    # prompt += '当前情绪积极程度为%.0f' % (agent.emotion * 100) + '%,'
    # prompt += '当前社会信心为%.0f' % (agent.social_confidence * 100) + '%.\n'
    # prompt += '[浏览印象]%s\n' % impression
    # prompt += '此人现在要回复一条评论，请从此人的角度输出要回复评论的编号，只输出一个整数。'
    return prompt
