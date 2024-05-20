CONFIG = {
    # Configuration for LLM
    'LLM_config': {
        'model_name': 'glm-3-turbo',
        'API_KEY': 'XXX'
    },

    # Configuration for Agents (Initialize and Hyper-parameters).
    'agent_config': {
        'default_score': {
            'memory': '',
            'emotion': 0.5,
            'social_confidence': 0.5,
            'opinion': ''
        },
        'execute_gap_params': {
            'alpha': 1.0,
            'beta': 14400,
            'T': 5.0
        }
    },

    # Configuration for file paths.
    # - Path for loading users and posts(i.e., tweets).
    # - Path for record and checkpoint.
    'data_path': {
        'normal_user': 'data/user_1000.csv',
        'tweets': 'data/tweets.csv',
        'record': 'output/record.json',
        'load_checkpoint': 'checkpoint/check.pickle',
        'save_checkpoint': 'checkpoint/check.pickle'
    },

    # Configuration of Simulation.
    'simulation_config': {
        'max_step': 3,
        'degree': 'medium',
        'baseline': 'full',
    },

    # Configuration of Social Media.
    'media_config': {
        'tweet_index': 1,
        'comment_pagesize': 4,
        'reply_pagesize': 4
    },

    # Configuration for Intervention
    'intervention': {
        # Whether detect the poisoning comments.
        'comment_poisoning_detect': {
            'type': 'LLM',
            'split_time': 0.0
        }

    },

    'time_config': {
        'start_time': '2022-09-09 06:00:00',
        'end_time': '2022-09-09 22:00:00',
        'epoch_num': 2
    }

}
