import argparse
import time

from LLM import create_LLM
from Recorder import Recorder
from Simulator import Simulator
from Config import CONFIG
from utils import CheckPoint, update_config


def run(load_checkpoint=False):
    if not load_checkpoint:
        recorder = Recorder(CONFIG)
        recorder.add_record('META', {'configuration': CONFIG})
        st = time.time()
        llm = create_LLM(CONFIG)
        checkpoint = CheckPoint(CONFIG, st)
        simulator = Simulator(CONFIG, llm, recorder, checkpoint)
        simulator.run()
        ed = time.time()
        recorder.add_record('META', {'simulation_total_time': ed - st})
        recorder.write_prompt_level()
    else:
        checkpoint = CheckPoint(CONFIG, None)
        checkpoint.load_checkpoint()
        simulator = checkpoint.simulator
        recorder = simulator.recorder
        simulator.run()
        recorder.add_record('META', {'simulation_total_time': '[CHECKPOINT]'})
        recorder.write_prompt_level()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, extra_args = parser.parse_known_args()
    CONFIG = update_config(CONFIG, extra_args)
    run(load_checkpoint=False)
