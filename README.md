# TrendSim

## 📝Introduction

Poisoning attacks in social media trends are critical problems in the web community. Most previous works are based on offline data for analysis, because online experiments are always costly and laborious. Recently, large language models (LLMs) have shown their ability to simulate human behaviors, which are capable of supporting online experiments in an economical and efficient way. In this paper, we propose **TrendSim**, an autonomous agent simulation method based on LLMs, to understand poisoning attacks in social media trends. We design a simulation framework that aligns with real platforms, and develop LLM-based autonomous agents to simulate the psychology and behaviors of users. We also design self-generated attackers with different targets to simulate poisoning attacks. To verify the effectiveness of TrendSim, we further evaluate the alignment between simulation and real circumstances. Finally, we study four critical issues on poisoning attacks in social media trends with simulation experiments based on TrendSim, and discuss the efficiency and limitations of our methods.

## 📌Features of TrendSim

### 📱Simulator of Social Media Trends

![figure1](./assets/framework.png)

### 👾LLM-based Autonomous User Agent

![figure3](./assets/user_agent.png)

### 😈Self-generated Attacker Agent

![figure3](./assets/attacker.png)

## 🧰Our Contributions

- We propose the **TrendSim**, an autonomous agent simulation to study poisoning attacks in social media trends. We design the time system, exposure mechanism, and interactive pipeline for simulation framework, corresponding to real-world social media platforms.

- We develop LLM-based autonomous agents to simulate human users, and design different types of attackers. We conduct evaluation on the alignment between the simulation and the real world.

- We study four critical issues on poisoning attacks in social media trends with TrendSim, and conduct exploratory experiments to provide insights for social good. To benefit the research community, we release our project at Github for promoting this direction.

## 🚀Run TrendingSim

### 💻1 Prepare for the simulation.

Create an environment.

```shell
conda create -n trendsim python=3.9
```

Activate the environment.

```shell
conda activate trendsim
```

Install the packages.

```shell
cd TrendSim
pip install -r requirements.txt
```

### 🤖2 Prepare your Large Language Model.

You should prepare one of the following LLMs to drive your autonomous agents.

**GLM(API)**

For preparing `GLM-3-turbo/GLM-4`, you can refer to https://www.zhipuai.cn.

**GPT(API)**

We also provide an interface to utilize `GPT3.5-turbo/GPT-4` by API, and you can refer to `LLM.GPT_LLM`.

**Other Local LLMs**

We also provide an interface to utilize other local LLMs (e.g., Baichuan2-7B-Chat), and you can prepare your LLMs according to `LLM.LocalLLM` and `run_api.py`.

### ▶️3 Run your Large Language Model. 

For `GLM-3-turbo`, you can configure your API key in `Config.py`.

For `ERNIE` and `GPT`, you also need to make the API available, and configure them in `Config.py`.

For other local LLMs, you just need to make them compatible with `LLM.LocalLLM`, and start to run.

### 🎯4 Run the simulation.

First of all, configure the file `Config.py` with the simulation information.

(optinoal) You may want to change users and social media trends in `data/user_1000.csv` and `data/tweets.csv`.

Then, run the simulation.

```shell
python run.py
```

You can also change configurations in commend line.

```shell
python run.py -tweet_index 6 -record_path output/record_tweet_14_mild.json -degree mild -baseline full
```

Finally, you can check the records in your output path.
