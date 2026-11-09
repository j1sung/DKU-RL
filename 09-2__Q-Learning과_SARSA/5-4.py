import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle

greedy_select = lambda x: np.random.choice(np.argwhere(x == np.max(x)).flatten())
def init_q(): return np.zeros(env.action_space.n)

def Q_learning(env, gamma=0.99, rho=0.01, eps=1.0, eps_decay=0.999, eps_min=0.05):
    Q = defaultdict(init_q)  # q 함수 초기화 (defaultdict 사용하여 q 함수 저장)
    for i in tqdm(range(2000000)):  # 200만 개 에피소드를 반복 (tqdm으로 진행 상황 알림)
        s, info = env.reset()
        while True:
            if np.random.random() < eps:
                a = np.random.randint(0, env.action_space.n)
            else:
                a = greedy_select(Q[s])
            s1, r, terminated, truncated, info = env.step(a)
            if terminated:
                Q[s][a] = Q[s][a] + rho * (r - Q[s][a])
            else:
                Q[s][a] = Q[s][a] + rho * ((r + gamma * np.max(Q[s1])) - Q[s][a])
            s = s1
            eps = max(eps_min, eps * eps_decay)  # ε 스케줄링
            if terminated or truncated:
                break
    return Q

env = gym.make('Blackjack-v1', render_mode='rgb_array')
Q = Q_learning(env)
pickle.dump(Q, open('f5-4.pkl', 'wb'))  # pickle로 Q 함수 저장
env.close()