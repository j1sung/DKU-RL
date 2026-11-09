# 학습을 마친 신경망으로 CartPole 시험
import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model
import imageio

model = load_model('./f6-2.keras')  # 신경망 불러옴

env = gym.make("CartPole-v1", render_mode='rgb_array')
s_dim = env.observation_space.shape[0]  # 상태 공간 차원
frames = []

length = 0
s, info = env.reset()
while True:
    q = model.predict(np.reshape(s, [1, s_dim]), verbose=0)  # 신경망이 예측한 행동
    a = np.argmax(q[0])  # 탐욕 선택
    s, r, terminated, truncated, info = env.step(a)
    length += 1

    frames.append(env.render())

    if terminated or truncated:
        print("에피소드의 점수:", length)
        break

env.close()

imageio.mimsave('cartpole.gif', frames, fps=30)