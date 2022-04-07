# RL study

### 이론

- 그 주의 담당자는 그 주의 분량에 MarkDown 형태로 자료를 정리하고, 강의형태로 리뷰를 진행.
- 담당순서는 순환형태로 돌아가며 진행
- 스터디 진행 분량 (이틀당 대략 50분 강의 수강 +-. 최대 10분 정도)
- 책 - 기초부터 시작하는 강화학습/신경망 알고리즘 (위키북스)
- 강의 - https://fastcampus.co.kr/courses/202611/clips/



<details><summary>이론 일정</summary>


**Part 1. 강화학습 소개**

- [1차시](https://github.com/black-subb/study_Reinforcement_learning/issues/1#issue-1121989354)
  - 01\. 강화학습(RL)이란? 
  - 03\. RL 구현을 위한 환경설정 

**Part 2. 가치기반 강화학습의 풀이법**

- [2차시](https://github.com/black-subb/study_Reinforcement_learning/issues/2)
  - Ch 01. 마르코프 결정과정
    - 01\. MP, MRP
- [3차시](https://github.com/black-subb/study_Reinforcement_learning/issues/3#issue-1135288989)
  - Ch 02. 동적 계획법
    - 01\. DP(동적계획법)
- [4차시](https://github.com/black-subb/study_Reinforcement_learning/issues/4#issue-1137866355)
  - Ch 02. 동적 계획법
    - 03\. 비동기적 동적계획법
- [5차시](https://github.com/black-subb/study_Reinforcement_learning/issues/5#issue-1144913368)
  - Ch 03. 모델 없이 세상 알아가기
    - 01\. 도박의 도시 몬테카를로(MC) 그리고 MC 정책추정 - 1
- [6차시](https://github.com/black-subb/study_Reinforcement_learning/issues/6#issue-1148625300)
  - Ch 03. 모델 없이 세상 알아가기
    - 02\. 도박의 도시 몬테카를로(MC) 그리고 MC 정책추정 - 2
- [7차시](https://github.com/black-subb/study_Reinforcement_learning/issues/7#issue-1148625579)
  - Ch 03. 모델 없이 세상 알아가기
    - 04\. Temporal Difference (TD) 정책추정
- [8차시](https://github.com/black-subb/study_Reinforcement_learning/issues/7#issue-1148625579)
  - Ch 04. 모델없이 세상 조종하기
    - 01\. [MC Control] MC기법을 활용한 최적 정책 찾기 
    - 03\. [SARSA] TD기법을 활용한 최적 정책 찾기 
- [9차시]()
  - Ch 05. 어깨넘어 배워서 세상 조종하기
    - 01\. Off-policy MC control
    - 02\. Off-policy TD control, Q-learning

**Part 3. 함수 근사기법**

- [10차시]()
  - Ch 01. 함수 근사 소개
    - 01\. 함수 근사 어떻게 RL에
    - 02\. 함수 근사 첫걸음 - 선형회귀 모델
- [11차시]()
  - Ch 02. 심층 신경망을 활용한 함수근사
    - 01\. 선형 근사 - 1
    - 02\. 선형 근사 - 2
- [12차시]()
  - Ch 02. 심층 신경망을 활용한 함수근사
    - 05\. Naive Deep Q-Learning - 1
    - 06\. Naive Deep Q-Learning - 2
- [13차시]()
  - Ch 02. 심층 신경망을 활용한 함수근사
    - 07\. 합성곱 신경망 기초

</details>

### 실습

- [openai gym skiing](https://gym.openai.com/envs/Skiing-v0/)에 여러 알고리즘 적용
  - montecarlo, REINFOCE, Actor-critic, PPO

<details><summary>환경 설정</summary>

- gym[atari] 설치 후 autorom 추가 설치

```sh
pip install -U gym[atari]
pip install autorom
autorom
```

- 환경 설정 테스트

```python
import gym
// print(gym.envs.registry.all())
env = gym.make('Skiing-v0', render_mode='human')
env.reset()
for _ in range(100):
    env.step(env.action_space.sample()) # take a random action
env.close()
```

- 참고 링크

  - [openai gym getting started](https://gym.openai.com/docs/)

  - [ModuleNotFoundError: No module named 'gym.envs.atari'](https://github.com/openai/gym/issues/2498)

  - [Error in importing environment OpenAI Gym](https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym)

</details>
