# Multi-agent Pac-Man
<p align="center">
      <img src="https://user-images.githubusercontent.com/80669616/137313558-d2edb776-6b35-4c53-a9ff-14f405cb2db4.png" width="700"><br>Pac-Man GUI simulator
</p>

By using three kinds of algorithm `Minimax`, `Alpha-beta pruning`, `Expectimax`, `customized evaluation function`  
You can simulate Multi-agent Pac-Man.(python)

## 본 프로젝트의 주요 쟁점

Adversarial Game의 일종인 Multi-agent Pac-Man 게임에서 다음의 세가지 알고리즘을 적용하여  
CSP(Constraint Satisfaction Problem)을 해결하는 것을 목표로 합니다.

- Minimax
- Alpha-beta pruning
- Expectimax

세부 구현은 **submission.py** 파일을 참고해주세요.

## 시뮬레이션 방법

터미널 창에 다음의 예시를 참고하여 명령을 입력하면 됩니다.
> python pacman -p Expectimax  
(Expectimax 알고리즘으로 팩맨이 시뮬레이션 됩니다.)  
> pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -q  
(Expectimax+customized evaluation function)    

## Expectimax + customized-Evaluation function의 20번 시뮬레이션 결과
<p align="center">
      <img src="https://user-images.githubusercontent.com/80669616/137313662-444605bf-f35f-4221-a473-82403f319d52.png" width="700"><br>Pac-Man GUI simulator
</p>


> 본 프로젝트는 POSTECH CSED342 인공지능 과목에서 진행한 프로젝트의 일부분입니다.
