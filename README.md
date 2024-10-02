# SH

참고 코드 
1. https://github.com/junxiaosong/AlphaZero_Gomoku 이건 fiar + alphazero
2. https://github.com/geochri/AlphaZero_Chess/blob/master/src/train.py 이건 chees + alphazero
3. https://github.com/Zeta36/chess-alpha-zero/blob/master/src/chess_zero/agent/player_chess.py
3번도 chess긴 한데 좀 코드가 다른?



env.py는 현재는 안 쓰는데 코드 보려고 넣어둠.

추가로 확인해야할 거 network 에 들어가는 layer개수랑 dimenstion 맞추기
mcts쪽도 


### Windows
```bash
pip install numpy==1.25.1
pip install cmake
sudo apt-get install clang
pip install 'pettingzoo[classic]'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


"ERROR: Failed building wheel for open_spiel" 에러가 발생하는데
https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/
다운 받고 C++를 사용한 Desktop 개발 세부사항에서 MSVC v142 - VS 2019 C++ x64/x86 빌드 도구와 Windows 10 SDK를 함께 설치한다.

만약에 그래도 안되면 llvm 직접 설치해야하는 거 같음. 

아니 환경변수 등록했는데 왜 안되는거임 ???  뭔데 


### Ubuntu (Linux)
```bash
pip install numpy==1.25.1
pip install cmake
sudo apt-get install clang
pip install 'pettingzoo[classic]'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### macOS
```bash
pip install numpy==1.25.1
pip install cmake
brew install llvm
pip install 'pettingzoo[classic]'
pip install torch torchvision torchaudio

```


8x8 체스 보드의 관측 공간 세부사항:
111개의 채널이 8x8 보드 위의 다양한 정보를 제공한다. 각 채널의 의미는 다음과 같다:

채널 0-3: 캐슬링 권리

채널 0: 백이 퀸사이드 캐슬링을 할 수 있으면 모두 1로 설정

채널 1: 백이 킹사이드 캐슬링을 할 수 있으면 모두 1로 설정

채널 2: 흑이 퀸사이드 캐슬링을 할 수 있으면 모두 1로 설정

채널 3: 흑이 킹사이드 캐슬링을 할 수 있으면 모두 1로 설정

채널 4: 현재 차례가 백인지 흑인지 나타냄

채널 5: 50수 무승부 룰을 위한 수 카운트. 보드의 특정 위치가 n번째 움직임을 나타냄

채널 6: 뉴럴 네트워크가 패딩된 컨볼루션에서 보드 경계를 찾을 수 있도록 모든 값이 1로 설정됨

채널 7-18: 각 기물과 색상 조합을 위한 채널. 예를 들어, 흑색 나이트는 특정 채널에 의해 표현되며, 그 위치에 기물이 있으면 해당 인덱스가 1로 설정됨.
앙파상 가능성은 5번째 줄 대신 8번째 줄에 취약한 폰을 표시하여 나타낸다.

7번 : white 폰

8번 : white 나이트

9번 : white 비숍

10번 : white 룩

11번 : white 퀸

12번 : white 킹

13번 : black 폰

14번 : black 나이트

15번 : black 비숍

16번 : black 룩

17번 : black 퀸

18번 : black 킹 


채널 19: 동일한 위치가 이전에 나왔는지(2회 반복된 경우) 여부를 나타냄

채널 20-111: 이전 7번의 보드를 표현하며, 각각 13개의 채널로 표현된다. 가장 최근의 보드가 첫 13개의 채널을 차지하며, 그다음 두 번째 보드가 이어진다. 이 13개의 채널은 위에서 언급한 7-20번 채널과 동일한 방식으로 구성된다.

보드 스태킹 방식
AlphaZero처럼, 이 시스템은 이전 8번의 보드 상태를 쌓아서 누적한 형태로 제공한다.

하지만 AlphaZero와 달리, 이 시스템에서는 항상 백색 플레이어 쪽을 기준으로 보드가 고정되어 있다. 백색 플레이어의 킹은 항상 첫 번째 줄에 위치하며, 양 플레이어가 동일한 보드 레이아웃을 관측하게 된다.

추가적으로, env.observe('player_1') 함수는 흑색 플레이어의 관점에서 보드를 보는 기능을 제공하여, 에이전트가 흑색과 백색 모두 능숙하게 플레이할 수 있도록 돕는다.

##  Action Space
AlphaZero 체스에서는 체스의 모든 가능한 움직임을 8x8x73 크기의 배열로 표현한다. 여기서 각 차원이 의미하는 바는 다음과 같다:

8x8: 체스 보드의 모든 좌표 (즉, 64개의 위치). 이것은 체스 말이 시작되는 위치를 나타낸다.
각각의 위치는 보드 상의 칸을 나타낸다. 예를 들어, (0, 0)은 A1을, (7, 7)은 H8을 의미한다.

73: 각 말이 해당 위치에서 수행할 수 있는 가능한 움직임을 나타낸다. 73개의 평면(planes)으로 이루어져 있으며, 이 중에서 움직임의 세부 사항이 코드화되어 있다.
73개의 평면(planes) 구성:

첫 56개의 평면: '퀸'의 움직임처럼 동작하는 모든 말의 움직임을 나타낸다.
8방향(북, 북동, 동, 남동, 남, 남서, 서, 북서)으로 최대 7칸까지 이동할 수 있는 경우를 포함한다.
따라서 8방향 * 최대 7칸 = 56개의 평면이 이 움직임을 나타낸다.

다음 8개의 평면: 말(knight)의 움직임을 나타낸다.
말은 고유한 형태로 이동하기 때문에 8개의 평면에 이 이동을 기록한다.

마지막 9개의 평면: 폰(pawn)이 승진할 때 나타나는 움직임을 포함한다.
폰이 7번째 줄에서 승진할 때 나이트, 비숍, 룩으로 승진하는 움직임을 포함하며, 두 개의 대각선 이동과 승진 형태를 기록한다.

평면화(Flattening)된 행동 공간
이 다차원 배열을 평면화하면 8×8×73 = 4672개의 이산적인 행동 공간이 된다. 이렇게 하면 한 번에 단일 정수 인덱스를 사용하여 체스에서 가능한 모든 움직임을 나타낼 수 있다.

예시: 평면화된 인덱스에서 원래 좌표 복원
각 행동 a는 4672개의 가능한 행동 중 하나를 나타내며, 이를 통해 원래 (x, y, c) 좌표로 변환할 수 있다.


변환 공식:

a // (8*73) : 말이 있는 열(column) 좌표 (x 값)
(a // 73) % 8 : 말이 있는 행(row) 좌표 (y 값)
a % (8*73) % 73 : 평면(plane), 즉 어떤 움직임인지 나타냄 (c 값)

x = 6  # 열
y = 0  # 행
c = 12 # 평면(어떤 움직임)
a = x * (8 * 73) + y * 73 + c  # 평면화된 행동 공간의 인덱스

# 원래 좌표로 복원
print(a // (8*73), (a // 73) % 8, a % (8*73) % 73)

여기서 (6, 0, 12)는 (G1)에서 시작하는 말이 plane 12에 해당하는 특정한 움직임을 수행한다는 것을 의미한다. 
체스 보드에서 (6, 0)은 G1을 나타낸다.