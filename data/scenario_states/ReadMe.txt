<예시>
load Gen1.mat

하면 각 States 와 Nsam, sam 변수가 로드됨

[History 변수 설명]
time			: time
states X		: X
states Y		: Y
LateralAcceleration	: acc


Nsam = 장애물 갯수
sam = Nsam*3 행렬([Xcenter Ycenter Radius])