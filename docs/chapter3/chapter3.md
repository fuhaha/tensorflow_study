#3장 군집화 

##텐서플로와 파이선 자료형 

|텐서플로 자료형|파이선자료형|설명|
|---|---|---|
|DT_FLOAT|tf.float32|32비트 실수|
|DT_INT16|tf.int16|16비트 정수|
|DT_INT32|tf.int32|32비트 정수|
|DT_INT64|tf.int64|64비트 정수|
|DT_STRING|tf.string|문자열|
|DT_BOOLEAN|tf.bool|불리언|

##텐서 차원 표현 
구조(shape)
랭크(rank)
차원번호(dimension number)

구조, 랭크, 차원 번호 관계

|구조|랭크|차원번호|
|---|---|---|
|[]|0|0-D|
|[D0]|1|1-D|
|[D0,D1]|2|2-D|
|[D0,D1,D2]|3|3-D|
||||
|[D0,D1,D2,,,,Dn]|n|n-D|

##텐서 변환 함수 

|함수|설명|
|---|---|
|tf.shape|텐서의 구조를 알아냅니다|
|tf.size|텐서의 크기를 알아냅니다|
|tf.rank|텐서의 랭크를 알아냅니다|
|tf.reshape|텐서의 원소는 그대로 유지하면서 텐서의 구조를 바꿉니다.|
|tf.squeeze|텐서에서 크기가 1인 차원을 삭제 합니다.|
|tf.expand_dims|텐서에 차원을 추가합니다|
|tf.slice|텐서의 일부분을 삭제합니다.|
|tf.split|텐서를 한 차원을 기준으로 여러개의 텐서로 나눕니다.|
|tf.tile|한 텐서를 여러번 중복해서 새 텐서를 만듭니다|
|tf.concat|한 차원을 기준으로 텐서를 이어 붙입니다.|
|tf.reverse|텐서의 지정된 차원을 역전시킵니다.|
|tf.transpose|텐서를 전치합니다.|
|tf.gather|주어진 인덱스에 따라 텐서의 원소를 모읍니다|

##차원 확장 하기 
ipython, expand_dims

#텐서플로 데이터 저장소 확보
1. 데이터 파일로 얻기
2. 상수나 변수로 미리 로드
3. 파이선 코드로 작성해 제공

###데이터파일로 얻기 
4장 mnist 데이터 로드 파일 참고 

###변수와 상수 
<code>
tf.constant()  #상수 선언
tf.variable() #변수 선언
</code>

상수를 생성하는 방법

|함수|설명|
|---|---|
|tf.zeros_like|모든 원소를 0으로 초기화한 텐서를 생성|
|tf.ones_like|모든 원소를 1로 초기화환 텐서를 생성|
|tf.fill|주어진 스칼라 값으로 원소를 초기화환 텐서를 생성|
|tf.constant|함수로 지정된 값을 이용하여 상수 텐서를 생성|

ipython tf-const-gen

텐서 생성 관련 함수 

|함수|설명|
|---|---|
|tf.random_normal|정규분포를 따르는 난수로 텐서 생성|
|tf.truncated_normal|정규분포를 따르는 난수로 텐서를 생성, 크기가 표준편차의 2배보다 큰 값은 제거 |
|tf.random_uniform|균등분포를 따르는 난수로 텐서를 생성|
|tf.random_shuffle|첫 번째 차원을 기준으로 텐서의 원소를 섞습니다|
|tf.set_random_seed|난수 시드를 설정|

이 함수들은 생성할 텐서 차원 구조를 매개변수로 입력 
리턴 되는 텐서 변수는 입력된 매개 변수와 동일한 구조 

변수를 사용하기 전에는 데이터 그래프를 구성한 후 run()함수를 실행하기 전에 
반드시 초기화해야 함 
초기화 방법 : tf.initialize_all_variables() 


###파이선 코드로 텐서 변수 생성

심볼릭 변수를 플레이스홀더를 통해 사용 

placeholder() 함수를 호출
원소의 자료형, 텐서의 구조를 매개변수로 보내 설정.

파이선 코드에서 Session.run(), Tensor.eval() 메소드를 호출할때 
feed_dict 매개변수를 통해 플레이스 홀더를 지정하여 전달 

ipython chapter1 
tf.mul -> tf.multiply(version1 수정 사항)

#K 평균 알고리즘 

K 평균 알고리즘은 군집화 문제를 풀기 위한 자율 학습 알고리즘의 일종 
자율 학습 혹은 비지도 학습
이 알고리즘은 간단한 방법으로 주어진 데이터를 지정된 군집개수(K)로 그룹화 
한 군집내의 데이터를 동일한 성질을 가짐.
다른 그룹과는 구분됨
-> 한 군집내의 모든 원소들은 군집 밖의 데이터보다 서로 더 닮음


###알고리즘 결과
중심(centroid)으로 불리는 K개의 점
이 점들은 각기 다른 그룹의 중심점을 나타내며 데이터들은 K개의 군집 중 하나에만 속할 수 있음. 
한 군집내의 모든 데이터들은 다른 어떤 중심들 보다 자기 군집 중심과의 거리가 더 가까움. 

군집을 구성할 때 직접 오차함수를 최소화하려면 계산 비용이 비쌈. 
가장 널리 사용되는 방법은 반복을 통해 수렴하는 반복 개선 방법 

####알고리즘 수행 단계 
#####1. 초기 단계 : K개 중심의 초기 집합을 결정
#####2. 할당 단계 : 각 데이터를 가장 가까운 군집에 할당 
#####3. 업데이트 단계 : 각 그룹에 대해 새로운 중심을 계산 

youtube k-means 
https://www.youtube.com/watch?v=_aWzGGNrcic

K-means 예제 코드 : ipython k-means 

K-means 코드 설명 : ipython k-means-comment







