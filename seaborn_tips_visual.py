import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tips = sns.load_dataset('tips')
titanic = sns.load_dataset('titanic')
flights = sns.load_dataset('flights')

# print(type(tips))
print(titanic)

# Regplot : 데이터를 점으로 나타내면서 선형성을 함께 확인하다.
# 지불 금액에 따른 팁의 양 : ax 객체로 선언
# ax = sns.regplot(x='total_bill', y='tip', data=tips)
# fit_reg=False 선형선 출려하지 않는 옵션
# ax = sns.regplot(x='total_bill', y='tip', data=tips, fit_reg=False)
# ax.set_xlabel('TB')     # x축 이름 설정
# ax.set_ylabel('Tip')    # y축 이름 설정
# ax.set_title('Total bill ans Tip')  # 그래프 제목 설정
# plt.show()


# Joinplot: 안쪽은 점 데이터로 분포를 확인하고 바깥쪽은 막대그래프로 밀집도를 확인한다.
#           데이터의 경향을 파악하기 좋다
# joint = sns.jointplot(x='total_bill', y='tip', data=tips)
# kind= 'hex' 옵션을 주면 6각 분포로 출력
# joint.set_axis_labels(xlabel='TB', ylabel='Tip')
# 굳이 라벨을 설정하지 않아도 기본 column명으로 라벨 형성
# plt.show()


# kde: 이차원 밀집도 그래프 - 등고선 형태로 밀집 정도를 확인할 수 있다.
# kde, ax = plt.subplots()
# ax = sns.kdeplot(x=tips['total_bill'],
#                  y=tips['tip'],
#                  shade=True)
# ax.set_title('Kernel Density Plot')
# plt.show()


# Barplot: 막대그래프
# ax = plt.plot()
# ax = sns.barplot(x='time', y='total_bill', data=tips)
# plt.show()


# Boxplot: 박스 그래프
# ax = sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker', palette='Set3')
# plt.show()


# Pairplot(data=tips)
# - 수치에 해당하는 그래프를 전반적으로 그려줌
# - 관계 그래프를 확인할 수 있음
# - 전반적인 상태를 확인할 수 있어 처음 데이터를 확인할 때 전체를 파악하기 좋음
# sns.pairplot(data=tips)
# plt.show()
# 한 번에 그리지 않고 원하는 그래프와 데이터로 pairplot을 그릴 수 있다.
# pg = sns.PairGrid(tips)
# pg.map_upper(sns.regplot) # 위쪽 그래프
# pg.map_lower(sns.kdeplot) # 아래쪽 그래프
# pg.map_diag(sns.distplot)  # 가운데 그래프
# plt.show()


# countplot: 해당 카테고리 별 데이터의 개수를 보여주는 그래프
# 요일별 팁 받은 횟수 출력
# sns.countplot(x='day', data=tips)
# plt.show()


# heatmap : 카테괴별 데이터 분류
titanic_size = titanic.pivot_table(index='class', columns='sex', aggfunc='size')
# aggfunc='size' 각 데이터의 건수에 대한 출력
# 그냥 pivot table은 평균, 분산 등의 결과가 출력됨
# sns.heatmap(titanic_size, annot=True, fmt='d', cmap=sns.light_palette('red'))
# annot=True    숫자가 출력될 수 있게
# fmt='d'       지수 형태가 아닌 숫자로
# camp=sns.light_palette('red')  색상 결정
# plt.show()
fp = flights.pivot('month', 'year', 'passengers') # 열인덱스, 행인덱스, 데이터 순서로 들어감
# sns.heatmap(fp, linewidths=1, annot=True, fmt='d')
# plt.show()


# Pandas에서 바로 plot그리기
df = pd.DataFrame(np.random.randn(100,3),
                  index=pd.date_range('1/28/2020', periods=100),
                  columns=['A', 'B', 'C'])
# 랜덤으로 100행 3열의 DataFrame을 생성
# index를 pd.data_range를 통해서 2020.01.28일을 기준으로 100일

# 판다스 데이터프레임에서 각 여에 대해서 그래프를 그릴 수 있음
df.plot() # 일변 변화량 또는 변동폭을 그래프로 나타냄
df.cumsum().plot()
# cumsum() 누적합 : 누적합으로 그래프를 그리면 해당 열의 값이 어떻게 변해가는지 확일할 수 있음
# 금융, 주식 등에서 수익률을 계산할 때 활용할 수 있음
# plt.show()


# pie 그래프
df = titanic.pclass.value_counts()
df.plot.pie(autopct='%.2f%%')
plt.show()