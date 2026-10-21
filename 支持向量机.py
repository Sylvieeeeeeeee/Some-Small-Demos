import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd

#*
#
# 这个入门 demo 想完成的任务是，训练一个支持向量机 来通过学生的体测数据，分类出学生的性别，其中 1是男生 ，0 是女生
#
# *#
#*
# x_data 代表体侧数据，左边是50米跑成绩，右边是肺活量·
#
# *#
x_data = [
[9.2, 2073],
[7.5, 2949],
[8.1, 2173],
[7.5, 2949],
[8.1, 2173],
[9.9, 2163],
[9.5, 4058],
[9.3, 2565],
[9.5, 4058],
[9.3, 2565],
[9.9, 2163],
[9.3, 2565],
[9.2, 2658],
[9.1, 6076],
[9.6, 2772],
[9.5, 3260],
[7.9, 6064],
[9.1, 3076],
[9.6, 2772],
[9.5, 3260],
[7.9, 3064],
[9.3, 2658],
[8.4, 2357],
[9.5, 2165],
[8.5, 2165],
[8.5, 2660],
[7.5, 2867],
[7.8, 3159],
[9.7, 1687],
[9.3, 1868],
[9.9, 1958],
[9.7, 1756],
[9.0, 1575],
[9.1, 1700],
[9.7, 1690],
[8.9, 1888],
[12.1, 1949],
[11.1, 1548],
[10.1, 1624],
[9.8, 1657],
[9.6, 1574],
[9.2, 1544],
[9.0, 1687],
[11.2, 1985],
[10.3, 1699],
[12.7, 1940],
[9.6, 1850],
[9.4, 1566],
[9.0, 1745],
[9.9, 1553],
[10.1, 1650],
[9.2, 1620],
[9.0, 1750],
[9.7, 1548],
[9.8, 1655],
[9.9, 1542],
[11.1, 1534],
[9.0, 1666],
[9.7, 1787],
[9.2, 1940],
[9.3, 1650],
[11.7, 1671],
[12.1, 1842],
[10.3, 1549],
[9.1, 1749],
[9.7, 1450],
[9.1, 2010],
[10.0, 2080],
[10.1, 2188],
[9.7, 1752],
[8.9, 1863],
[8.7, 1973],
[10.0, 1788]
]

# 性别表示
data_list = [1] * 40 + [0]*33

#转化为numpy数组
gender = np.array(data_list)
grade = np.array(x_data)

# ?生成实例数据
print(grade)

# ?将数据分为训练集和测试集
grade_train,grade_test,gender_train,gender_test = train_test_split(grade,gender,test_size = 0.3,random_state = 42)

# 创建支持向量机模型
svm = SVC(kernel='linear')

#训练模型
svm.fit(grade_train,gender_train)

# ?获取支持向量和决策函数
support_vectors = svm.support_vectors_
coef = svm.coef_
intercept = svm.intercept_

# 输出线性方程表达式??????
print("Linear SVM Equation:")
print(f"  w0 * x + w1 * y + b = 0")
print(f"  where w0 = {coef[0,0]:.2f}, w1 = {coef[0,1]:.2f}, b = {intercept[0]:.2f}")

# 预测
gender_pred= svm.predict(grade_test)

# 评估模型
accuracy = accuracy_score(gender_test, gender_pred)
report = classification_report(gender_test, gender_pred)
conf_matrix = confusion_matrix(gender_test, gender_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("Classification Report:\n",report)
print("Confusion Matrix:\n",conf_matrix)

# 可视化
plt.figure(figsize = (10,6))
plt.scatter(grade[:, 0], grade[:, 1], c=gender, cmap='coolwarm', s=30)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 绘制决策边界
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1,
facecolors='none', edgecolors='k')
plt.title('Decision Boundary and Support Vectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()