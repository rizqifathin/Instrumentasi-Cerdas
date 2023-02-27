import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

data = pd.read_excel('/home/rizqi/Downloads/data sensor tgs.xlsx')

sens = ["TGS 2610 (mV)","TGS 2602 (mV)","TGS 2620 (mV)","TGS 2600 (mV)","TGS 2611 (mV)","kelas"]

nfitur = 6
nr = len(data.index)
nc = len(data.columns)
samp = 300
rows = int(nr/samp)
cols = int((nc-1)*nfitur) 
ciri = [[0]*cols for _ in range(rows)]
maks = [[0]*(nc-1) for _ in range(rows)] 
kls = np.zeros(rows)
rank = np.zeros(30)

j = 0
k = 0
m = 0
temp = 0
#Ekstraksi Ciri 
for i in range(nr) :
	if ((i+1) % 300) == 0 :
		for b in range(nc-1) :
			maks[j][k] = data[sens[k]][temp:i+1].max()

			ciri[j][m] = data[sens[k]][temp:i+1].mean()
			ciri[j][m+1] = data[sens[k]][temp:i+1].var()
			ciri[j][m+2] = maks[j][k] - data[sens[k]][temp:i+1].min()
			ciri[j][m+3] = data[sens[k]][temp:i+1].sum() 
			ciri[j][m+4] = maks[j][k]/data[sens[k]][temp:i+1].min()
			ciri[j][m+5] = ciri[j][m+2]/data[sens[k]][temp:i+1].min()
			k = k+1
			m = m+6 
			
		kls[j] = data[sens[5]][i]
		temp = i+1 
		j = j+1
		k = 0
		m = 0

dataMaks = pd.DataFrame(maks,columns=sens[0:5])
dataMaks.insert(5,"Kelas",kls,True)
print("Data Maksimal :\n\n",dataMaks) 

header = pd.MultiIndex.from_product([["TGS 2610 (mV)","TGS 2602 (mV)","TGS 2620 (mV)","TGS 2600 (mV)","TGS 2611 (mV)"],
	["mean","variance","integral","difference","relative","fractional"]])
fitur = pd.DataFrame(ciri,columns=header)
fitur.insert(30,"Kelas",kls,True)
fitur.to_excel("Data fitur.xlsx")

#Feature Ranking (untuk menentukan kepekaan sensor terhadap tahu berformalin/tidak)
x = fitur.iloc[:, :-1].values
y = fitur.iloc[:, 30].values
for i in range(10):
	model_rank = ExtraTreesClassifier(n_estimators=100)
	model_rank.fit(x,y)
	rank = np.add(rank,model_rank.feature_importances_)

rank_avg = rank/10
rank_sen_avg = np.zeros(5)
b = 0
c = 6

for i in range(5):
	rank_sen_avg[i] = sum(rank_avg[b:c])
	b = b+6
	c = c+6

print("\nKepekaan Sensor : \n",rank_sen_avg,"\n")

#Memisahkan data test dan data train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

pca_model = PCA(n_components=2)
pca_model.fit(x_train)
x_train_pca = pca_model.transform(x_train)
x_test_pca = pca_model.transform(x_test)

#KNN Tanpa PCA
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

acc = accuracy_score(y_test,y_pred)
print("====================KNN tanpa reduksi PCA====================")
cv_scores = cross_val_score(knn, x_train, y_train, cv=5)
print("Accuracy cross validation:",cv_scores)
print('Accuracy cross validation mean:', np.mean(cv_scores))

print("\nHasil pengujian menggunakan data uji")
print("Y true : ",y_test)
print("Y pred : ",y_pred,"\n")
print(classification_report(y_test, y_pred))

#KNN Dengan PCA
knn_pca = KNeighborsClassifier(n_neighbors = 3)
knn_pca.fit(x_train_pca, y_train)
y_pred_pca = knn_pca.predict(x_test_pca)

acc_pca = accuracy_score(y_test,y_pred_pca)
print("\n====================KNN dengan reduksi PCA====================")
cv_scores_pca = cross_val_score(knn_pca, x_train_pca, y_train, cv=5)
print("Accuracy cross validation PCA:",cv_scores_pca)
print('Accuracy cross validation mean PCA:', np.mean(cv_scores_pca))

print("\nHasil pengujian menggunakan data uji")
print("Y true : ",y_test)
print("Y pred : ",y_pred_pca,"\n")
print(classification_report(y_test, y_pred))
