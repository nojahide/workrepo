import os
import fnmatch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from skimage import io
import pickle


#顔判別器の作成
images = []
labels = []
filenames = []
fileindexies = []

path = r"xxxxx"

i = 0   #ファイル名があとで紐づけできるように番号振っておく

for f in os.listdir(path):
    
    image_path = os.path.join(path, f)
    
    if fnmatch.fnmatch(f, '*.jpg'):
        #グレースケールで読み込む(convert(L)でグレースケール)
        #gray_image = Image.open(image_path).convert(L)
        ##numpy配列に格納
        #image = np.array(gray_image,uint8)

        gray_image = io.imread(image_path)

        #imageを1次元配列に変換
        image = gray_image.flatten()

        #images[]にimageを格納
        images.append(image)

        #ファイル名とインデックスを記憶
        filenames.append(f)
        fileindexies.append(i)
        i += 1
        
        #ファイル名からラベルを取得
        labels.append(str(f[0:3]))

print("labels(1): ", type(labels))
print("images(1): ", type(images))

#行列に変換
labels = np.array(labels)
images = np.array(images)
fileindexies = np.array(fileindexies)


#学習用評価用に分ける前にimageの配列にファイルindexを結合
fileindexies.reshape(fileindexies.size, -1)
for_split_X = np.column_stack([images, fileindexies])

# split into a training and testing set 学習用評価用に分ける
X_train, X_test, y_train, y_test = train_test_split(
    for_split_X, labels, test_size=0.25, random_state=42)

# 学習用評価用に分けたものからファイルindexを分離＆削除
fileindex_train = X_train[:,-1]
fileindex_test = X_test[:,-1]

X_train = np.delete(X_train, -1, 1)    #np.delete(対象array, 対象index, 対象次元)
X_test = np.delete(X_test, -1, 1)    #np.delete(対象array, 対象index, 対象次元)

#指定列だけとりだす(ちょっとデバッグでデータ覗いてみる)
X_train_col = X_train[:,1851]
X_train_col2 = X_train[:,1852]

# k 分割交差検証　デフォルト 5分割

#SDG分類クラス
#from sklearn import svm
#clf_svm = svm.SVC()

#  svm LinearSVC  
clf_linerSVC = svm.LinearSVC()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf_linerSVC, X_train, y_train)
print('Cross-Validation scores {}'.format(scores))

print('Cross-Validation average {}'.format(np.average(scores)))

#------------------
# 学習
#------------------
clf_linerSVC.fit(X_train,y_train)

LinerSCVCoef = clf_linerSVC.coef_
#指定列だけとりだす(ちょっとデバッグでデータ覗いてみる)
coef_col1 = LinerSCVCoef[:,1851]
coef_col2 = LinerSCVCoef[:,1852]

#学習モデルを保存する
filename = "face_model.sav"
pickle.dump(clf_linerSVC,open(filename,"wb"))

print("save the model to file.")

#正答率を求める
y_pred = clf_linerSVC.predict(X_test)
ac_score = metrics.accuracy_score(y_test,y_pred)
print("正答率 = ",ac_score)

#さて認識失敗したファイルを探しましょうかね
i = 0 #index用
incorr_fname = []
for pred in y_pred:
    if pred != y_test[i]:   #不正解な奴
        print("incorrect data: ", filenames[fileindex_test[i]], ", index:", i, "pred=", pred)
        incorr_fname.append(filenames[fileindex_test[i]])
    i += 1    

#print(classification_report(y_test, y_pred, target_names=target_names))
#print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

#result = [y_test, y_pred]
