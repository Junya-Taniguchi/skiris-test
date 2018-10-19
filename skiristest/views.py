from django.http import HttpResponse
from django.shortcuts import render
from sklearn import datasets
from sklearn import svm
import numpy as np

def home(request):
    return render(request, 'home.html')

def answer(request):
    sl = request.GET['sepallength']
    sw = request.GET['sepalwidth']
    pl = request.GET['petallength']
    pw = request.GET['petalwidth']
    sl = float(sl)
    sw = float(sw)
    pl = float(pl)
    pw = float(pw)


    #Irisの測定データの読み込み
    iris = datasets.load_iris()
    #線形サポートベクターマシン
    clf = svm.LinearSVC()
    #サポートベクターマシンによる訓練
    clf.fit(iris.data, iris.target)
    #種類の分類
    answer = (clf.predict([[sl,sw,pl,pw]]))
    if answer==1:
        answer='アイリスセトサ'
    elif answer==2:
        answer='アイリスバージカラー'
    else :
        answer='アイリスバージニカ'

    return render(request, 'answer.html',{'answer':answer})
