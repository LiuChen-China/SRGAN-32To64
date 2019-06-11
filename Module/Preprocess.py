# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import json
import cv2
import os

curDir = os.path.dirname(os.path.abspath(__file__))#当前目录
baseDir = os.path.dirname(curDir)#根目录
staticDir = os.path.join(baseDir,'Static')#静态文件目录
dbPath = os.path.join(staticDir,'db.sqlite3')#数据库路径
imgDir = os.path.join(staticDir,'Image')#图片目录
if os. path.exists(dbPath):
    os.remove(dbPath)
#创建表格
conn = sqlite3.connect(dbPath)
cursor = conn.cursor()
sql = '''
    create table img (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        desc text
        )
    '''
cursor.execute(sql)
conn.commit()

def normalizeImg(img):
    '''将图片标准化到-1,1'''
    img = (img/255 - 0.5)*2
    return img

#图片路径
imgPaths = [os.path.join(imgDir,path) for path in os.listdir(imgDir)]
#存入所有数据
for path in imgPaths:
    img64 = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    img64One = normalizeImg(img64)
    img32 = cv2.resize(img64,(32,32),)
    img32One = normalizeImg(img32)
    desc = {
            'img64One':img64One.tolist(),'img32One':img32One.tolist()}
    desc = json.dumps(desc)
    sql = "insert into img values(null,'%s')"%desc
    cursor.execute(sql)
conn.commit()
conn.close()
