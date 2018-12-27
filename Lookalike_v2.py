# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 2018

@author: Yifan Peng for Lookalike Project using XGBoost Version 2
"""
import pandas as pd
import numpy as np
import os
import pandas as pd
import re
from xgboost.sklearn import XGBClassifier
from sklearn.utils import shuffle
os.chdir("/home/hadoop/sdl/hdfs_data/66")
listfile=os.listdir('lkd')

#reading seed data s1 for seed1, s2 for seed2
s1=pd.read_table("/home/hadoop/sdl/hdfs_data/64/lkd/seed/1_visible/part-00000",names=['userId'])
s2=pd.read_table("/home/hadoop/sdl/hdfs_data/64/lkd/seed/2_visible/part-00000",names=['userId'])

# reading device and location data
dev = pd.DataFrame(columns = ["userId", "device"])
aoi= pd.DataFrame(columns = ["userId", "location"])
for date in range(20170306, 20170320):
    
    daily_dev = pd.read_table("/home/hadoop/sdl/hdfs_data/64/lkd/" + str(date) + "/dev/part-00000", 
                              header = None, names = ["userId", "device"])
    daily_aoi = pd.read_table("/home/hadoop/sdl/hdfs_data/64/lkd/" + str(date) + "/aoi/part-00000", 
                              header = None, names = ["userId", "location"])
    dev = dev.append(daily_dev)
    aoi = aoi.append(daily_aoi)

#for location
df = aoi.drop('location', axis=1).join(aoi['location'].str.split(' ', expand=True).stack().reset_index(drop=True).rename('location'))
df['value'] = 1
df = df.drop_duplicates()
df = df.dropna()
df = df.reset_index(drop=True)
loc=df.pivot_table(index='userId',columns='location', values='value',fill_value=0) # pivot table for location

#for device
dev_k = dev.drop('device', axis=1).join(dev['device'].str.split(' ', expand=True).stack().reset_index(drop=True).rename('device'))
dev_k = dev_k.drop_duplicates()
dev_k = dev_k.dropna()
dev_k = dev_k.reset_index(drop=True)
dev_z = []
for i in range(0,len(dev_k)): 
    if re.search('(iPhone|iPad)', dev_k['device'][i]):
        dev_z.append('apple')
    elif re.search('(HUAWEI|-TL|-AL|-CL|HONOR|H60-L03|P7|-UL|huawei|G750|C199|P8max)', dev_k['device'][i]):
        dev_z.append('huawei')
    elif re.search('(OPPO|oppo|R7|R8|A31|1107|A33|R9|A53|A59|A37|Plustm)', dev_k['device'][i]):
        dev_z.append('oppo')
    elif re.search('(Xiaomi|2014812|2014813|MI|HM|2014|Mi|Redmi|4LTE|1LTE|4A|2A)', dev_k['device'][i]):
        dev_z.append('xiaomi')
    elif re.search('(vivo|Vivo|VIVO|X5|X6|X9|Xplay|Y51|V3|BBK|Y66|Y67L|Y23L|Y31|Y33|Y37|Y27|Y13|Y927)', dev_k['device'][i]):
        dev_z.append('vivo')
    elif re.search('(samsung|SM|GT)', dev_k['device'][i]):
        dev_z.append('samsung')
    elif re.search('(Meizu|MX4|M355|metal|meizu|m2|MX6)', dev_k['device'][i]):
        dev_z.append('meizu')
    elif re.search('(Lenovo|ZUK|A788t|A320t|K30)', dev_k['device'][i]):
        dev_z.append('lenovo')
    elif re.search('ONEPLUS', dev_k['device'][i]):
        dev_z.append('oneplus')
    elif re.search('(ZTE|leimin)', dev_k['device'][i]):
        dev_z.append('zte')
    elif re.search('(nubia)', dev_k['device'][i]):
        dev_z.append('nubia')
    elif re.search('(Le|LeMobile|X500|X501|X608|X620|X520)', dev_k['device'][i]):
        dev_z.append('letv')
    elif re.search('(GiONEE|GIONEE|GN|F100|gionee)', dev_k['device'][i]):  
        dev_z.append('gionee')
    elif re.search('(8675-A|Coolpad|8712|Y82|5360)', dev_k['device'][i]):
        dev_z.append('coolpad')
    elif re.search('QiKU', dev_k['device'][i]):
        dev_z.append('qiku')
    elif re.search('(Sony|sony)', dev_k['device'][i]):
        dev_z.append('sony')       
    else:
        dev_z.append('other')
dev_k['device']=dev_z
dev_k['value']=1
dev_k=dev_k.reset_index()
dev_k=dev_k.drop('index',1)
dev_L=dev_k.pivot_table(index='userId',columns='device', values='value',fill_value=0) # pivot table for device
dev_L=dev_L.reset_index()

# read app data
appid_tag=pd.read_table("lkd/appid_tag/part-00000",names=['app_id','tag_id'])
ap_tags={} #dict
tagids=[] #list

file=open("lkd/appid_tag/part-00000",'r')
for i in open("lkd/appid_tag/part-00000",'r'):  
    l=file.readline() 
    l=l.strip().split('\t')
    if l[0] not in ap_tags:
        ap_tags[l[0]]=[]   
    for j in l[1].split(' '):
        if j not in ap_tags[l[0]]:
            ap_tags[l[0]].append(j)
        if j not in tagids:
            tagids.append(j)
appid_ct={} #counts
app={}
app_days={} 
for d in listfile[:14]:
    file=open("lkd/"+d+"/app/part-00000",'r')
    for i in open("lkd/"+d+"/app/part-00000",'r'):
        l=file.readline()
        l=l.strip().split('\t')
        if l[0] not in app:
            app[l[0]]={}
        if l[0] in app_days:
            app_days[l[0]]['days']+=1
            app_days[l[0]]['app_n']+=len(l[1].split(' '))
        else:
            app_days[l[0]]={'days':1}
            app_days[l[0]]['app_n']=len(l[1].split(' '))
        for x in l[1].split(' '):
            if x in app[l[0]]:
                app[l[0]][x]+=1
            else:
                app[l[0]][x]=1
            if x in appid_ct:
                appid_ct[x]+=1
            else:
                appid_ct[x]=1
tag={}
tagid_ct={}
for tdid in app:
    tag[tdid]={}
    for appid in app[tdid]:
        if appid in ap_tags:
            for tagid in ap_tags[appid]:
                if tagid in tag[tdid]:
                    tag[tdid][tagid]+=app[tdid][appid]
                else:
                    tag[tdid][tagid]=app[tdid][appid]
                if tagid in tagid_ct:
                    tagid_ct[tagid]+=app[tdid][appid]
                else:
                    tagid_ct[tagid]=app[tdid][appid]  

appids=[x for x in appid_ct]
appid=pd.DataFrame({'appid':appids,'freq':[appid_ct[x] for x in appids]})
tagid=pd.DataFrame({'tagid':tagids,'freq':[tagid_ct[x] for x in tagids]})
appid['tag']=[1 if appid['appid'][i] in ap_tags else 0 for i in range(appid.shape[0])]
appid=appid.sort_values('freq',ascending=False)
tagid=tagid.sort_values('freq',ascending=False)
appid_nt=appid[appid['tag']==0]  

tn=pd.read_csv('/home/hadoop/sdl/hdfs_data/64/lkd/tag_name.csv') #read tag name
tag_dict={}
tagid['tagid']=tagid['tagid'].astype(int)
for i in range(tn.shape[0]):
    if tn['tagId'][i] in set(tagid['tagid'][:80]): #tag keep 80
        tag_dict[str(tn['tagId'][i])]=tn['name'][i]
                 
app_tdid=[x for x in app_days]
app_al=pd.DataFrame({'userId':app_tdid})
app_al['app_days']=[app_days[i]['days'] for i in app_tdid]
app_al['app_n']=[app_days[i]['app_n'] for i in app_tdid]
app_al['app_n']=app_al['app_n']/app_al['app_days']
app_al['tag_n']=[len(tag[i]) for i in app_tdid]

for x in tag_dict:
    app_al[tag_dict[x]]=[tag[i][x] if x in tag[i] else 0 for i in app_tdid]
    
appid_nt=appid_nt.reset_index()
for z in range(300): # if no tag, keep 300
    x=appid_nt['appid'][z]
    app_al[x]=[app[i][x] if x in app[i] else 0 for i in app_tdid]

# merge data
dev_L['userId']=dev_L['userId'].astype(int)
app_al['userId']=app_al['userId'].astype(int)
a1=pd.merge(app_al,dev_L,how="right",on="userId")
loc1=loc.reset_index()
loc1['userId']=loc1['userId'].astype(int)
loc1=loc1.reset_index()
loc1=loc1.drop('index',1)
a2=pd.merge(a1,loc1,how='left')
a2 = a2.fillna(0)

s1_id=set(s1['userId']) #seed1
s2_id=set(s2['userId']) #seed2
a2['s1']=[1 if x in s1_id else 0 for x in a2['userId']]
a2['s2']=[1 if x in s2_id else 0 for x in a2['userId']]
a2.set_index('userId')

# categorize data
index=[x for x in a2.columns if x not in ['userId','s1','s2','s1_prediction','s2_prediction','p_n','n']] 
s1=a2[a2['s1']==1]
s2=a2[a2['s2']==1]
t1=a2[a2['s1']==0] #test 1
t2=a2[a2['s2']==0] #test 2

# predict seed1
t1['s1_prediction']=0
t1['p_n']=0
s=s1.shape[0]
for i in range(20):
    if i%3==0:
        print(i)
    t1=shuffle(t1) 
    model_1 = XGBClassifier(n_jobs=-1,n_estimators=200)
    xtrain=pd.concat([s1,t1.iloc[:s,:]])
    model_1.fit(xtrain[index],xtrain['s1'])
    pred=model_1.predict_proba(a2[index])[:,1]
    a2['s1_prediction'][s:]+=pred
    a2['p_n'][s:]+=1
    
a2['s1_prediction']=a2['s1_prediction']/a2['p_n'] # Average













