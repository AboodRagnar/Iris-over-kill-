import pandas as pd
from keras.layers import Activation
from keras.models import Sequential
from  keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import  matplotlib.pyplot as plot
from keras.models import load_model
import seaborn as sns
pd.set_option('display.width',500)
pd.set_option('display.max_rows',15)
pd.set_option('display.max_columns',11)
Normalizing=MinMaxScaler(feature_range=(0,1))

class NNClassification:
    def __init__(self,RawDataPath):
      self.Data=shuffle(pd.read_csv(RawDataPath))
      self.X=[]
      self.Y=[]
      self.XSamples=[]
      self.YSamples=[]

    def IntitiateData(self,Lable):
        
        x=self.Data.drop(columns=[Lable])
        y=self.Data[Lable]
        self.X=x.iloc[:1591]
        self.Y=y.iloc[:1591]
        self.XSamples=x.iloc[1591:]
        
        self.YSamples=y.iloc[1591:]



       




Getnn=NNClassification('D:/AI/Keras/DataSet/winequality_red.csv')
Getnn.IntitiateData('quality')


model=Sequential([


Dense(22,input_shape=(11,),activation='relu'),
Dense(44,activation='relu'),
Dense(1,activation='relu')
])


model.compile(optimizer=Adam(0.01), loss='MSE',metrics=['MSE'])
model.fit(Getnn.X,Getnn.Y,validation_split=0.1,batch_size=15,epochs=600)
model.save('WineQualityPrediction.h1')
model.ev
M=load_model('WineQualityPrediction.h1')
print(Getnn.YSamples,' Acuall value')
print(M.predict(Getnn.XSamples, batch_size=10,verbose=0)," Predict")