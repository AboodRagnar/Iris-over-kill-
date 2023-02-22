import pandas 
from keras.layers import Dense,Input,Activation,BatchNormalization
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
#from keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy
# lis=np.array(['a','d','t','t'])
# li1s=dict((k,v) for k,v in enumerate(lis))
# ll=[i for i in li1s ]
# print(ll)
# print(li1s)



Data=pd.read_csv('D:/AI/Keras/DataSet/Iris.csv')

labelEn=LabelEncoder()
HotEncoder=OneHotEncoder()
lableEn=labelEn.fit_transform(Data['Species'])
lableEn=lableEn.reshape(-1,1)
Y=HotEncoder.fit_transform(lableEn)
X=Data.drop(columns=['Species','Id'])

model=Sequential()

model.add(Dense(10,input_shape=(4,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(20,activation='relu'))


model.add(Dense(3))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X[:148],Y[:148],batch_size=10,epochs=30,verbose=1)

print(labelEn.inverse_transform(model.predict_classes(X[149:])))
print('actual y', Y[149:])
"""
Data=pd.read_csv('D:/AI/Keras/DataSet/Iris.csv')


Data.drop(columns=['Id'], inplace=True)
Data=shuffle(Data)
X=Data.drop(columns=['Species'])
Y=Data['Species']
Label=LabelEncoder()
y_pre=Y[:10]
Y=Label.fit_transform(Y)
Y=to_categorical(Y)



In_X=Input((4,))
L1=Dense(8,activation='relu')(In_X)
L1=Dropout(0.4)(L1)
L2=Dense(16,activation='relu')(L1)

output=Dense(3,activation='softmax')(L2)
model=Model(In_X,output)
model.compile(Adam(),loss='categorical_crossentropy', metrics=['accuracy'])
x_pre=X[:10]



model.fit(X[10:],Y[10:],batch_size=1,epochs=15)
print("Predicted output ",Label.inverse_transform(K.argmax(model.predict(x_pre,batch_size=10))))
print('*********************')
print("Acual output ",y_pre)
"""
