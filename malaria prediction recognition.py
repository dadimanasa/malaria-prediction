
# coding: utf-8

# In[53]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[54]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_fd5fa09af502480094a9e83b6463ea0b = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='R41u6GBaapneRgP7nsnKV-sUIfjxn6MVV-ChNuhPGXBP',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_fd5fa09af502480094a9e83b6463ea0b.get_object(Bucket='malariaprediction-donotdelete-pr-k4mjie3jebxxhu',Key='malaria_prediction.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

malaria_dataset= pd.read_csv(body)
malaria_dataset.head()



# In[55]:


malaria_dataset.head()


# In[56]:


malaria_dataset.drop(['cold','hypoglycemia','hyperpyrexia','Convulsion','Anemia','prostraction'],axis=1,inplace=True)


# In[58]:


malaria_dataset


# In[5]:


malaria_dataset.shape


# In[6]:


malaria_dataset.isnull().any()


# In[7]:


values = {"fever":{"no":0, "yes":1},"rigor":{"no":0, "yes":1},"fatigue":{"no":0, "yes":1},
          "headace":{"no":0, "yes":1},"bitter_tongue":{"no":0, "yes":1},"vomitting":{"no":0, "yes":1},
          "diarrhea":{"no":0, "yes":1},"jundice":{"no":0, "yes":1},
          "cocacola_urine":{"no":0, "yes":1},
         "severe_maleria":{"no":0, "yes":1}}

malaria_dataset.replace(values,inplace=True)


# In[8]:


y= malaria_dataset['severe_maleria']


# In[9]:


y


# In[10]:


new_malaria = malaria_dataset.drop('severe_maleria', axis=1)


# In[11]:


new_malaria.head()


# In[12]:


new_malaria = pd.get_dummies(new_malaria)


# In[13]:


new_malaria


# In[14]:


new_malaria = new_malaria.drop('age', axis=1)


# In[15]:


new_malaria


# In[16]:


print(malaria_dataset.groupby('severe_maleria').size())


# In[17]:


import seaborn as sns

sns.countplot(malaria_dataset['severe_maleria'],label="Count")


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_malaria, y, test_size=0.3, random_state=0)


# In[19]:


X_train.shape


# In[20]:


y_train.values


# In[59]:


X_test.shape


# In[22]:


y_test.values


# In[23]:


from sklearn.tree import DecisionTreeClassifier
classifier1=DecisionTreeClassifier(criterion="entropy",random_state=0)


# In[24]:



classifier1.fit(X_train, y_train)


# In[25]:


y_predict=classifier1.predict(X_test)


# In[26]:


y_predict


# In[27]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[28]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
cm


# In[29]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict)
roc_auc=metrics.auc(fpr,tpr)
roc_auc


# In[30]:


plt.title("Reciver operating characteristics")
plt.plot(fpr,tpr,label="AUC=%0.2f"%roc_auc,color="blue")
plt.legend()
plt.show()


# # Random Forest

# In[31]:


from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)


# In[32]:


classifier.fit(X_train,y_train)


# In[33]:


y_predict=classifier.predict(X_test)
y_predict


# In[34]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[35]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
cm


# In[36]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict)
roc_auc=metrics.auc(fpr,tpr)
roc_auc


# In[37]:


plt.title("Reciver operating characteristics")
plt.plot(fpr,tpr,label="AUC=%0.2f"%roc_auc,color="blue")
plt.legend()
plt.show()


# In[38]:


x=['DT','RF']
y=[0.5,0.48]
plt.bar(x,y,label="plot")
plt.xlabel("Alg")
plt.ylabel("AUC")
plt.title("Multiple plots")
plt.legend()
plt.show()


# In[39]:


get_ipython().system(u'pip install watson_machine_learning_client --upgrade')


# In[40]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[41]:


wml_credentials={"instance_id": "3a1ac593-8c95-40df-8d57-d0a87953d7a6",
  "password": "a10bcd7d-9248-4393-ae8b-fa1d8a863f07",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "3a941747-4658-45b0-86e6-671b1287b7b5",
  "access_key": "u_4O8sYSLQ2_YZ7J23gbHMn7JJ4NxuNWL3lzJHUE1s7t"}


# In[42]:


client= WatsonMachineLearningAPIClient(wml_credentials)


# In[43]:


import json
instance_datails = client.service_instance.get_details()
print(json.dumps(instance_datails,indent=2))


# In[44]:


model_props= {client.repository.ModelMetaNames.AUTHOR_NAME:"manasa",
             client.repository.ModelMetaNames.AUTHOR_EMAIL:"dhadimsnasa@gmail.com",
             client.repository.ModelMetaNames.NAME:"Malariaprediction"}


# In[45]:


model_artifact = client.repository.store_model(classifier1,meta_props=model_props)


# In[46]:


published_model_uid = client.repository.get_model_uid(model_artifact)


# In[47]:


published_model_uid


# In[48]:


created_deployment = client.deployments.create(published_model_uid,name='Malariaprediction')


# In[49]:


scoring_endpoint= client.deployments.get_scoring_url(created_deployment)
scoring_endpoint


# In[50]:


client.deployments.list()

