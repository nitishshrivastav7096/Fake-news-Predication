import os
import numpy as np
import pandas as pd
import re             #for searching in document
from nltk.corpus import stopwords  #nltk = natural language tool kit
from nltk.stem.porter import PorterStemmer #stem take word and remove prefix ans sufix of word and return root word of it
from sklearn.feature_extraction.text import TfidfVectorizer #conver text into no.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
#print(stopwords.words('english')) #printing stopword in English

def welcome():
    print("Hello ! Welcome in Fake News Predication System")
    print("Press ENTER key to proceed")
    input()

def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for x in content_list:
        if x.split('.')[-1]=='csv':
           csv_files.append(x)
    if len(csv_files)==0:
        return 'No csv file in the directory'
    else:
        return csv_files

def display_and_select_csv(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'...',file_name)
        i+=1
    return csv_files[int(input("Select file to create ML model"))]

    

def main():
    welcome()
    try:
        csv_files=checkcsv()
        if csv_files=='No csv file in the directory':
            raise FileNotFoundError('No csv file in the directory')
        csv_file=display_and_select_csv(csv_files)
        print(csv_file,' is selected')
        print("creating dataset")
        news_dataset=pd.read_csv(csv_file)
        print("dataset is created")
        print(news_dataset.shape)
        print(news_dataset.head())
        print(news_dataset.isnull().sum())   #find the no. of mission values
        news_dataset=news_dataset.fillna('') #replace the mission value in null place
        news_dataset['content']=news_dataset['author']+' '+news_dataset['title']
        print(news_dataset['content'])
        x=news_dataset.drop(columns='label',axis=1)
        y=news_dataset['label']

        #Stemming:
        '''Stemming is the process of reducing a word to its Root word
                example: acotr,acting---->act'''
        print("Steamming of data is started...")
        
        
        news_dataset['content']=news_dataset['content'].apply(stemming)
        print("Steamming of data is done")
        print(news_dataset['content'])
        x=news_dataset['content'].values
        y=news_dataset['label'].values

        #converting the textual data to numerical data
        vectorizer=TfidfVectorizer() #Tf use for change the repeated word into value #idf if reapeated word having no meaning # 
        vectorizer.fit(x)
        x=vectorizer.transform(x) #converting values in future

        #splitting the dataset to training & test data
        s=float(input("please insert the test data size"))
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=s, stratify=y,random_state=2)
        print("Model is creating...")
        model=LogisticRegression()
        model.fit(x_train,y_train)
        print("Model is created")

        #evelution
        #accuracy score in training data
        x_train_predication=model.predict(x_train)
        accuracy1=accuracy_score(x_train_predication,y_train)
        print("accuracy score for training data is %2.2f%%"%(accuracy1*100))
        #accuracy score on test data
        x_test_predication=model.predict(x_test)
        accuracy2=accuracy_score(x_test_predication,y_test)
        print("accuracy score for test data is %2.2f%%"%(accuracy2*100))

        #Making a Predication System

        '''new=input("Enter the News")
        new=stemming(new)
       
        vectorizer.fit(new)
        new=vectorizer.transform()'''
        result=x_test[3]
        result=model.predict(result)
        if result[0]==0:
            print("The news is Real")
        else:
            print("The news is fake")
        
                                                       
    except FileNotFoundError:
        print("No csv file in the directory")
        print("Press ENTER key to exit")
        exit()


def stemming(content):
    port_stem=PorterStemmer()
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=''.join(stemmed_content)
    return stemmed_content

                           
if __name__=='__main__':
    main()
    input()
        
    
