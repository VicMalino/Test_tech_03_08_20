import numpy as np
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def test():
    # I) Dataset load + clean
    names = ["INDEX","TARGET_FLAG","TARGET_AMT","KIDSDRIV","AGE","HOMEKIDS","YOJ","INCOME","PARENT1","HOME_VAL","MSTATUS","SEX","EDUCATION","JOB","TRAVTIME","CAR_USE","BLUEBOOK","TIF","CAR_TYPE","RED_CAR","OLDCLAIM","CLM_FREQ","REVOKED","MVR_PTS","CAR_AGE","URBANICITY"]
    dataset = read_csv("train_auto.csv", sep=",",  header=0, names=names)
    print(dataset.head(20))
       
        #devise -> nombre
    #Income
    dataset["INCOME"] = dataset["INCOME"].str.replace(',', '')
    dataset["INCOME"] = dataset["INCOME"].str.replace('$', '')
    #HomeVal
    dataset["HOME_VAL"] = dataset["HOME_VAL"].str.replace(',', '')
    dataset["HOME_VAL"] = dataset["HOME_VAL"].str.replace('$', '')
    #BLUEBOOK
    dataset["BLUEBOOK"] = dataset["BLUEBOOK"].str.replace(',', '')
    dataset["BLUEBOOK"] = dataset["BLUEBOOK"].str.replace('$', '')
    #OLDCLAIM
    dataset["OLDCLAIM"] = dataset["OLDCLAIM"].str.replace(',', '')
    dataset["OLDCLAIM"] = dataset["OLDCLAIM"].str.replace('$', '')
    
        #Drop Na
    dataset = dataset.dropna(axis=0)
       
        #String into integer
    #Parents
    dataset["PARENT1"] = dataset["PARENT1"].map({'Yes': 1, 'No': 0}).astype(int)
    #MStatus
    dataset["MSTATUS"] = dataset["MSTATUS"].map({'Yes': 1, 'z_No': 0}).astype(int)
    #Sexe
    dataset["SEX"] = dataset["SEX"].map({'M': 1, 'z_F': 0}).astype(int)
    #Eductaion
    dataset["EDUCATION"] = dataset["EDUCATION"].map({'<High School': 0, 'z_High School': 1, 'Bachelors': 2, 'Masters': 3, 'PhD':4}).astype(int)
    #Job
    dataset["JOB"] = dataset["JOB"].map({'Student': 0, 'Lawyer': 1, 'Clerical': 2, 'z_Blue Collar': 3, 'Doctor':4, 'Home Maker':5, 'Professional':6, 'Manager':7}).astype(int)
    #Car Use
    dataset["CAR_USE"] = dataset["CAR_USE"].map({'Private': 1, 'Commercial': 0}).astype(int)
    #Red Car
    dataset["RED_CAR"] = dataset["RED_CAR"].map({'yes': 1, 'no': 0}).astype(int)
    #Revoked
    dataset["REVOKED"] = dataset["REVOKED"].map({'Yes': 0, 'No': 1}).astype(int)
    #Car Type
    dataset["CAR_TYPE"] = dataset["CAR_TYPE"].map({'Minivan': 0, 'z_SUV': 1, 'Sports Car': 2, 'Panel Truck': 3, 'Van': 4, 'Pickup':5}).astype(int)
    #URBANICITY
    dataset["URBANICITY"] = dataset["URBANICITY"].map({'Highly Urban/ Urban': 0, 'z_Highly Rural/ Rural': 1}).astype(int)
    
    
    # II) Training
    array = dataset.values
    X = array[:,0:26]
    y = array[:,0:26]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.25, random_state=1)
    
    # III) Which model
    #on essaye plusieurs modeles pour voir
    modeles = []
    modeles.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    modeles.append(('LDA', LinearDiscriminantAnalysis()))
    modeles.append(('KNN', KNeighborsClassifier()))
    modeles.append(('CART', DecisionTreeClassifier()))
    modeles.append(('NB', GaussianNB()))
    
    # IV) Evaluate models
    #On utilise la cross evaluation pour determiner quel modele est le plus pertinent
    resultats = []
    names = []
    
    for nom, modele in modeles:
                kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
                crossval_resultats = cross_val_score(modele, X_train, Y_train, cv=kfold, scoring='accuracy')
                print(crossval_resultats)
                results.append(crossval_resultats)
                nom.append(nom)
                print('%s: %f (%f)' % (nom, crossval_resultats.mean(), crossval_resultats.std()))
    
    # V) Choose model
    model = GaussianNB()
    model.fit(X_train, Y_train)
    
    prediction = model.predict()
    print("Predicted Value: %s" % prediction)
    
print(test())
