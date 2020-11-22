import random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(1234)

class dataPreprocess(object):
    
    def __init__(self, filepath):
        '''
        Description:
            Initialization/Constructor function
        '''
        self.filepath = filepath
    
    def preprocess(self):
        '''
        Description:
            Function to preprocess the data and store as '.pkl' files
        INPUT:
            Path to the dataset folder (keep the dataset as 'ratings_data.txt' for ratings data and 'trust_data.txt' for social trust data
        OUTPUT:
            This function doesn't return anything but stores the data is '.pkl' files at the same folder.
        '''
 
        ratingsData = np.loadtxt(self.filepath+'/ratings_data.txt', dtype = np.int32)
        trustData = np.loadtxt(self.filepath+'/trust_data.txt', dtype = np.int32)
        
        ratingsList = []
        trustList = []

        users = set()
        items = set()

        for row in ratingsData:
            userId = row[0]
            itemId = row[1]
            rating = row[2]
            if userId not in users:
                users.add(userId)
            if itemId not in items:
                items.add(itemId)
            ratingsList.append([userId,itemId,rating])
        
        userCount = len(users)
        itemCount = len(items)

        for row in trustData:
            user1 = row[0]
            user2 = row[1]
            trust = row[2]
            trustList.append([user1, user2, trust])

        newDF = pd.DataFrame(ratingsList, columns=['userId','itemId','rating'])
        X = np.array([newDF['userId'],newDF['itemId']]).T
        y = np.array([newDF['rating']]).T

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        train = pd.DataFrame(X_train,columns = ['userId','itemId'])
        train['rating'] = pd.DataFrame(y_train)
        test = pd.DataFrame(X_test,columns = ['userId','itemId'])
        test['rating'] = pd.DataFrame(y_test)

        trainUsers = []
        trainItems = []
        trainRatings = []
        for index in range(len(train)):
            trainUsers.append(train['userId'][index])
            trainItems.append(train['itemId'][index])
            trainRatings.append(train['rating'][index])
            
        testUsers = []
        testItems = []
        testRatings = []
        for index in range(len(test)):
            testUsers.append(test['userId'][index])
            testItems.append(test['itemId'][index])
            testRatings.append(test['rating'][index])
        
        userItemDict = {}
        for index in range(len(train)):
            if train['userId'][index] not in userItemDict:
                userItemDict[train['userId'][index]] = [train['itemId'][index]]
            else:
                userItemDict[train['userId'][index]].append(train['itemId'][index])

        userRatings = {}
        for index in range(len(train)):
            if train['userId'][index] not in userRatings:
                userRatings[train['userId'][index]] = [train['rating'][index]]
            else:
                userRatings[train['userId'][index]].append(train['rating'][index])        
        
        itemUserDict = {}
        for index in range(len(train)):
            if train['itemId'][index] not in itemUserDict:
                itemUserDict[train['itemId'][index]] = [train['userId'][index]]
            else:
                itemUserDict[train['itemId'][index]].append(train['userId'][index])

        itemRatings = {}
        for index in range(len(train)):
            if train['itemId'][index] not in itemRatings:
                itemRatings[train['itemId'][index]] = [train['rating'][index]]
            else:
                itemRatings[train['itemId'][index]].append(train['rating'][index])

        trust = pd.DataFrame(trustList, columns=['userId','friendID','trust'])

        userUserDict = {}
        for index in range(len(trust)):
            if trust['userId'][index] not in userUserDict:
                userUserDict[trust['userId'][index]] = {trust['friendID'][index]}
            else:
                userUserDict[trust['userId'][index]].add(trust['friendID'][index])

        ratings = []
        for i in userRatings.keys():
            ratings.append(userRatings[i])
        r = [i for row in ratings for i in row]
        r = list(set(r))
        ratingsL = {}
        for i in range(1,len(r)+1):
            if i not in ratingsL.keys():
                ratingsL[i] = r[i-1]
            else:
                continue

        user = []
        friend = []
        trusts = []
        for index in range(len(trust)):
            user.append(trust['userId'][index])
            friend.append(trust['friendID'][index])
            trusts.append(trust['trust'][index])
        user = list(set(user))
        friend = list(set(friend))

        with open(self.filepath+'/dataset.pickle', 'wb') as files:
            pickle.dump(userItemDict, files, pickle.HIGHEST_PROTOCOL)
            
            pickle.dump(userRatings, files, pickle.HIGHEST_PROTOCOL)
            
            pickle.dump(itemUserDict, files, pickle.HIGHEST_PROTOCOL)
            
            pickle.dump(itemRatings, files, pickle.HIGHEST_PROTOCOL)
            
            pickle.dump(trainUsers, files, pickle.HIGHEST_PROTOCOL)
            pickle.dump(trainItems, files, pickle.HIGHEST_PROTOCOL)
            pickle.dump(trainRatings, files, pickle.HIGHEST_PROTOCOL)
            
            pickle.dump(testUsers, files, pickle.HIGHEST_PROTOCOL)
            pickle.dump(testItems, files, pickle.HIGHEST_PROTOCOL)
            pickle.dump(testRatings, files, pickle.HIGHEST_PROTOCOL)

            pickle.dump(userUserDict, files, pickle.HIGHEST_PROTOCOL)

            pickle.dump(ratingsL, files, pickle.HIGHEST_PROTOCOL)

            pickle.dump(user, files, pickle.HIGHEST_PROTOCOL)
            pickle.dump(friend, files, pickle.HIGHEST_PROTOCOL)
            pickle.dump(trust, files, pickle.HIGHEST_PROTOCOL)
            