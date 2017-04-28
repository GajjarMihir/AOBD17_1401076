import codecs 
from math import sqrt

class recommender:

    def __init__(self, data, k=1, metric='pearson', n=5):
        """ initialize recommender
        If data is dictionary the recommender is initialized to it.
        For all other data types of data, no initialization occurs
        k is the k value for k nearest neighbor
        metric is which distance formula to use
        n is the maximum number of recommendations to make"""
        self.k = k
        self.n = n
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
        # for some reason I want to save the name of the metric
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson
        #
        # if data is dictionary set recommender data to it
        #
        if type(data).__name__ == 'dict':
            self.data = data

    def convertProductID2name(self, id):
        """Given product id number return product name"""
        if id in self.productid2name:
            return self.productid2name[id]
        else:
            return id


    def skills(self, id, n):
        """Return n top skills for user with id"""
        
        skills = self.data[id]
       
        skills = list(skills.items())
        skills = [(self.convertProductID2name(k), v)
                   for (k, v) in skills]
        # finally sort and return
        skills.sort(key=lambda artistTuple: artistTuple[1],
                     reverse = True)
        skills = skills[:n]
                
        
    def pearson(self, rating1, rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        # now compute denominator
        denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n)
                       * sqrt(sum_y2 - pow(sum_y, 2) / n))
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator


    def computeNearestNeighbor(self, username):
        """creates a sorted list of users based on their distance to
        username"""
        distances = []
        for instance in self.data:
            if instance != username:
                distance = self.fn(self.data[username],
                                   self.data[instance])
                distances.append((instance, distance))
        # sort based on distance -- closest first
        distances.sort(key=lambda artistTuple: artistTuple[1],
                       reverse=True)
        return distances

    def recommend(self, user):
       """Give list of recommendations"""
       recommendations = {}
       # first get list of users  ordered by nearness
       nearest = self.computeNearestNeighbor(user)
       #
       # now get the skills for the user
       #
       skills = self.data[user]
       #
       # determine the total distance
       totalDistance = 0.0
       for i in range(self.k):
          totalDistance += nearest[i][1]
       # now iterate through the k nearest neighbors
       # accumulating their skills
       for i in range(self.k):
          # compute slice of pie 
          weight = nearest[i][1] / totalDistance
          # get the name of the person
          name = nearest[i][0]
          # get the skills for this person
          neighborskills = self.data[name]
          # get the name of the person
          # now find bands neighbor rated that user didn't
          for artist in neighborskills:
             if not artist in skills:
                if artist not in recommendations:
                   recommendations[artist] = (neighborskills[artist]
                                              * weight)
                else:
                   recommendations[artist] = (recommendations[artist]
                                              + neighborskills[artist]
                                              * weight)
       # now make list from dictionary
       recommendations = list(recommendations.items())
       recommendations = [(self.convertProductID2name(k), v)
                          for (k, v) in recommendations]
       # finally sort and return
       recommendations.sort(key=lambda artistTuple: artistTuple[1],
                            reverse = True)
       # Return the first n items
       return recommendations[:self.n]
	   

users = {"0": {"Oracle": 2.5,
                      "Microsoft Office": 3.0, "jboss": 3.0,
                      "AngularJS": 3.5,
                      "CCNA": 2.5,"MongoDB": 5.0},
         
         "2":{"Oracle": 2.0, "Matlab": 3.5,
                 "SUN OpenSSO": 3.5,
                 "AngularJS": 3.5, "Unix": 3.0,"MongoDB": 1.5},

         "4":{"Mapreduce": 2.0, "Perforce": 3.5,
                 "Microsoft Office": 4.0,
                 "Java": 3.5, "jboss": 5.0,"IBM HTTP Server": 4.0},

         "6": {"Oracle": 1.5,
                      "SUN OpenSSO": 3.0, "Ruby": 2.0,
                      "AngularJS":2.5,
                      "CCNA": 2.5,"Scripting": 3.5},
		 "8": {"Mapreduce": 2.0,
                      "SUN OpenSSO": 4.0, "Unix": 2.0,
                      "AngularJS": 3.5,
                      "CCNA": 2.5,"JIRA": 2.0},
         
         "9":{"Oracle": 2.0, "Matlab": 3.5,
                 "Microsoft Office": 3.0,
                 "Java": 3.5, "jboss": 3.0,"MongoDB": 3.5},

         "13":{
                 "SUN OpenSSO": 1.5,
                 "AngularJS": 3.5, "Unix": 5.0,"IBM HTTP Server": 4.0},

         "16": {"Oracle": 2.0,
                      "Microsoft Office": 2.0, "Ruby": 3.0,
                      "AngularJS": 2.5,
                      "CCNA": 1.5,"Scripting": 2.0},
		 "19": {"Mapreduce": 2.0,
                      "Microsoft Office": 4.0, "jboss": 5.0,
                      "Java": 3.5,
                      "CCNA": 2.5,"Git": 3.0},
         
         "21":{"Mapreduce": 2.0, "Matlab": 3.5,
                 "Microsoft Office": 4.0,
                 "AngularJS": 3.5, "Unix": 3.0,"MongoDB": 5.0},

         "24":{"Oracle": 2.0, "Perforce": 3.5,
                 "SUN OpenSSO": 2.5,
                 "AngularJS": 3.5, "Unix": 5.0,"IBM HTTP Server": 3.0},

         "26": {"Mapreduce": 3.0,
                      "Microsoft Office": 2.0, "Ruby": 5.0,
                      "Java": 2.5,
                      "CCNA": 1.5,"MongoDB": 2.0},
					  
		 "30": {"Oracle": 2.0,
                      "SUN OpenSSO": 3.5, "Unix": 2.0,
                      "AngularJS": 3.5,
                      "CCNA": 2.5,"jboss": 2.0},
         
         "32":{"Mapreduce": 2.0, "Matlab": 3.5,
                 "SUN OpenSSO": 4.0,
                 "Java": 3.5, "Unix": 3.0,"MongoDB": 5.0},

         "44":{"Oracle": 2.0, "Perforce": 3.5,
                 "Microsoft Office": 4.0,
                 "AngularJS": 3.5, "Unix": 5.0,"IBM HTTP Server": 4.0},
         
        }



