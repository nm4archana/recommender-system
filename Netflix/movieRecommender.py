"""
Date   : Nov 19 2017
@author: Archana Neelipalayam Masilamani
Task Description:
Recommend movies by Collaborative Filtering method for users by
using Matrix Factorizarion - Alternating Least Square method.
This project is implemented using  spark MLlib library and Python

The dataset used is Netflix data
"""

# Loading the dataset
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys

sc = SparkContext()
#Load the ratings file
#data = sc.textFile("/home/aneelipa/netflix-prize-data/small_ratings_dataset.txt")
data = sc.textFile(sys.argv[1])

header = 0
def parseLine(line):
    global header 
    length = len(line)
    if length==1:
        line = str(line[0])
        header = line.replace(':','')
    else:
        return int(line[0]),int(header),float(line[1])
    
def mapData(data):
    rdd =  data.map(lambda line: line.split(",") )

    #Removing Timestamp column and haveing only userId, MovieId and Ratings
    rating_data = rdd.map(parseLine)
    rating_data = rating_data.filter(lambda line: line is not None  and line[1]is not 0)
    return rating_data

rating_data = mapData(data)
rating_data.cache()


# Load movies
#data = sc.textFile("/home/aneelipa/netflix-prize-data/movie_titles.csv")
data = sc.textFile(sys.argv[3])

def loadMovies(data):
    rdd = data.map(lambda line: line.split(',') )
    movie_data = rdd.map(lambda line: (int(line[0]),line[2]))
    return movie_data

movie_data = loadMovies(data)


#Viewing the data
def viewData(rating_data):
    total_ratings = rating_data.count()
    total_movies = rating_data.map(lambda line: (line[1])).distinct().count()
    total_customers = rating_data.map(lambda line: (line[0])).distinct().count()
    print("No. of movies rated: ",total_movies)
    print("No. of ratings given: ",total_ratings)
    print("No. of customers:",total_customers)
    return total_ratings,total_movies,total_customers

viewData(rating_data)

#Creating a recommendation model based on the training data
rank_val = [5,10,15]
iterations = 5
#model.productFeatures().first()
#model.userFeatures().first()

# Splitting the data into train and test - 80% and 20%
(training, test) = rating_data.randomSplit([0.8, 0.2])
testdata = training.map(lambda p: (p[0], p[1]))


minimum_error = float('inf')
best_rank = 0

def trainModel(training, rank, iterations):
    model = ALS.train(training, rank, iterations)
    #Predict the ratings for all the Test Data
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = rating_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    #Compute RMSE for the predicted and the actual movie ratings
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Rank: " + str(rank))
    print("Mean Squared Error " + str(MSE))
    rmse = MSE**0.5
    print("Root Mean Squared Error = " + str(rmse))
    return rmse,model

for rank in rank_val: 
    rmse,model = trainModel(training, rank, iterations) 
    #Compute the minimum RMSE to find the best rank
    if minimum_error>rmse:
        rmse = minimum_error
        best_rank = rank
        
print("The best rank is: "+str(best_rank))
    

# For the large dataset, we have to read the training and testing data
# once again, transform it and then train a model using the best rank
# For small dataset, we import the same data with which we found the best model
# The purpose of this is to test it in local mode

#Train the model with best rank 

#data = sc.textFile("/home/aneelipa/netflix-prize-data/small_ratings_dataset.txt")
data = sc.textFile(sys.argv[2])

rating_data = mapData(data)
total_ratings,total_movies,total_customers = viewData(rating_data)


#Viewing how the data is spread - Ratings distribution 
rating_distr = rating_data.map(lambda row: (row[2], 1)).groupByKey().mapValues(sum)
rating_distr = rating_distr.map(lambda row:(row[0],(round((row[1]/total_ratings)*100,2))))


# Splitting the data into train and test - 80% and 20%
(training, test) = rating_data.randomSplit([0.8, 0.2])
testdata = training.map(lambda p: (p[0], p[1]))
#Training model with best rank
rmse,model = trainModel(training, best_rank, iterations)
#model = ALS.train(training, best_rank, iterations)

#user_id = 822109
#no_of_movies = 15
user_id = int(sys.argv[4])
no_of_movies = int(sys.argv[5])


def dropMv(line):
    if line[0] == user_id:
        return line

#Filter the movie ID's not rated and predict the ratings for those movies
user_rated_data = rating_data.map(dropMv)
user_rated_data = user_rated_data.filter(lambda line: line is not None)
user_rated_mvid = list(user_rated_data.map(lambda line: int(line[1])).distinct().toLocalIterator())
predict_data = rating_data.filter(lambda line: line[1] not in user_rated_mvid).map(lambda line: line[1]).distinct()
predict_data = predict_data.map(lambda line: (user_id,line))


#Predicting ratings for movies not rated by the user
predicted_val = model.predictAll(predict_data)
predicted_val_sorted = predicted_val.sortBy(lambda line: line[2],ascending=False)

#Sorting the movies based on predicted ratings to find the top recommendations
recommendedMovieID = predicted_val_sorted.map(lambda l: (l[1],l[2])).take(no_of_movies)
recommendedMovieID = sc.parallelize(recommendedMovieID)

new_list = []
print("Top fifteen movie recommendations: \n")

for i in list(recommendedMovieID.toLocalIterator()):
    a = str(movie_data.lookup(int(i[0]))).strip('[\'\"]')
    b=i[1]
    print(a,b)
    new_list.append(a+", "+str(b))
       
result = sc.parallelize(new_list)
#result.saveAsTextFile('/home/aneelipa/netflix-prize-data/output')
result.saveAsTextFile(sys.argv[6])

"""
#Join the movies dataset and top movie id's tp print the movie names along with predicted ID's
result = movie_data.join(recommendedMovieID)
result = result.map(lambda line: line[1]).sortBy(lambda line: line[1],ascending=False)

#Print the top movies
print("Top",no_of_movies,"movie recommendation for User ID is: ",user_id,"\n")
print("Movie Name ","Rating","\n")
list_result = list(result.toLocalIterator())
for i in list_result:
    print(i[0].strip('[\'\"]'),i[1]);

#result.saveAsTextFile('/home/aneelipa/netflix-prize-data/output')
result.saveAsTextFile(sys.argv[6])

"""
