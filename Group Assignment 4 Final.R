#STEP 1: data collection
wbcd <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"), header=FALSE)
# the database above does not having header
# the order of features/columns is different from the one below using local .csv file

# we load the data from working directory 
wbcd <- read.csv("wdbc.csv", header=TRUE)
# if you received the error: the file cannot be connected, following the following procedure
# if you are able to load the data successfully, jump to "STEP 2" 
# check and update your working directory by the following instruction
getwd()
# !!! then go to menu "Sessions-> Set Working Directory" to set the working diretory 
# now check current directory again to see if it is working directory (containing your files)
getwd()
# reload the data 
wbcd <- read.csv("Downloads/wdbc.csv", header=TRUE)


#STEP 2: exploring and preparing data 
names(wbcd)
summary(wbcd)
str(wbcd)  # watch out: variable X is NA
wbcd <- wbcd[, -33]

#remove medical ID number. patient ID number is the first attribute in the dataset.
wbcd = wbcd[ , -1] 
str(wbcd)   #sucessfully removed attribute X that is NA

# now display attributes, ID is gone:
names(wbcd)
# display dimenstion of the data:
dim(wbcd)

#explanation of normalization:
x = c(1, 2, 3,4, 5)
x.normalized = (x-min(x))/(max(x)-min(x))
x.normalized
#define function normalize: 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
y = c(2,3,4,5,6)
y.n= normalize(y)
y.n

#continue the application on breast cancer analysis
summary(wbcd)
# get X by excluding the diagnosis (removing column 1)
wbcd.X = subset(wbcd, select = -1)
summary(wbcd.X)
dim(wbcd.X)
#normalize X so all attributes contribute equally in calculating distance
wbcd_X.normalized = as.data.frame(lapply(wbcd.X, normalize))
summary(wbcd_X.normalized)

#STEP3: split data into traning and test sets
#training the model on the data
training.X = wbcd_X.normalized[1:350, ]
test.X = wbcd_X.normalized[351:569, ]
# here first 350 data points are used for trainig, the others are used for test/validation
train.Y=wbcd[1:350, 1]
test.Y=wbcd[351:569, 1] 
# important to install the package that has knn function
install.packages("class")
library("class")
help("knn")
dim(wbcd)

# knn use mojority votes for classification, and breaks a tie at random
# to avoid randomness and make the results re-produceable, set a seed before training 
set.seed(1)

#Hint 1 and 2
# STEP 4: Train models for K = 1..21 and store errors

# create vector to store test errors
err <- rep(0, 26)

for (k in 1:26) {
  
  # run KNN with current k
  wbcd.pred <- knn(training.X, test.X, train.Y, k = k)
  
  # calculate test error
  err[k] <- mean(wbcd.pred != test.Y)
}
#Step 3 done by Matthew Bussell
K <- 1:26
plot(1/K, err,
     type = "b",
     xlab = "1/K (Model Flexibility)",
     ylab = "Test Error",
     main = "Test Error vs 1/K")

#this plot was done by Joshua A
plot(K, err,
     type = "b",
     pch = 19,
     col = "blue",
     xlab = "Number of Neighbors (K)",
     ylab = "Test Error",
     main = "Test Error vs K")

# show all errors
print(err)

# find best K (smallest test error)
best.k <- which.min(err)



#STEP 4: evaluating performance
#depending on the data, you may need to use various metrics
table(wbcd.pred, test.Y)
err = mean(wbcd.pred != test.Y)
err


#STEP 5:
#Review the code above, revise the code to improve performance 
# try different K values and chose the best K value
#Group Assignment 4 Step 2: Antonio Mora
cv.k <- 5              # number of folds
n <- nrow(wbcd_X.normalized)
fold.size <- floor(n / cv.k)

cv.err <- rep(0, 26)   # store averaged CV errors

for (k in 1:26) {
  
  fold.errors <- rep(0, cv.k)
  
  for (i in 1:cv.k) {
    
    # define test indices for fold i
    start <- (i-1)*fold.size + 1
    end <- i*fold.size
    
    test.index <- start:end
    
    # split training and testing sets
    cv.test.X <- wbcd_X.normalized[test.index, ]
    cv.train.X <- wbcd_X.normalized[-test.index, ]
    
    cv.test.Y <- wbcd[test.index, 1]
    cv.train.Y <- wbcd[-test.index, 1]
    
    # run KNN
    cv.pred <- knn(cv.train.X, cv.test.X, cv.train.Y, k = k)
    
    # compute fold error
    fold.errors[i] <- mean(cv.pred != cv.test.Y)
  }
  
  # average error across folds
  cv.err[k] <- mean(fold.errors)
}

# show cross validation errors
print(cv.err)

# find best K based on CV
best.k.cv <- which.min(cv.err)

# plot cross-validation error vs K
plot(1:26, cv.err,
     type = "b",
     pch = 19,
     col = "red",
     xlab = "Number of Neighbors (K)",
     ylab = "Average CV Test Error",
     main = "5-Fold Cross-Validation Error vs K")
