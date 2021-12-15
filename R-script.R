rm(list = ls())    #delete objects
cat("\014")

options(scipen=999)

library(tidyverse)

library(factoextra)
library(Rtsne)
library(ggplot2)
library(ggthemes)
library(gridExtra)

library(glmnet)
library(ranger)
library(caret)

set.seed(123)

# load the data into R
data <- read_csv("https://raw.githubusercontent.com/tmvien/human-activites/master/train.csv")
# checking Null value
sum(is.na(data))

# change order of columns so that target variable will be the first column
data <- data[,c(ncol(data),1:561)]

# Number of Training Examples in Each Class
data %>%
  ggplot(aes(x=Activity, fill=Activity)) +
  geom_bar(stat = "count") +
  scale_fill_viridis_d() +
  theme(
    axis.text.x=element_text(angle = 15)
    , panel.background = element_rect(fill="#E8EDFB")
    , legend.position='none'
  )

# Distribution of FFT Entropy Magnitude (L2 Norm) of 3-Dimensional Signals
data %>%
ggplot(aes(x=`fBodyBodyGyroMag-entropy()`, fill=Activity)) +
  geom_density(alpha=0.4) +
  scale_fill_viridis_d() + 
  labs(x = "Feature 536-fBodyBodyGyroMag-entropy()") +
  theme(
    panel.background = element_rect(fill="#E8EDFB")
  )

# Boxplot of FFT Mean Magnitude (L2 Norm) of 3-Dimensional Signals 
data %>%
  ggplot(aes(x = Activity, y=`fBodyAccMag-mean()`, fill=Activity)) +
  geom_boxplot(alpha = 0.4) +
  labs(y = "Feature 530-fBodyAccMag-mean()") +
  theme(
    axis.text.x=element_text(angle = 10)
    , panel.background = element_rect(fill="#E8EDFB")
    , legend.position='none'
  )

# Distribution of Angle between X and Gravity Mean 
data %>%
  ggplot(aes(x=`angle(X,gravityMean)`, fill=Activity)) +
  geom_density(alpha=0.6) +
  scale_fill_viridis_d() + 
  labs(x = "Feature 560-angle(X,gravityMean)") +
  theme(
    panel.background = element_rect(fill="#E8EDFB")
  )

# A pca object
d.pca <- prcomp(data[,-1])
# scree plot
g1 <- fviz_eig(d.pca)
# 2PCs visualization
g2 <- fviz_pca_ind(d.pca, geom.ind = "point", pointshape = 21, 
                   pointsize = 2, 
                   fill.ind = data$Activity, 
                   col.ind = "black", 
                   palette = "jco", 
                   addEllipses = TRUE,
                   label = "var",
                   col.var = "black",
                   repel = TRUE,
                   legend.title = "Diagnosis") +
  theme(plot.title = element_text(hjust = 0.5))
grid.arrange(g1, g2, ncol=2)

# t-SNE transformation
tSNE.fit <- data[,-1] %>%
  Rtsne()
(tSNE.df <- tSNE.fit$Y %>% 
    as.data.frame() %>%
    rename(tSNE1="V1",
           tSNE2="V2") %>%
    mutate(Activity=data$Activity)) %>%
  ggplot(aes(x = tSNE1, 
             y = tSNE2)) +
  geom_point(aes(fill = Activity, alpha = 0.4), colour = "black", shape = 21, size = 3) + 
  scale_color_viridis_d() + 
  theme(
    legend.position="bottom"
    , panel.background = element_rect(fill="#E8EDFB")
  )

# encode response variable with number
y <- ifelse(data$Activity == "WALKING", 1,
            ifelse(data$Activity == "WALKING_UPSTAIRS", 2,
                   ifelse(data$Activity == "WALKING_DOWNSTAIRS", 3, 
                          ifelse(data$Activity == "SITTING", 4, 
                                 ifelse(data$Activity == "STANDING", 5, 6)
                          )
                   )
            )
)
y <- factor(y, labels = c("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"))
# checking if there is any mismatch
sum(which(data$Activity == "WALKING") != which(y==1))
sum(which(data$Activity == "WALKING_UPSTAIRS") != which(y==2))
sum(which(data$Activity == "WALKING_DOWNSTAIRS") != which(y==3))
sum(which(data$Activity == "SITTING") != which(y==4))
sum(which(data$Activity == "STANDING") != which(y==5))
sum(which(data$Activity == "LAYING") != which(y==6))

# create matrix for train data
X <- as.matrix(data[,-1])
n <- dim(X)[1]
p <- dim(X)[2]
# rm the original data
rm(data)
# using 80% of data for train and 20% for validation
train <- sample(n, n*0.8)
# number of K fold cross-validation
K <- 10
# a matrix for storing misclassification rate
m.score <- matrix(0, ncol = 4, nrow = 2)
colnames(m.score) <- c("MRidge", "MLasso", "MElastic", "KNN")
rownames(m.score) <- c("Model", "Time")

######### Ridge
# define alpha
a = 0
# record time
start.time      <-    proc.time()
cv.rd <- cv.glmnet(X[train,], y[train], family = "multinomial", nfolds=K, type.measure = "class", alpha=a)
rd.fit <- glmnet(X[train,], y[train], lambda = cv.rd$lambda.min,  family = "multinomial", alpha = a)
end.time        <-    proc.time() - start.time
m.score[2,1]    <-    end.time['elapsed']
y.hat.rd        <-    predict(rd.fit, newx = X[-train,], type = "class")
m.score[1,1]    <-    mean(y[-train] == y.hat.rd)
######### Lasso
# define alpha
a = 1
# record time
start.time      <-    proc.time()
cv.ls   <- cv.glmnet(X[train,], y[train], family = "multinomial", nfolds=K, type.measure = "class", alpha=a)
ls.fit  <- glmnet(X[train,], y[train], lambda = cv.ls$lambda.min,  family = "multinomial", alpha = a)
end.time        <-    proc.time() - start.time
m.score[2,2]    <-    end.time['elapsed']
y.hat.ls        <-    predict(ls.fit, newx = X[-train,], type = "class")
m.score[1,2]    <-    mean(y[-train] == y.hat.ls)
########## ElasticNet
# define alpha
a = 0.5
# record time
start.time      <-    proc.time()
cv.el <- cv.glmnet(X[train,], y[train], family = "multinomial", nfolds=K, type.measure = "class", alpha=a)
el.fit <- glmnet(X[train,], y[train], lambda = cv.el$lambda.min,  family = "multinomial", alpha = a)
end.time        <-    proc.time() - start.time
m.score[2,3]    <-    end.time['elapsed']
y.hat.el        <-    predict(el.fit, newx = X[-train,], type = "class")
m.score[1,3]    <-    mean(y[-train] == y.hat.el)

########### KNN
# Define training control
train.control   <- trainControl(method = "cv", number = K)
# record time
start.time      <-    proc.time()
# Cross validate and train the model
knn.fit <- train(x=X[train,], y=y[train], method = "knn",
                 trControl = train.control,
                 metric = "Accuracy",
                 tuneGrid = expand.grid(.k=seq(3, 51, 2)))
end.time        <-    proc.time() - start.time
m.score[2,4]    <-    end.time['elapsed']
y.hat.knn       <-    predict(knn.fit,newdata = X[-train,] , type = "raw")
m.score[1,4]    <-    mean(y[-train] == y.hat.knn)

# 10 fold cv plot
knn.result <- knn.fit$results[,c("k", "Accuracy")]
par(mfrow=c(2,2))
plot(cv.rd, main="Multinomial Ridge Logistic")
plot(cv.ls, main="Multinomial Lasso Logistic")
plot(cv.el, main="Multinomial Elastic Net Logistic")
plot(knn.result$k, knn.result$Accuracy, type = "b",
     col = "blue", ylab = "Accuracy", xlab ="k",
     main = "KNN")