setwd("C:\\Users\\aksha\\OneDrive\\Documents\\GitHub\\who-is-more-influential\\Data Analysis")
data=read.csv("train_new.csv")
head(data)
names(data)
data.pca <- prcomp(data[,3:13], center = TRUE, scale. = TRUE) 
print(data.pca)
#plot method
plot(data.pca, type = "l")
summary(data.pca)
#PC = predict(trans, iris[,1:4])
names(data.pca)
class(data.pca$x)
train.data=as.data.frame(data.pca$x)
train.data=train.data[,1:8]
train.data$Choice=data$data.Choice

#transform test into PCA
test=read.csv("test_new.csv")
test.data <- predict(data.pca, newdata = test)
test.data <- as.data.frame(test.data)
test.data=test.data[,1:8]

write.csv(train.data, "pca_delta_train.csv")
write.csv(test.data, "pca_delta_test.csv")
