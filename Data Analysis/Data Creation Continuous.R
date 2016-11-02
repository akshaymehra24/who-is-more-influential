setwd("C:\\Users\\aksha\\OneDrive\\Documents\\GitHub\\who-is-more-influential\\Data Analysis")
data=read.csv("train.csv")
head(data)
names(data)
newData=data.frame(data$Choice
                   ,data$A_follower_count-data$B_follower_count
                   ,data$A_following_count-data$B_following_count
                   ,data$A_listed_count-data$B_listed_count
                   ,data$A_mentions_received-data$B_mentions_received
                   ,data$A_retweets_received-data$B_retweets_received
                   ,data$A_mentions_sent-data$B_mentions_sent
                   ,data$A_retweets_sent-data$B_retweets_sent
                   ,data$A_posts-data$B_posts
                   ,data$A_network_feature_1-data$B_network_feature_1
                   ,data$A_network_feature_2-data$B_network_feature_2
                   ,data$A_network_feature_3-data$B_network_feature_3)
head(newData)
write.csv(newData, "train_new.csv")

data=read.csv("test.csv")
head(data)
names(data)
newData=data.frame(data$A_follower_count-data$B_follower_count
                   ,data$A_following_count-data$B_following_count
                   ,data$A_listed_count-data$B_listed_count
                   ,data$A_mentions_received-data$B_mentions_received
                   ,data$A_retweets_received-data$B_retweets_received
                   ,data$A_mentions_sent-data$B_mentions_sent
                   ,data$A_retweets_sent-data$B_retweets_sent
                   ,data$A_posts-data$B_posts
                   ,data$A_network_feature_1-data$B_network_feature_1
                   ,data$A_network_feature_2-data$B_network_feature_2
                   ,data$A_network_feature_3-data$B_network_feature_3)
head(newData)
write.csv(newData, "test_new.csv")
