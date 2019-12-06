# FINA-4350-Predicting-Soybean-Spot-Price-Change-with-Trade-War-Related-Tweets
Term Project Instructor: Dr. Matthias Buehlmaier       Group: YayNLP
# Background

Since US-Sino trade war (“Trade War”) burst out in March 2018, the event has affected a wide range of business in the two countries, and the impact even spreads worldwide. Soybean has been among the commodities with the largest import amount to China from the U.S., which inevitably bore the brunt of the Trade War. 

In early April 2018, right after the U.S. published its list of tariffs on Chinese products, Ministry of Commerce of China struck back by imposing tariffs on 128 products it imports from the U.S., including soybeans with an additional 25% tariff. The high tariff triggered U.S. soybean exports to China. From U.S. Census Bureau, we see that U.S. soybean exports, as well as its percentage bound for China, showed an abnormal shrink after the event. It provides evidence that China plays an important role in U.S. soybean exports. The Trade War and increasing tariff can have a significant impact on the price of soybean, which not only factors in the market value of the product but also reflects investors’ view on the trade war’s direction.


As the Trade War rendered a poor sale of U.S. soybean, the commodity’s price shows a downturn. As a direct indicator of soybean price, Soybean Futures price at the Chicago Board of Trade (CBOT) showed a significant decrease after the Trade War and dropped to a ten-year historical low on May 13, 2019.



# Objectives

Based on the above observations, we think that there can be a potential investment opportunity on Soybean Futures or ETF, if we see any predictive power of Twitter opinions on the spot price of soybeans during the on-going Trade War. Commodity also serves as a better subject of study compared to stock in the sense that people tweet less about their P/L with commodities. Therefore, we have less noises from people’s posts after price changes.

We would like to apply the knowledge from this course to analyze the relationship between Trade War and the historic price of the U.S. Soybean commodity. We are going to capture people’s opinions on the projected direction of Trade War negotiation from Twitter posts, and fit in statistical models to see if the moving trend of Soybean Futures Price can be predicted by the market opinion. We try to explore the underlying events that link the trade war with the Soybean price. Investment analysts can use these key words we found  as the areas of fundamental research in the future when they study Soybean Commodities.


# Data

We used the tweets information provided by Dr 高 and filtered them with "trade war" related keywords to represent the general market opinion on "trade war". 

There are mainly two reasons we choose to use Dr 高's data instead of collecting data by ourselves. Firstly, Twitter API does not allow us to listen to historical data with keywords. The trade war started as early as the beginning of 2018, which is way before the starting of our course. If we use tweets we collected from early November 2019 we will lose the most of major events. One alternative method is to trace back tweets from a list of selected users. However, Twitter users, except for Donald Trump, normally post less than 10 "trade war" related tweets throughout the year. Thus, we need to create a list of more than thousands of active user on this topic to collect a sufficient amount of data, which is both unrealistic and biased for the selection process. Secondly, most of our group members stayed in mainland China after the class suspension. We met many technical problems with VPN, protocols etc. One of the most significant drawbacks for building up our data by filtering the existing tweets data provided by Dr 高 is that the data was pre-filtered during the listening process. We find this pre-filtering issue is acceptable given the following reasons. The keywords used by Dr 高 collected the market-related information that goes along with our research topic. We might want to keep only market-related trade war opinion to reduce the noise in data as well even if we collected the data by ourselves. Besides, though it is not a random sampling of the data pool, the pre-filtering process is relatively neutral to our topic. 

Then we filtered the data with "trade war", "tariff" and exclude tweets including "mexic" to remove tweets concerning the US-Mexico trade dispute. We carefully selected the keywords for filtering, knowing that it could introduce bias/guidance to the data. For each keyword added, we exported the data and manually checked the additionally introduced tweets' relevancy to identify if the keyword appropriate or not. We found that "trade war" itself covered the topic quite well, which give us 24000+ tweets. We also noticed that Mexico related data should be excluded during the data inspection process. Adding “trade dispute into” the keywords list can only increase the data size by a few hundred, which is inconsequential compared to the total amount. We do not want to risk being selective during the filtering process for adding a few hundred data points. So we only included "trade war" and "tariff" (“tariff”increased the data size for ⅓ even after excluding Mexico related tweets) at the end. 

Financial data & No. post price analysis 
We downloaded the historic spot price of Soybean traded at CBOT during our period of study from Capital IQ. Since CBOT does not trade on weekends, we filled in the weekend prices with the price on the next Monday. For each day, we calculate the price change from the day before (lag_price change) for further processing.

# Method & Result
Please refer to jupyter notebook for details (please open html version for best quality)

We built a set of supervised machine learning model with tweets text data as explanatory variable and the price change one day after as the response variable.

1. Import raw tweet data and merge it with lagged spot price data
We import the raw tweets data provided by Dr 高 and merged each tweet with the lagged change of soybean spot price. (i.e. We use the tweets written on Date(t) to predict the soybean spot price change on Date(t+1).) The reason we use lagged the price change is that we want to avoid the look back effect and explore the predicting power of twitter.

2. Filter trade war related tweets
We filtered the tweets contain “trade war” or “tariff” but not include “mexic”. The filtered  trade war related tweets have a size of 38,050 from 2018 Feb 04 to 2019 Nov 30.

3. Text Preprocessing
We preprocessed the text by removing Twitter listening keywords, non-English words, numbers, website links. At the beginning, we also removed punctuation manually before, but then we realised that sklearn package will take care of it. Removing punctuation in advance will cause difficulties for the tokenization package to recognize patterns. Besides, we also performed stemming and lemmatization with NLTK package. However, they have limited effect in grouping all forms of a single word as we still found some similar words appeared in the final most important list. Some of them are caused by English and American spelling (eg. Adviser and advisor) while the others caused by the user’s careless spelling (eg. tariffs, tarrif, tarriff) that cannot be easily handled by the package.

4. Split the data set
We splitted the training/testing data on 2019 Nov 01. We use all the data from beginning to 2019 Oct 31 as the training data and the data from 2019 Nov 01 to 2019 Nov 30 as the testing data. (Indeed it is the validation data set since we are using the testing accuracy to decide hyper parameter “number of terms”)

5. Convert text to Numbers
We tried converting text data to numbers by both the Bag of Words technique and tf-idf method. From the final result, the performance of these two converting method has no significant difference. (The works are conducted in 2 separate jupyter notebook for two methods.) Though we also fitted models on Bag of Words version of work, our further exploration and analysis are built only on the tf-idf method. When we fit the tf-idf sparse matrix, we decided to drop words that appears in more than 90% of the documents and those words appeared less than 20 times (roughly 0.05% of all filtered data). Thus, our initial tf-idf matrix has 1631 words in total.

6. Fit a black box model & obtain the list of variable importance
A multiple layer neural network model(MLP-NN) was fitted as our initial exploration and obtain the list of words sorted by its importance. The accuracy of the MLP-NN full model (with all 1631 words as its variables) is 0.8508 on the training data and only 0.3368 on the testing data. It works will with training data while performing poorly with testing data, indicating the existence of overfitting. Demonstrating the need to improve data’s signal to noise ratio and further variables reduction. Thus, we run a variable importance evaluation to rank the words with their relative importance. Variable importance is a commonly used technique to inspect black box models. It ranked the importance of each word by calculating feature value permutation. The key idea is that if a small change in a variable will cause a big change in the result, that variable is important. The left of the following is the table of most important 19 variables. The shades of green in cells color and the weighting scores indicate the importance of corresponding features. Thus, we constructed 2 more condensed tf-idf matrix with only the top 100 words and the top 19 most important words.

7. Fit Multiple layer Neural Network Model(MLP-NN), Naive Bayes Model(NB), Logistic Model with all (1631) terms(right) , the top 100 terms(middle) and the top 19 most important terms(left) for hyper parameter “number of words considered” specification. Multiple layer Neural Network Model(MLP-NN) performed best with the top 19 most important terms(left), which is reasonable considering the complicated model is more vulnerable to noises in the data. From our exploration, we found that the signal to noise ratio in our data is still quite high after a series of noise removal process. So we believe the MLP-NN is not an ideal solution. However, the MLP-NN model provided us with a set of partial dependence plots that facilitated our interpretation in the relationship between terms and price-changes. Naive Bayes Model(NB) reaches its highest accuracy on testing data with 100 most important variables. Logistic model (right): Logistic model performed quite well on all three set of input variables and also reaches the highest predicting accuracy on testing data with the top 19 most important terms(left).



8. Model selection

Finally we selected the logistic model with the top 19 most important terms. It is a simple and highly explainable model with a surprisingly satisfying accuracy. The above results shown that, given the nature of text analytics, project should carefully choose the machine learning model. Choosing a simpler model (i.e. Logistic model) could be a better idea than the more complicated models considering the problem in signal-to-noise ratio. 

# Conclusion
1.Model interpretation & further exploration 
As we interpret our models, we are happy to see that they are consistent in explaining the direction of price movement given a key word. For example, the Partial Dependence Plot (PDP) from our MLP-NN model and the coefficients from our Logistics Regression all indi cate that 'trump', 'tarrifs’(Yes, people make typos when they tweet.) and ‘speech’ are linked with a drop in the spot price of Soybean. So, the market potentially link trump with downside risks in the Soybean’s price.

Meanwhile, our model sorts out some interesting keywords such as ‘friday’, ‘philstockworld’ and ‘facebook’. To understand the context of these words, we go through the filtered Twitter posts to see the sample contents. It turns out that ‘friday’ shows up because Trump has announced around five breaking news on Fridays, which linked it with market movements. Philstockword is the account name of a key opinion leader in the Twitter investment community, and his posts are frequently retweeted. The key words we found can further explored in fundamental research to understand the Soybean Commodity market.

So far, our models are based on Tweet contents and analyse the importance of words. To aggregate the daily sentiment of the community, Rao & Srivastava (2012) have mentioned a method to calculate ln [(1+Mp)/(1+Mn)], where Mp and Mn are the number of positive and negative posts respectively. We didn’t develop the other sets of model based on the daily aggregate sentiment, as we think this is not the focus of the course.

2.Possible speculation 
The models we develop can supplement fundamental analysis, and help investors form a view on short-term price movement and design their position in Soybean Commodity Futures and ETF.


# Limitation:

The following paragraph discusses the limitations of this project.


First, the dependent variable in the models, which is the change in the price of soybean, was converted into a categorical variable that classified the price change into 3 quantiles indicated by “0,1,2” for modelling. Some information was lost during the transformation, and the predicting power of our results also decreased.

Second, we confronted unforeseeable technical problems executing this project. Since our group members are in Mainland China after the suspension of the semester, we need to use VPN to access Twitter. However, HKUVPN blocks Twitter occasionally. We tried with other VPNs, but they do not support the HTTP agreement required by Tweepy. We cannot download data directly from Twitter. The backup plan is to use the dataset provided by Dr Buehlmaier, though the information reflected by the dataset about US-Sino trade war might not be all-sided due to the initial filtering. 

Third, reTweets might inflate the predicting results of our models. We did not rule out reTweets because we want to assign higher weights to widely-recognized opinions. However, some highly reTweeted post might have overrated weights, which might inflate the predictions. We found it is difficult to find a good balance between acknowledging popular opinions and avoid them to dominant the result.

Lastly, the spot price of soybean includes both trade war information and the unique characteristics of the agriculture industry, and may not be accurately predicted with only trade war-related opinions. We may combine our NLP and fundamental research to achieve better prediction power.


# Acknowledgement

We would like to give special acknowledgement to Dr 高 for providing the tweet data collected. Without your help, we would not be able to make this project.





Reference

Rao, T., & Srivastava, S. (2012, August). Analyzing stock market movements using twitter sentiment analysis. In Proceedings of the 2012 international conference on advances in social networks analysis and mining (ASONAM 2012) (pp. 119-123). IEEE Computer Society.



