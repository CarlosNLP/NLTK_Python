1) The file "sentiment_analysis" file is the one used as reference where the model will be loaded
This means that we need the "pickled" directory with the corresponding objects to make this work. Under "03_Sentiment_Analysis", run the file "training_model.py" so that this directory is created.

2) Once you have the pickled objects, run the file "twitter_sentiment_analysis.py". This file will load "sentiment_analysis.py" to make everything work automatically. After loading the model, the live tweets will start appearing along with the positive or negative classification and the confidence. In the last line of this file, you can modify the tweets you want to filter by according to the text entered.

3) Once the tweets start appearing, a file named "twitter_log.txt" will be created in the same directory. This file will basically contain the positive or negative classification for each tweet.

4) Run the file "twitter_graph.py" to continuously check the file "twitter_log.txt" and visualize a graph where we will see if something tends to be positive or negative.