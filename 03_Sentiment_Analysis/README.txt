1) training_model.py

Loading a new dataset with more than 10,000 short reviews (it will need the short_reviews directory with its files)
Using the voted classifier to get the results
Saving the objects with pickles into a "pickled" directory

NOTE: feature_sets.pickle is around 300 MB size so cannot be uploaded to GitHub
If you want to use this, you will need to run the script on your side and let some time to save the pickled objects

2) sentiment_analysis.py

Once the training is done (training_model.py), we can use this file as a reference for try_me.py
This means the script sentiment_analysis.py is not manually executed
Loading the pickled objects once called from try_me.py
Method "sentiment" inputting the provided text and outputting the classification and the confidence

3) try_me.py

Using sentiment_analysis.py as a reference to load all the objects and tools
We can try it out by entering a comment and calling the "sentiment" method