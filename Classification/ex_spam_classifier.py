# Build a spam classifier (a more challenging exercise):
# • Download examples of spam and ham from Apache SpamAssassin’s public
# datasets. https://spamassassin.apache.org/old/publiccorpus/
# • Unzip the datasets and familiarize yourself with the data format.
# • Split the datasets into a training set and a test set.
# • Write a data preparation pipeline to convert each email into a feature vector.
# Your preparation pipeline should transform an email into a (sparse) vector
# indicating the presence or absence of each possible word. For example, if all
# emails only ever contain four words, “Hello,” “how,” “are,” “you,” then the email
# “Hello you Hello Hello you” would be converted into a vector [1, 0, 0, 1]
# (meaning [“Hello” is present, “how” is absent, “are” is absent, “you” is
# present]), or [3, 0, 0, 2] if you prefer to count the number of occurrences of
# each word.
# • You may want to add hyperparameters to your preparation pipeline to control
# whether or not to strip off email headers, convert each email to lowercase,
# remove punctuation, replace all URLs with “URL,” replace all numbers with
# “NUMBER,” or even perform stemming (i.e., trim off word endings; there are
# Python libraries available to do this).
# • Then try out several classifiers and see if you can build a great spam classifier,
# with both high recall and high precision.
