# social_fabric
Extracts data and predicts personality traits from phone logs

This is intended for data from the Social Fabric research project. preprocess.py extracts a wide range of quantities from folders containing the log files (call/text/gps/bluetooth/facebook) of the users and match those with the psychological profiles of each user (not included in the repo for privacy reasons, of course).
psych_shared.py contains a range of methods and classes to predict profiles from the extracted data.
