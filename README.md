All code contained in source_code.py

Requirements:
	sklearn
	category_encoders
	catboost
	numpy
	pandas

To run:

	python source_code.py

On run:
	
	Processing:
		Data from tcd ml 2019-20 income prediction test (without labels).csv and tcd ml 2019-20 income prediction training (with labels).csv is processed using category_encoder's TargetEncoder() and sklearn's MinMaxScaler()
		Rows with NaN values are dropped from training data
		NaN values in test data are replaced with median of that values column

	Local RMSE is created with training data 0.75/0.25 split
	User is asked if they would like to create a full submission based off this RMSE
	If Y or y:
	    Model trained based off full training data using CatBoostRegressor and predictions are written to submission file
	else:
	    Program ends
	    
