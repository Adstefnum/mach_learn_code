import pandas as pd

#loading multiple data
def get_data():

	files = ["2014_Financial_Data.csv","2015_Financial_Data.csv","2016_Financial_Data.csv","2017_Financial_Data.csv","2018_Financial_Data.csv"]
	files_path = "C:\\Users\\USER\\Documents\\PYTHON PROJECTS\\DATA SCIENCE\\datasets\\fin_tech\\"
	df = pd.concat([pd.read_csv(files_path+file) for file in files ])
	return df

df = get_data()

#printing column names
for col in df.columns:
	print(col)

#drop nan values:what does it do?
df.dropna()

#fills nan values with a value but let it be either too large or too small so they
#are treated as outliers in your dataset
df.fillna()

#limits the amount of data you bring in
#then you have less data to perform train_test_split on
#but you have data to do further test with
df.shift()