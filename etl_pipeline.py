# import libraries
import pandas as pd
import sqlalchemy

# load messages dataset
messages = pd.read_csv('messages.csv')

# load categories dataset
categories = pd.read_csv('categories.csv')

# merge datasets
df = messages.merge(categories, on=['id'], how='left')

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand=True)

# select the first row of the categories dataframe
row = categories.loc[0].values.tolist()

# extracting a list of new column names for categories.
category_colnames = [name[:-2] for name in row]

# rename the columns of `categories`
categories.columns = category_colnames

# Converting the category values
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]

    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column], downcast='integer')

# drop the original categories column from `df`
df.drop(columns='categories', inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)

# drop duplicates
df.drop_duplicates(inplace=True)

engine = sqlalchemy.create_engine('sqlite:///DisasterResponse.db')
df.to_sql('messages', engine, index=False)
