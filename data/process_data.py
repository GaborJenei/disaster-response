import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    """
    Reads up the messages and categories into a Pandas DataFrame

        Parameters
            messages_filepath(str): file path of messages data file
            categories_filepath(str):file path of categories data file

        Returns
            Pandas DataFrame
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on=['id'], how='left')

    return df


def clean_data(df):
    """
    Cleans the DataFrame holding the messages and categories

        Parameters
            df(DataFrame): DataFrame to clean

        Returns
            CleanedPandas DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0].values.tolist()

    # extracting a list of new column names for categories.
    category_columns = [name[:-2] for name in row]

    # rename the columns of `categories`
    categories.columns = category_columns

    # Converting the category values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')

    # adjust related column, where value is 2, re-assign them as 0
    categories.loc[categories['related'] == 2, 'related'] = 0

    # Drop Child alone column, it's empty
    categories.drop(columns=['child_alone'], inplace=True)

    # drop the original categories column from `df`
    df_out = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df_out = pd.concat([df_out, categories], axis=1)

    # drop duplicates
    df_out.drop_duplicates(inplace=True)

    return df_out


def save_data(df, database_filename):
    """
    Saves DataFrame into an SQL database

        Parameters
            df(DataFrame): DataFrame to save
            database_filename(str): filename to use for the output database

        Returns
            None (saved db file)
    """
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    """
    ETL pipeline function to run from terminal

        Returns
            database file with messages data
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepath of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
