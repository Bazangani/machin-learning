import numpy as np
import pandas as pd
from ANN.ELK import elk_connect
from sklearn.neighbors import LocalOutlierFactor

# function Clean_database :
# input : month ( a number between 1 to 12 ) retrieve data frame this month
#  output : a database consist of 1 month samples of data 


def clean_dataframe(today):
      clean_df = pd.DataFrame()

      #init
      test_name_df = pd.read_csv('C:/Dev/df.csv')
      train_df = elk_connect(today)
      train_df['ID'] = train_df['...'].astype('category').cat.codes
      train_df[['total', 'skipped']] = train_df[['total', 'skipped']].astype(int)

    # create test_name_df
      test_name_df = (test_name_df.append(train_df[['ID', 'class']]))
      test_name_df.drop_duplicates(subset=['class'], keep='first', inplace=True)
      test_name_df.reset_index(drop=True, inplace=True)
      test_name_df.to_csv('C:/Dev/df.csv', index=False)

    # outlier detection
      number_of_test = int(max(train_df['ID']))
      for i in range(0, number_of_test):

           one_test = train_df[train_df['ID'] == i]
           train_df = train_df[~train_df.isin(one_test)].dropna()

           one_test['@timestamp2'] = one_test['@timestamp'].astype('category').cat.codes
           X = np.array(one_test[['time_ela', '@timestamp2']])

           if len(X) > 1:
               clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
               y_pred = clf.fit_predict(X)
               one_test = one_test.set_index(y_pred)
               one_test = one_test[one_test.index == 1]
               clean_df = clean_df.append(one_test)
               clean_df.reset_index(drop=True)

#########################################################################################

      new_database = pd.DataFrame()

      def split_sequence(one_test, n_steps):

         sequence = list(one_test['time_ela'])

         X, y = list(), list()

         start_ix = 0
         prob_skipped = []
         test_total_sum=[]

         for i in range(len(sequence)):

            # find the end of this pattern
             end_ix = start_ix + n_steps
            # check if we are beyond the sequence
             if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
             seq_y, seq_x = sequence[start_ix], sequence[start_ix + 1:end_ix + 1]
            # compute the probability of error
             sum_err=sum(one_test['skipped'])
             p_skipped = sum_err / len(one_test)
             prob_skipped.append(p_skipped)

             sum_total=sum(one_test['total'])
             test_total_sum.append(sum_total)

             start_ix = end_ix + 1

             X.append(seq_x)
             y.append(seq_y)

         return X, y, prob_skipped, test_total_sum

#################################################################################
    # create time series for each data and compute probability of error for time series

      for i in range(0, number_of_test):
          i = i + 1  # number of tests

          one_test = clean_df[clean_df['ID'] == i]
          clean_df = clean_df.reset_index(drop=True)
          one_test = one_test.reset_index(drop=True)
          clean_df = clean_df[~clean_df.isin(one_test)].dropna()

        # sort the time series from last one to first one  (ascending should be True)
          one_test = one_test.sort_values(by=['@timestamp'], ascending=True)
          n_step = 10  # length of time series

          if len(one_test) > 10:
            # test_name = one_test.iloc[0, 1]
              test_name = one_test['test_class'][0]

            # compute time series when we have more then 10 samples
              X, y, prob_skipped, test_total_sum = split_sequence(one_test, n_step)

              time = pd.DataFrame(X)  # 10 sample
              next_time = pd.DataFrame(y)  
              prob_sk = pd.DataFrame(prob_skipped)  # probability of error
              tests_total_sum=pd.DataFrame(test_total_sum)

              one_test_data = pd.concat([time, next_time, prob_sk, tests_total_sum], axis=1)  
              one_test_data['name'] = test_name  # assign test class to the result

              test_ID= test_name_df[test_name_df['test_class'] == test_name]

              ID= test_ID['ID']

              one_test_data['ID'] = int(ID)  

              new_database = new_database.append(one_test_data)  # create a database for all the tests

      new_database.to_csv('result.csv')

      return new_database


