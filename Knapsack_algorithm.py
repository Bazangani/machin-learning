import numpy as np
import pandas as pd


def knapsack(input_Knap, W):

    DATABASE = pd.DataFrame()
    RESULT_Knap = pd.DataFrame()
    Node_number = 1
    iteration = 0

    while len(input_Knap != 0):
        iteration = iteration+1

        # create Value array for knapsack algorithm
        VALUE = []
        l = len(input_Knap)
        for exponent in range(1, l + 1):
            VALUE.append(2**exponent)

        input_Knap['VALUE'] = VALUE
        input_Knap = input_Knap.sort_values(by=['Execution_time'], ascending=True)

        wt = list(np.int_(input_Knap['Execution_time']))
        val = input_Knap['VALUE'].tolist()
        n = len(val)
        df = pd.DataFrame()

        K = [[0 for w in range(W + 1)] for i in range(n + 1)]

        # Build table K[][] in bottom
        # up manner
        for i in range(n + 1):
            for w in range(W + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif wt[i - 1] <= w:
                    K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
                else:
                    K[i][w] = K[i - 1][w]

                    # stores the result of Knapsack
        res = K[n][W]

        w = W
        for i in range(n, 0, -1):
            if res <= 0:
                break
            # either the result comes from the
            # top (K[i-1][w]) or from (val[i-1]
            # + K[i-1] [w-wt[i-1]]) as in Knapsack
            # table. If it comes from the latter
            # one/ it means the item is included.
            if res == K[i - 1][w]:
                continue
            else:

                # This item is included.
                choice = input_Knap.iloc[[i - 1]]
                df = df.append(choice)
                df["group_ID"] = Node_number

                # Since this weight is included
                # its value is deducted
                res = res - val[i - 1]
                w = w - wt[i - 1]

        DATABASE = DATABASE.append(df)  # dope repetitive value
        Node_number = Node_number + 1
        input_Knap = input_Knap[~input_Knap.isin(DATABASE)].dropna()

    RESULT_Knap = RESULT_Knap.append(DATABASE)
    return (RESULT_Knap)









