import pandas as pd

def accuracy():
    return None

def precision(category_list, df_actual, df_predict):
    """
    When I predict X, how often is it actual X?

    = True_Positive(x) / (True_Positive(x) + False_Positive(x))

    Returns ditionary with all categories + average precision

    """

    # 2 dictionaries
    t_pos = {x: 0 for x in category_list}
    f_pos = {x: 0 for x in category_list}

    for actual, prediction in zip(df_actual, df_predict):
        if prediction == actual:
            t_pos[prediction] += 1
        else:
            f_pos[prediction] += 1
    
    cat_predictions = {}
    total = 0

    for c in category_list:
        cat_predictions[c] = (t_pos[c] / (t_pos[c] + f_pos[c]))
        total += cat_predictions[c]
    
    cat_predictions["AVERAGE"] = total / len(category_list)

    return cat_predictions

