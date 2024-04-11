import numpy as np

def find_best_contingency(best_c_matrix):

    # declare return dict
    mapping_dict = {
        "new_order": [],
        "mapping": {}
    }

    # declare tracker, this will used to track the mapping
    tracker = 0

    # interate over the rows of columns
    for row in best_c_matrix:

        # find the index of max value in each row
        max_row_index = np.argmax(row)

        # get the corresponding value; this will be used check if there is a better position
        row_value = row[max_row_index]

        # find the column that corresponds to the index of the max value in the row
        col = best_c_matrix[:, max_row_index]

        # find the index of the max value in the column of corresponding to the row index
        max_col_index = np.argmax(col)

        # get the corresponding value
        col_value = col[max_col_index]

        # check which value is greater
        # this effectively checks down the column to see if there is a better match
        if row_value >= col_value:
            mapping_dict['new_order'].append(max_row_index)
            mapping_dict["mapping"][max_row_index] = tracker
            print("mapping:", tracker, max_row_index)
        
        # if there is better match, take the second highest value index
        else:
            sorted_indices = np.argsort(row)
            second_largest_index = sorted_indices[-2]
            mapping_dict['new_order'].append(second_largest_index)
            mapping_dict["mapping"][second_largest_index] = tracker
            print("mapping:", tracker, second_largest_index)

        # update tracker
        tracker += 1

    # pull out the new order of the columns
    new_order = mapping_dict["new_order"]

    # create a list of the all the predicted labels
    predicted_label_cols = np.arange(best_c_matrix.shape[1]) # -1

    # create set of columns that are missing
    missing_cols = set(predicted_label_cols) - set(new_order)

    # append those columns to the new order
    #tracker = len(new_order) - 1
    for i in missing_cols:
        new_order.append(i)
        mapping_dict["mapping"][i] = tracker
        tracker += 1
        print("mapping:", tracker, i)

    # reordered the contingency matrix
    reordered_matrix = best_c_matrix[:, new_order]

    # add the return dict
    mapping_dict['best_contingency_matrix'] = reordered_matrix

    reordered_matrix

    return mapping_dict