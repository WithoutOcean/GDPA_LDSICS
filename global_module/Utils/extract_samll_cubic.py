import numpy as np
import LDA_SLIC

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignment_index(assign_0, assign_1, col):
    new_index = assign_0 * col + assign_1
    return new_index


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    # print( selected_rows.shape)
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    # print('selected_patch',selected_patch.shape)
    # LS = LDA_SLIC.LDA_SLIC(selected_patch)
    # Q, S, A, Seg = LS.simple_superpixel(scale=20)
    # print('q',Q.shape)
    # print('s', S.shape)
    # print('a', A.shape)
    # print('s', Seg.shape)
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    #(10249,9x9x200)
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    # print('small_cubic_data',small_cubic_data.shape)
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    # print(len(data_assign))
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data