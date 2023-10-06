import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def get_Sn(
        view_num,
        alldata_len,
        missing_rate
):
    one_rate = 1-missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()#categories=view_num)
        view_preserve = enc.fit_transform(
                    randint(0, view_num,
                    size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005: #0.005:
        enc = OneHotEncoder()#categories=view_num)
        view_preserve = enc.fit_transform(
                                                    randint(0,
                                                      view_num,
                                                      size=(alldata_len, 1)))\
                                                      .toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (
                        randint(0, 100,
                        size=(alldata_len, view_num))
                        < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter
                     + view_preserve) > 1).astype(np.int))

        #防止inf的出现
        if (1 - a / one_num)==0:
            continue

        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)

        matrix_iter = (randint(0, 100,
                            size=(alldata_len, view_num))
                            < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter
                   + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


def save_Sn(Sn, str_name):
    np.savetxt(str_name + '.csv', Sn, delimiter=',')


def load_Sn(str_name):
    return np.loadtxt(str_name + '.csv', delimiter=',')
