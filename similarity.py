# -*- coding: utf-8 -*-
# author: Willa Cheng

# this file includes functions caclulating similarities and distance 
# based on chapter 2.4 of book Data Mining by Han, Kamber and Pei
import numpy as np

def num_matches(x, y, d_type):
    """
    Return number of matches between vector inputs. 
    vector inputs must have same length. 
    params
        x : one of the input, nominal or binary attribute
        y : the other input, nominal or binary attribute
        d_type : currently accepting 'nominal' or 'binary'
    """
    # initiate final value 
    n_matches = 0
    t, s, r, q = 0
    # check which d_type data belongs to (only allow nominal and binary)
    d_type_list = ['nominal', 'binary']
    if d_type not in d_type_list: 
        raise ValueError("d_type must be in", d_type_list)
    else: 
        pass    
    # check if x and y had same length 
    if len(x) != len(y):
        raise ValueError("Arrays must have the same size")
    else:
        pass 
    # iterate through both vectors and count number of matches 
    if d_type == 'nominal': 
        for i, j in zip(x, y):
            if i == j:
                n_matches += 1
            else: 
                continue 
    # binary - return list of (t, s, r, q) for n_matches 
    elif d_type == 'binary':
        for i, j in zip(x, y):
            if i == j: 
                if i == 0: 
                    t += 1
                elif i == 1: 
                    q +=1 
            if i != j: 
                if i == 0: 
                    s += 1
                elif i == 1: 
                    r += 1
        n_matches = [t, s, r, q]
    return n_matches

def dissimilarity(x, y, d_type, method=None):
    """
    Return dissimilarity for nominal attributes and binary attributes. 
    params
        x : one of the input, nominal or binary attribute
        y : the other input, nominal or binary attribute
        d_type : currently accepting 'nominal' or 'binary'
        method : binary use only, currently accepting 'symmetric' or 'asymmetric'
    """
    # check if method in the lists
    method_lst = ['symmetric', 'asymmetric']
    if method not in method_lst: 
        raise ValueError("Methods must be in", method_lst)
    else: 
        pass 
    # nominal 
    if d_type == 'nominal': 
        dis = 1 - num_matches(x, y, d_type='nominal') / len(x)
    # binary 
    elif d_type == 'binary': 
        if method == None: 
            raise ValueError("Methods cannot be None for binary d_type.")
        else: 
            pass 
        t, s, r, q = num_matches(x, y, d_type='binary')
        if method == 'symmetric': 
            dis = (r + s) / (q + r + s + t)
        elif method == 'asymmetric':
            dis = (r + s) / (q + r + s)
    return dis 

def minkowski(x, y, h): 
    """
    Return minkowski distance. 
    vector inputs must have same length. 
    """
    # check if x and y had same length 
    if len(x) != len(y):
        raise ValueError("Arrays must have the same size")
    else:
        pass 
    # check if h >= 1
    if h < 1: 
        raise ValueError("h must have value greater than or equal to 1.")
    # initiate value
    d = 0
    # start iteration
    for i, j in zip(x, y): 
        d += (abs(i - j))**h
    # root d 
    d = d**(1/float(h))
    return d

def distance(x, y, method='euclidean', h=None):
    """
    """
    # check if method in in list 
    m_list = ['euclidean', 'manhattan', 'minkowski']
    h_list = ['euclidean', 'manhattan']
    if method not in m_list: 
        raise ValueError("Methods are limited to ", m_list)
    else: 
        pass 
    # h is required when methods are not euclidean or manhattan 
    if method not in h_list: 
        if h == None: 
            raise ValueError("Value of h cannot be none when methods are not in", h_list)
        else: 
            pass 
    else: 
        pass
    # initiate return value 
    d = 0
    # return value based on methods 
    if method == 'euclidean': 
        d = minkowski(x, y, h=2)
    elif method == 'manhattan': 
        d = minkowski(x, y, h=1)
    elif method == 'minkowski': 
        d = minkowski(x, y, h)
    return d

def cos_sim(x, y): 
    """
    Return cosine similarity between vector x and y 
    """
    sim = np.dot(x, y) / (distance(x, x, method='euclidean') * distance(y, y, method='euclidean'))
    return sim