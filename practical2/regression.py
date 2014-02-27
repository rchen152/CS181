import freq_reg as freg
import math
import numpy as np
import matplotlib.pyplot as plt
import regression_starter as rs
import util

def format_arr(feat_cols, regy_vec, fns, check = lambda _ : True):
    num_data = len(regy_vec)
    num_feats = len(feat_cols)
    lst = []
    for i in range(num_data):
        datum = [feat_cols[j][i,0] for j in range(num_feats)]
        if check(datum):
            lst.append([tuple([fns[j](datum[j]) for j in range(num_feats)]),
                        fns[num_feats](regy_vec[i])])
    return np.array(lst)

def screens(basis_fns, fns, inv_fn, outfile):
    mat,key,regy,_ = rs.extract_feats([rs.metadata_feats])
    screen_ind = key['number_of_screens']
    screens = mat.getcol(screen_ind).todense()
    train_arr = format_arr([screens], regy, fns)
    coeffs = freg.coeffs(basis_fns, train_arr)

    test,_,_,ids = rs.extract_feats([rs.metadata_feats],'testcases.xml',
                                    global_feat_dict = key)
    test_len = test.shape[0]
    preds = []
    for i in range(test_len):
        prod = freg.product((fns[0](test[i,screen_ind]),), coeffs, basis_fns)
        if prod < 0:
            prod = 0
        preds.append(inv_fn(prod))
    util.write_predictions(preds, ids, outfile)

def screens_idid():
    screens([lambda x:1, lambda x:x[0]], [lambda x:x, lambda x:x], lambda x:x,
            'screens_idid-2.csv')

def screens_idlog():
    screens([lambda x:1, lambda x:x[0], lambda x:x[0]**2],
            [lambda x:x, lambda x:math.log(x)], lambda x:math.e**x, 'screens_idlog-2.csv')

def screens_loglog():
    screens([lambda x:1, lambda x:x[0], lambda x:x[0]**2],
            [lambda x:math.log(x), lambda x:math.log(x)], lambda x:math.e**x,
            'screens_loglog-2.csv')

def screens_budget_lglglg():
    mat,key,regy,_ = rs.extract_feats([rs.metadata_feats])
    screen_ind = key['number_of_screens']
    budget_ind = key['production_budget']
    screens = mat.getcol(screen_ind).todense()
    budget = mat.getcol(budget_ind).todense()

    budget_fns = [lambda x:math.log(x) for i in range(3)]
    budget_check = lambda x:x[1] > 0.
    budget_arr = format_arr([screens,budget], regy, budget_fns, budget_check)
    no_budget_arr = format_arr([screens],regy,[lambda x:math.log(x),lambda x:math.log(x)])

    budget_basis_fns = [lambda x:1, lambda x:x[0], lambda x:x[0]**2,
                        lambda x:x[1], lambda x:x[1]**2]
    no_budget_basis_fns = [lambda x:1, lambda x:x[0], lambda x:x[0]**2]

    budget_coeffs = freg.coeffs(budget_basis_fns, budget_arr)
    no_budget_coeffs = freg.coeffs(no_budget_basis_fns, no_budget_arr)

    test,_,_,ids = rs.extract_feats([rs.metadata_feats],'testcases.xml',
                                    global_feat_dict = key)
    test_len = test.shape[0]
    preds = []
    for i in range(test_len):
        prod = 0
        if test[i,budget_ind] > 0.:
            x = (budget_fns[0](test[i,screen_ind]), budget_fns[1](test[i,budget_ind]))
            prod = freg.product(x, budget_coeffs, budget_basis_fns)
        else:
            x = (math.log(test[i,screen_ind]),)
            prod = freg.product(x, no_budget_coeffs, no_budget_basis_fns)
        if prod < 0:
            prod = 0
        preds.append(math.e**prod)
    util.write_predictions(preds, ids, 'screens_budget_lglglg-2.csv')

def screens_budget_summer_lglglg():
    mat,key,regy,_ = rs.extract_feats([rs.metadata_feats])
    screen_ind = key['number_of_screens']
    budget_ind = key['production_budget']
    summer_ind = key['summer_release']

    screens = mat.getcol(screen_ind).todense()
    budget = mat.getcol(budget_ind).todense()
    summer = mat.getcol(summer_ind).todense()

    def safelog(x):
        if x <= 0.:
            return 0.
        else:
            return math.log(x)
    fns = [safelog, safelog, safelog, safelog]
    bs_check = lambda x:x[1] > 0. and x[2] > 0.
    bns_check = lambda x:x[1] > 0. and x[2] == 0.
    nbs_check = lambda x:x[1] == 0. and x[2] > 0.
    nbns_check = lambda x:x[1] == 0. and x[2] == 0.

    bs_arr = format_arr([screens,budget,summer], regy, fns, bs_check)
    bns_arr = format_arr([screens,budget,summer], regy, fns, bns_check)
    nbs_arr = format_arr([screens,budget,summer], regy, fns, nbs_check)
    nbns_arr = format_arr([screens,budget,summer], regy, fns, nbns_check)

    budget_basis_fns = [lambda x:1, lambda x:x[0], lambda x:x[0]**2,
                        lambda x:x[1], lambda x:x[1]**2]
    no_budget_basis_fns = [lambda x:1, lambda x:x[0], lambda x:x[0]**2]

    bs_coeffs = freg.coeffs(budget_basis_fns, bs_arr)
    bns_coeffs = freg.coeffs(budget_basis_fns, bns_arr)
    nbs_coeffs = freg.coeffs(no_budget_basis_fns, nbs_arr)
    nbns_coeffs = freg.coeffs(no_budget_basis_fns, nbns_arr)

    test,_,_,ids = rs.extract_feats([rs.metadata_feats],'testcases.xml',
                                    global_feat_dict = key)
    test_len = test.shape[0]
    preds = []
    for i in range(test_len):
        prod = 0
        x = [test[i,screen_ind], test[i,budget_ind], test[i,summer_ind]]
        logx = tuple([safelog(feat) for feat in x])
        if bs_check(x):
            prod = freg.product(logx, bs_coeffs, budget_basis_fns)
        elif bns_check(x):
            prod = freg.product(logx, bns_coeffs, budget_basis_fns)
        elif nbs_check(x):
            prod = freg.product(logx, nbs_coeffs, no_budget_basis_fns)
        elif nbns_check(x):
            prod = freg.product(logx, nbns_coeffs, no_budget_basis_fns)
        if prod < 0:
            prod = 0
        preds.append(math.e**prod)
    util.write_predictions(preds, ids, 'screens_budget_summer_lglglg-2.csv')

screens_budget_summer_lglglg()
