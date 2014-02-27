import freq_reg as reg
import math
import numpy as np
import matplotlib.pyplot as plt
import regression_starter as rs
import util

def screens_only_double_log():
    train_mat,key,train_regy,_ = rs.extract_feats([rs.metadata_feats])
    train_screens = train_mat.getcol(key['number_of_screens']).todense()
    train_len = len(train_screens)

    train_array = np.zeros((train_len,2))
    train_array[:,0] = train_screens.reshape((train_len))
    train_array[:,1] = train_regy.transpose()

    for i in range(train_len):
        train_array[i,0] = math.log(train_array[i,0])
        train_array[i,1] = math.log(train_array[i,1])

    test_mat,_,_,test_ids = rs.extract_feats([rs.metadata_feats],
                                             'testcases.xml',
                                             global_feat_dict = key)
    test_screens = test_mat.getcol(key['number_of_screens']).todense()
    test_len = test_screens.shape[0]
    for i in range(test_len):
        test_screens[i,0] = math.log(test_screens[i,0])

    preds = reg.regress([lambda x:1, lambda x:x, lambda x:x**2],
                        train_array, test_screens)

    pred_lst = [0 if preds[i] < 0 else math.e**preds[i]
                for i in range(test_len)]

    '''measure = 0.
    for i in range(test_len):
        measure += math.fabs(pred_lst[i] - train_regy[i])
    print measure/test_len'''

    plt.plot(test_screens,pred_lst,'bo')
    plt.show()

    util.write_predictions(pred_lst, test_ids, 'screens_double_log.csv')

def screens_only_log():
    train_mat,key,train_regy,_ = rs.extract_feats([rs.metadata_feats])
    train_screens = train_mat.getcol(key['number_of_screens']).todense()
    train_len = len(train_screens)

    train_array = np.zeros((train_len,2))
    train_array[:,0] = train_screens.reshape((train_len))
    train_array[:,1] = train_regy.transpose()

    for i in range(train_len):
        train_array[i,1] = math.log(train_array[i,1])

    test_mat,_,_,test_ids = rs.extract_feats([rs.metadata_feats],
                                             'testcases.xml',
                                             global_feat_dict = key)
    test_screens = test_mat.getcol(key['number_of_screens']).todense()
    test_len = test_screens.shape[0]

    preds = reg.regress([lambda x:1, lambda x:x, lambda x:x**2],
                        train_array, test_screens)

    plt.plot(test_screens,preds,'bo')
    plt.show()

    pred_lst = [0 if preds[i] < 0 else math.e**preds[i]
                for i in range(test_len)]

    '''measure = 0.
    for i in range(test_len):
        measure += math.fabs(pred_lst[i] - train_regy[i])
    print measure/test_len'''

    util.write_predictions(pred_lst, test_ids, 'screens_log.csv')

def screens_and_budget():
    train_mat,key,train_regy,_ = rs.extract_feats([rs.metadata_feats])
    train_screens = train_mat.getcol(key['number_of_screens']).todense()
    train_budget = train_mat.getcol(key['production_budget']).todense()

    train_len = len(train_screens)

    #train_ind_dict = {}
    budget = []
    for i in range(train_len):
        if train_budget[i,0] > 0.:
            #train_ind_dict[len(budget)] = i
            budget.append([(math.log(train_screens[i,0]),
                            math.log(train_budget[i,0])),
                           math.log(train_regy[i])])
    budget_arr = np.array(budget)

    no_budget_arr = np.zeros((train_len,2))
    no_budget_arr[:,0] = train_screens.reshape((train_len))
    no_budget_arr[:,1] = train_regy.transpose()
    for i in range(train_len):
        no_budget_arr[i,0] = math.log(no_budget_arr[i,0])
        no_budget_arr[i,1] = math.log(no_budget_arr[i,1])
    
    test_mat,_,_,test_ids = rs.extract_feats([rs.metadata_feats],
                                             'testcases.xml',
                                             global_feat_dict = key)
    test_screens = test_mat.getcol(key['number_of_screens']).todense()
    test_budget = test_mat.getcol(key['production_budget']).todense()
    test_len = len(test_screens)

    test_ind_dict = {}
    budget_lst = []
    no_budget_lst = []
    for i in range(test_len):
        if test_budget[i,0] > 0.:
            test_ind_dict['budget-' + str(len(budget_lst))] = i
            budget_lst.append([(math.log(test_screens[i,0]),
                                math.log(test_budget[i,0]))])
        else:
            test_ind_dict['none-' + str(len(no_budget_lst))] = i
            no_budget_lst.append([math.log(test_screens[i,0])])
    test_budget_arr = np.array(budget_lst)
    test_no_budget_arr = np.array(no_budget_lst)

    preds_budget = reg.regress([lambda (x,y):1,lambda (x,y):x,lambda (x,y):x**2,
                                lambda (x,y):y, lambda (x,y):y**2],
                               budget_arr, test_budget_arr)
    preds_no_budget = reg.regress([lambda x:1, lambda x:x, lambda x:x**2],
                                  no_budget_arr, test_no_budget_arr)

    preds = [0 for i in range(test_len)]
    for i in range(len(preds_budget)):
        preds[test_ind_dict['budget-'+str(i)]] = preds_budget[i]
    for i in range(len(preds_no_budget)):
        preds[test_ind_dict['none-'+str(i)]] = preds_no_budget[i]

    pred_lst = [0 if preds[i] < 0 else math.e**preds[i]
                for i in range(test_len)]

    '''measure = 0.
    for i in range(test_len):
        measure += math.fabs(pred_lst[i] - train_regy[i])
    print measure/test_len'''

    #plt.plot(test_screens,pred_lst,'bo')
    #plt.show()

    util.write_predictions(pred_lst, test_ids, 'screens_log_budget_log.csv')

def screens_and_budget():
    train_mat,key,train_regy,_ = rs.extract_feats([rs.metadata_feats])
    train_screens = train_mat.getcol(key['number_of_screens']).todense()
    train_budget = train_mat.getcol(key['production_budget']).todense()

    train_len = len(train_screens)

    budget = []
    for i in range(train_len):
        if train_budget[i,0] > 0.:
            budget.append([(math.log(train_screens[i,0]),
                            math.log(train_budget[i,0])),
                           math.log(train_regy[i])])
    budget_arr = np.array(budget)

    no_budget_arr = np.zeros((train_len,2))
    no_budget_arr[:,0] = train_screens.reshape((train_len))
    no_budget_arr[:,1] = train_regy.transpose()
    for i in range(train_len):
        no_budget_arr[i,0] = math.log(no_budget_arr[i,0])
        no_budget_arr[i,1] = math.log(no_budget_arr[i,1])
    
    test_mat,_,_,test_ids = rs.extract_feats([rs.metadata_feats],
                                             'testcases.xml',
                                             global_feat_dict = key)
    test_screens = test_mat.getcol(key['number_of_screens']).todense()
    test_budget = test_mat.getcol(key['production_budget']).todense()
    test_len = len(test_screens)

    test_ind_dict = {}
    budget_lst = []
    no_budget_lst = []
    for i in range(test_len):
        if test_budget[i,0] > 0.:
            test_ind_dict['budget-' + str(len(budget_lst))] = i
            budget_lst.append([(math.log(test_screens[i,0]),
                                math.log(test_budget[i,0]))])
        else:
            test_ind_dict['none-' + str(len(no_budget_lst))] = i
            no_budget_lst.append([math.log(test_screens[i,0])])
    test_budget_arr = np.array(budget_lst)
    test_no_budget_arr = np.array(no_budget_lst)

    preds_budget = reg.regress([lambda (x,y):1,lambda (x,y):x,lambda (x,y):x**2,
                                lambda (x,y):y, lambda (x,y):y**2],
                               budget_arr, test_budget_arr)
    preds_no_budget = reg.regress([lambda x:1, lambda x:x, lambda x:x**2],
                                  no_budget_arr, test_no_budget_arr)

    preds = [0 for i in range(test_len)]
    for i in range(len(preds_budget)):
        preds[test_ind_dict['budget-'+str(i)]] = preds_budget[i]
    for i in range(len(preds_no_budget)):
        preds[test_ind_dict['none-'+str(i)]] = preds_no_budget[i]

    pred_lst = [0 if preds[i] < 0 else math.e**preds[i]
                for i in range(test_len)]

    '''measure = 0.
    for i in range(test_len):
        measure += math.fabs(pred_lst[i] - train_regy[i])
    print measure/test_len'''

    #plt.plot(test_screens,pred_lst,'bo')
    #plt.show()

    util.write_predictions(pred_lst, test_ids, 'screens_log_budget_log.csv')

def product(x, coeffs, basis_fns):
    out = 0.
    for i in range(len(basis_fns)):
        out += coeffs[i]*basis_fns[i](x)
    return out

def screens_budget_summer():
    train_mat,key,train_regy,_ = rs.extract_feats([rs.metadata_feats])
    train_len = train_mat.shape[0]
    summer_lst = []
    no_summer_lst = []
    screen_ind = key['number_of_screens']
    budget_ind = key['production_budget']
    for i in range(train_len):
        if train_mat[i,key['summer_release']] == 1.:
            summer_lst.append([train_mat[i,screen_ind],train_mat[i,budget_ind],
                               train_regy[i]])
        else:
            no_summer_lst.append([train_mat[i,screen_ind],
                                  train_mat[i,budget_ind],train_regy[i]])
    summer_mat = np.array(summer_lst)
    no_summer_mat = np.array(no_summer_lst)

    basis_fns_budget = [lambda (x,y):1,lambda (x,y):x,lambda (x,y):x**2,
                        lambda (x,y):y, lambda (x,y):y**2]
    basis_fns_no_budget = [lambda x:1, lambda x:x, lambda x:x**2]

    def split(mat):
        screens = mat[:,0]
        budget = mat[:,1]
        regy = mat[:,2]
        mlen = mat.shape[0]

        budget_lst = []
        no_budget_lst = []
        for i in range(mlen):
            if budget[i] > 0.:
                budget_lst.append([(math.log(screens[i]),
                                    math.log(budget[i])),
                                   math.log(regy[i])])
            else:
                no_budget_lst.append([math.log(screens[i]),math.log(regy[i])])

        budget_arr = np.array(budget_lst)
        no_budget_arr = np.array(no_budget_lst)
        coeffs1 = reg.coeffs(basis_fns_budget,budget_arr)
        coeffs2 = reg.coeffs(basis_fns_no_budget,no_budget_arr)
        return (coeffs1,coeffs2)
                
    s_b_coeffs, s_nb_coeffs = split(summer_mat)
    ns_b_coeffs, ns_nb_coeffs = split(no_summer_mat)

    test_mat,_,_,test_ids = rs.extract_feats([rs.metadata_feats],'testcases.xml',
                                             global_feat_dict = key)
    test_len = test_mat.shape[0]
    preds = []
    for i in range(test_len):
        if test_mat[i,key['summer_release']] == 0.:
            if test_mat[i,key['production_budget']] == 0.:
                preds.append(product(math.log(test_mat[i,key['number_of_screens']]),
                                     ns_nb_coeffs,basis_fns_no_budget))
            else:
                preds.append(product((math.log(test_mat[i,key['number_of_screens']]),
                                      math.log(test_mat[i,key['production_budget']])),
                                     ns_b_coeffs,basis_fns_budget))
        else:
            if test_mat[i,key['production_budget']] == 0.:
                preds.append(product(math.log(test_mat[i,key['number_of_screens']]),
                                     s_nb_coeffs,basis_fns_no_budget))
            else:
                preds.append(product((math.log(test_mat[i,key['number_of_screens']]),
                                      math.log(test_mat[i,key['production_budget']])),
                                     s_b_coeffs,basis_fns_budget))

    pred_lst = [1 if preds[i] < 0 else math.e**preds[i] for i in range(test_len)]

    util.write_predictions(pred_lst, test_ids, 'screens_budget_summer.csv')

screens_budget_summer()
