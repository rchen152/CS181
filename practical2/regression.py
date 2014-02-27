import freq_reg as reg
import numpy as np
import regression_starter as rs
import util

train_mat,key,train_regy,_ = rs.extract_feats([rs.metadata_feats])
train_screens = train_mat.getcol(key['number_of_screens']).todense()
tlen = len(train_screens)

train_array = np.zeros((tlen,2))
train_array[:,0] = train_screens.reshape((tlen))
train_array[:,1] = train_regy.transpose()

test_mat,_,_,test_ids = rs.extract_feats([rs.metadata_feats],'testcases.xml',
                                         global_feat_dict = key)
test_screens = test_mat.getcol(key['number_of_screens']).todense()
test_screens = test_screens.reshape((len(test_screens[0])))
print test_screens
print len(test_screens)

preds = reg.regress([lambda x:1, lambda x:x], train_array, test_screens)
util.write_predictions(preds, test_ids, 'screens-1.csv')
