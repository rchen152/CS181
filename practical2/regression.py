import freq_reg
import regression_starter as rs
from sklearn import 

train_mat,key,train_regy,_ = rs.extract_feats([rs.metadata_feats])
train_screens = train_mat.getcol(key['number_of_screens']).todense().transpose()

test_mat,_,_,test_ids = rs.extract_feats([rs.metadata_feats],'testcases.xml',
                                         global_feat_dict = key)
test_screens = test_mat.getcol(key['number_of_screens']).todense().transpose()

preds = # get some predictions
util.write_predictions(preds, test_ids, 'screens-1.csv')
