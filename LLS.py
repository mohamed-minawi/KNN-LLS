import numpy as np

class LLS(object):
    def __init__(self):
        pass

    def getW(self, Tr_Data, Tr_Label):
        rowsnum = Tr_Data.shape[0]
        
        Tr_Data = np.c_[Tr_Data, np.ones(rowsnum)]
        Tr_Label = np.eye(10)[Tr_Label]
       
        self.W = np.dot(np.dot( np.linalg.inv( np.dot(Tr_Data.T, Tr_Data)   ), Tr_Data.T  ), Tr_Label)
        

    def predict(self, test_data):
        
        return np.argmax( np.dot(np.c_[test_data, np.ones(10000)], self.W), axis=1 )