import pandas as pd
import numpy as np
import wfdb
import ast


class OriginalData:
    def __init__(self):
        self.path = './Data/'
        self.sampling_rate = 100
        # load and convert annotation data
        self.Y = pd.read_csv(self.path + 'ptbxl_database.csv', index_col='ecg_id')
        self.Y.scp_codes = self.Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        self.X = self.load_raw_data(self.Y, self.sampling_rate, self.path)

        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(self.path + 'scp_statements.csv', index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        pass

    def load_raw_data(self, df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def getdata(self):
        # Apply diagnostic superclass
        self.Y['diagnostic_superclass'] = self.Y.scp_codes.apply(self.aggregate_diagnostic)
        # Split data into train and test
        test_fold = 10
        # Train
        X_train = self.X[np.where(self.Y.strat_fold != test_fold)]
        y_train = self.Y[(self.Y.strat_fold != test_fold)].diagnostic_superclass
        # Test
        X_test = self.X[np.where(self.Y.strat_fold == test_fold)]
        y_test = self.Y[self.Y.strat_fold == test_fold].diagnostic_superclass

        return X_train, y_train, X_test, y_test


class PrimeData:

    def __init__(self):
        X_train, y_train, X_test, y_test = OriginalData().getdata()
        self.train = []
        self.test = []

        for i in range(len(X_train)):
            if len(y_train.iloc[i]) == 1 and (y_train.iloc[i][0] == 'NORM' or y_train.iloc[i][0] == 'MI'):
                self.train.append((X_train[i], y_train.iloc[i]))

        for i in range(len(X_test)):
            if len(y_test.iloc[i]) == 1 and (y_test.iloc[i][0] == 'NORM' or y_test.iloc[i][0] == 'MI'):  # 找出明确判断结果的数据
                self.test.append((X_test[i], y_test.iloc[i]))

        print("原数据集训练样本数：{}，去除空值后样本数：{}".format(len(X_train), len(self.train)))
        print("原数据集测试样本数：{}，去除空值后样本数：{}".format(len(X_test), len(self.test)))

    # 构造新数据

    # shape is (None,1000,12)
    def dataset01(self):
        X_train_new = []
        y_train_new = []
        X_test_new = []
        y_test_new = []
        for i in range(len(self.train)):
            X_train_new.append(self.train[i][0])
            y_train_new.append(self.train[i][1])
        for i in range(len(self.test)):
            X_test_new.append(self.test[i][0])
            y_test_new.append(self.test[i][1])
        return np.array(X_train_new), y_train_new, np.array(X_test_new), y_test_new

    # shape is (None,12,1000)
    def dataset02(self):
        X_train_new = []
        y_train_new = []
        X_test_new = []
        y_test_new = []
        for i in range(len(self.train)):
            X_train_new.append(self.train[i][0].T)
            y_train_new.append(self.train[i][1])
        for i in range(len(self.test)):
            X_test_new.append(self.test[i][0].T)
            y_test_new.append(self.test[i][1])
        return np.array(X_train_new), y_train_new, np.array(X_test_new), y_test_new

    # shape is (None,12,1000,1)
    def dataset03(self):
        X_train_new = []
        y_train_new = []
        X_test_new = []
        y_test_new = []
        for i in range(len(self.train)):
            X_train_new.append(np.array(self.train[i][0].T).reshape(12, 1000, 1))
            y_train_new.append(self.train[i][1])
        for i in range(len(self.test)):
            X_test_new.append(np.array(self.test[i][0].T).reshape(12, 1000, 1))
            y_test_new.append(self.test[i][1])

        return np.array(X_train_new), y_train_new, np.array(X_test_new), y_test_new

    # shape is (None,1000,12,1)
    def dataset04(self):
        X_train_new = []
        y_train_new = []
        X_test_new = []
        y_test_new = []
        for i in range(len(self.train)):
            X_train_new.append(np.array(self.train[i][0].T).reshape(1000, 12, 1))
            y_train_new.append(self.train[i][1])
        for i in range(len(self.test)):
            X_test_new.append(np.array(self.test[i][0].T).reshape(1000, 12, 1))
            y_test_new.append(self.test[i][1])

        return np.array(X_train_new), y_train_new, np.array(X_test_new), y_test_new

    def label_convnet(self, y_train, y_test):
        label_train = []
        label_test = []
        for i in y_train:
            if (i[0] == 'NORM'):
                label_train.append(1)  ##正常
            elif (i[0] == 'MI'):
                label_train.append(0)  ##异常
        for i in y_test:
            if (i[0] == 'NORM'):
                label_test.append(1)  ##正常
            elif (i[0] == 'MI'):
                label_test.append(0)  ##异常
        return np.array(label_train), np.array(label_test)


class DataLoader:

    def __init__(self):
        self.data = PrimeData()
        self.X_train01, self.y_train01, self.X_test01, self.y_test01 = self.getdata01()
        self.X_train02, self.y_train02, self.X_test02, self.y_test02 = self.getdata02()
        self.X_train03, self.y_train03, self.X_test03, self.y_test03 = self.getdata03()
        self.X_train04, self.y_train04, self.X_test04, self.y_test04 = self.getdata04()
        pass

    def getdata01(self):
        X_train_new, y_train_new, X_test_new, y_test_new = self.data.dataset01()
        y1, y2 =self.data.label_convnet(y_train_new, y_test_new)
        y1 = pd.get_dummies(y1)
        y2 = pd.get_dummies(y2)
        return X_train_new, y1, X_test_new, y2

    def getdata02(self):
        X_train_new, y_train_new, X_test_new, y_test_new = self.data.dataset02()
        y1, y2 = self.data.label_convnet(y_train_new, y_test_new)
        y1 = pd.get_dummies(y1)
        y2 = pd.get_dummies(y2)
        return X_train_new, y1, X_test_new, y2

    def getdata03(self):
        X_train_new, y_train_new, X_test_new, y_test_new = self.data.dataset03()
        y1, y2 = self.data.label_convnet(y_train_new, y_test_new)
        y1 = pd.get_dummies(y1)
        y2 = pd.get_dummies(y2)
        return X_train_new, y1, X_test_new, y2

    def getdata04(self):
        X_train_new, y_train_new, X_test_new, y_test_new = self.data.dataset04()
        y1, y2 = self.data.label_convnet(y_train_new, y_test_new)
        y1 = pd.get_dummies(y1)
        y2 = pd.get_dummies(y2)
        return X_train_new, y1, X_test_new, y2

    def dataload01(self):
        print("X_train\'shape:{}  y_train\'shape:{}  X_test\'shape:{}  y_test\'shape:{}".format(self.X_train01.shape,
                                                                                                self.y_train01.shape,
                                                                                                self.X_test01.shape,
                                                                                                self.y_test01.shape))
        return self.X_train01, self.y_train01, self.X_test01, self.y_test01

    def dataload02(self):
        print("X_train\'shape:{}  y_train\'shape:{}  X_test\'shape:{}  y_test\'shape:{}".format(self.X_train02.shape,
                                                                                                self.y_train02.shape,
                                                                                                self.X_test02.shape,
                                                                                                self.y_test02.shape))
        return self.X_train02, self.y_train02, self.X_test02, self.y_test02

    def dataload03(self):
        print("X_train\'shape:{}  y_train\'shape:{}  X_test\'shape:{}  y_test\'shape:{}".format(self.X_train03.shape,
                                                                                                self.y_train03.shape,
                                                                                                self.X_test03.shape,
                                                                                                self.y_test03.shape))
        return self.X_train03, self.y_train03, self.X_test03, self.y_test03

    def dataload04(self):
        print("X_train\'shape:{}  y_train\'shape:{}  X_test\'shape:{}  y_test\'shape:{}".format(self.X_train04.shape,
                                                                                                self.y_train04.shape,
                                                                                                self.X_test04.shape,
                                                                                                self.y_test04.shape))
        return self.X_train04, self.y_train04, self.X_test04, self.y_test04
