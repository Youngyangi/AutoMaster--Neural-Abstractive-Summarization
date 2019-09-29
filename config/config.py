import os.path
import pathlib

pwd_path = pathlib.Path(os.path.abspath(__file__)).parent.parent

output_dir = os.path.join(pwd_path, 'Data_set')

train_path = os.path.join(output_dir, 'AutoMaster_TrainSet.csv')
test_path = os.path.join(output_dir, 'AutoMaster_TestSet.csv')

stop_word_path = os.path.join(pwd_path, 'utilities/stopwords-zh.txt')

train_seg_path = os.path.join(output_dir, 'AutoMaster_Train_Seg_Set.csv')
train_contend_path = os.path.join(output_dir, 'AutoMaster_Train_Contend.txt')

w2v_bin_path = os.path.join(output_dir, 'w2v.bin')