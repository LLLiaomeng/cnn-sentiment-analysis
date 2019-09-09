from torch.utils.data import Dataset, DataLoader
import os
from nltk.corpus import stopwords
import model
import train
import mydata

list_stopWords = list(set(stopwords.words('english')))

path = mydata.download_or_unzip()
t1 = os.path.join(path, "rt-polarity.neg")
t2 = os.path.join(path, "rt-polarity.pos")
filename = {t1:0, t2:1}
vocab, label, sentences, cv = mydata.load_data(filename, list_stopWords)

root = os.getcwd()
glove_path = os.path.join(root, "glove\\glove.6B.300d.txt")
word_vec = mydata.load_glove(glove_path, vocab)

word_idx, W = mydata.W_idx(word_vec)					#word-id-vector
data_idx = mydata.data_example(sentences, word_idx)		#sentences convert to id matrix

total_num = 0.0
correct_num = 0.0

for index in range(10):			#cross validation
    train_data, test_data, train_label, test_label = mydata.data_train_test(data_idx, label, cv, index)
    dataset1 = mydata.subDataset(train_data, train_label)
    train_loader = DataLoader(dataset1,batch_size= 100, shuffle = True, num_workers= 0)
    dataset2 = mydata.subDataset(test_data, test_label)
    test_loader = DataLoader(dataset2,batch_size= 100, shuffle = True, num_workers= 0)

    net = model.TextCNN(vec_dim=300, kernel_num=50, vec_num=30,
                  label_num=2,
                  kernel_list=[3,4,5])
    train.train_textcnn_model(W, net, train_loader, epoch=10, lr=0.0005)
    a, b = train.textcnn_model_test(W, net, test_loader)
    total_num += a
    correct_num += b

print('Average accuracy of the network on test set: %f %%' % (100 * correct_num / total_num))        