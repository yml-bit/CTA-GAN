# CTA-GAN
synthesis CTA using CT based on CTA-GAN model
trianing stage
fist stage：make you data list.
(1) process your onw data,it's best that paired NCCTA and CTA data keep on same path,as CT_CTA/person1/CT and CT_CTA/person1/CTA.
(2) make your own data list by data_process.py or your own code.it include the division of training set, validation set and test set. 

second stage:trianing
(1)Put your data list path where it belongs.note that as the dataprocess.py just get the CT list,you should make some chage according own data path about CTA.
(2) set own model save path or learning rate in CTA_GAN\Yaml\.yaml.
(3) select the training model in CTA_GAN\train.py and set trianing
(4) chage the training to testing,evaluate the trained model.
more detail can see the code.

