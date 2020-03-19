#!/usr/bin/python
# -*- coding: ascii -*-
import string, os, sys, code, time, multiprocessing, pickle, glob, re, time
import collections, shelve, copy, itertools, math, random, argparse, warnings, getpass
import datetime
import socket

import sys
import platform
print(platform.python_version())
# from pandas.core import sparse
import turicreate as tc
import pandas
import matplotlib
# import matplotlib.pyplot as mplpl

import pandas as pd

# import scikit-learn
import sklearn
import numpy as np
np.set_printoptions( linewidth=200 )
from pandas import Series
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from scipy.sparse import coo_matrix, hstack, csr_matrix
# terminal config
terminal_columns = 300
# this below is just to check if python is in interactive mode
import __main__
if not hasattr(__main__, '__file__'):
   terminal_size = os.popen('stty size', 'r').read().split()
   terminal_rows = int(terminal_size[0])
   terminal_columns = int(terminal_size[1])


# terminal_columns, terminal_rows = pd.util.terminal.get_terminal_size()

import numpy as np
# np.set_printoptions( linewidth=terminal_columns )
np.random.seed(1234567890)
random.seed(1)

import scipy
import scipy.stats
import scipy.optimize

import sklearn
from sklearn import datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.tree
import sklearn.svm
import sklearn.neighbors
# import sklearn.lda
# import sklearn.qda
import sklearn.naive_bayes
import sklearn.feature_selection
import sklearn.preprocessing
# import sklearn.grid_search
from sklearn.model_selection import GridSearchCV # import mord as m

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import basics_fake_news_test
from sklearn.utils import shuffle
# import tweetstxt_basic
# from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
# output properties
# terminal_columns, terminal_rows = pd.util.terminal.get_terminal_size()
# pd.set_option( "display.width", terminal_columns )
# pd.set_option( 'max_columns', terminal_columns/9 )
pd.set_option( 'display.max_columns', 99 )
pd.set_option( "display.width", 300 )

_SQRT2 = np.sqrt(2)


def is_local_machine():
   hostname = socket.gethostname()
   username = getpass.getuser()
   if "pms" in hostname or "MacBook-Pro-3" in hostname or \
           "wks-50-26" in hostname or 'mpi-sws.org' in hostname:
      return 1
   else:
      return 0


def get_dirs():
   hostname = socket.gethostname()
   username = getpass.getuser()
   # print "Your hostname and username:", hostname, username
   # are we on a local machine?
   if "pms" in hostname:
      localdir = "/Users/pms/Dropbox/projects/ad-reactions/"
      # not sure why this, but perhaps this is to work cross-project
      # localdir = "./"
   elif "wks-50-26" in hostname:
      localdir = "/home/babaei/Desktop/SVN/streaming_api/with_tweepy/fake_news_data/"

   elif "MacBook-Pro-3" in hostname or 'mpi-sws.org' in hostname:
      localdir = "/Users/Reza/Desktop/icwsm_svn/code/streaming_api/with_tweepy/fake_news_data/"
   # nope, we are on a server
   elif "pms" in username:
      localdir = "/local/pms/ad-reactions/"
      if "thor" in hostname:
         localdir = "/local/var/tmp/pms/recsys-for-posting/"
   elif "babaei" in username:
      localdir = "/local/Reza/polarization/data/"
      # localdir = "/home/babaei/Desktop/SVN/recsys-for-posting/"
   # remotedir = "/NS/twitter-7/work/Reza/recsys-for-posting/"
   remotedir = "/NS/twitter-7/work/Reza/fake_news/data"
   if is_local_machine(): remotedir = localdir
   return localdir, remotedir


def frange(start, stop, step):
    i = start
    res_list = []
    while i < stop:
        res_list.append(i)
        i += step
    return res_list[:]

def normalized_max_min_funct_sikitlearn(demographicsDict):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(demographicsDict)

    return (X_train_minmax)


import numpy as np
import turicreate as tc

def sort_write_data(df, img_inf):
   img_list=[]
   for i in [1,2,3]:
      for j in range(52):
         img_list.append(str(i) + '_zoomout_fake_'+str(j))


   df.loc[:,'img'] = df['news_id']*0.0
   df['img'] = img_list[:]

   sort_value = 'idpb'
   for ind in df.index.tolist():
      df['idpb'][ind] = np.abs(df['idpb'][ind])
   # sort_value = 'rmse'
   # sort_value = 'accuracy_0_1'
   # sort_value = 'acc_0_1_2'

   df_sort = df.sort_values(sort_value, ascending=0)
   # outF = open('outputFile/sort_rmse.text', 'w')
   # outF = open('outputFile/sort_'+sort_value+'_dem.text', 'w')
   outF = open('outputFile/sort_'+sort_value+'.text', 'w')
   outF.write('|| news id || news text|| Ground Truth || Disputability || TPB || Idelogical Disp '
              '|| Accuracy False and True || Accuracy False, Mixture, and True|| RMSE || Distribution||\n')
   for ind in df_sort.index.tolist():
      outF.write('||' + str(df_sort['news_id'][ind]) + '||' + str(df_sort['news_text'][ind]) + '||' +
                 str(df_sort['gt'][ind]) + '||' + str(df_sort['var'][ind]) + '||' + str(df_sort['tpb'][ind])+'||'+
                 str(df_sort['idpb'][ind]) + '||' + str(df_sort['accuracy_0_1'][ind]) + '||' +
                 str(df_sort['acc_0_1_2'][ind]) + '||' + str(df_sort['rmse'][ind]) + '||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/fig_real_predication_fn/'+str(df['img'][ind])+'}} ||\n')

   idpb_sort=df.sort_values('idpb', ascending=0)

   df_toprnk = idpb_sort.iloc[:10]
   # df_toprnk = idpb_sort
   print(np.mean(df_toprnk['rmse']))
   # print(np.mean(df_low['rmse']))
   print(np.mean(df_toprnk['accuracy_0_1']))
   # print(np.mean(df_low['accuracy_0_1']))
   print(np.mean(df_toprnk['acc_0_1_2']))
   # print(np.mean(df_low['acc_0_1_2']))


   print(np.mean(df_sort['rmse']))
   print(np.mean(df_sort['accuracy_0_1']))
   print(np.mean(df_sort['acc_0_1_2']))


   print(np.corrcoef(df_sort[sort_value], df_sort['gt'])[0])
   print(np.corrcoef(df_sort[sort_value], df_sort['var'])[0])
   print(np.corrcoef(df_sort[sort_value], df_sort['tpb'])[0])
   print(np.corrcoef(df_sort[sort_value], df_sort['idpb'])[0])


   print(scipy.stats.spearmanr(df_sort[sort_value], df_sort['gt']))
   print(scipy.stats.spearmanr(df_sort[sort_value], df_sort['var']))
   print(scipy.stats.spearmanr(df_sort[sort_value], df_sort['tpb']))
   print(scipy.stats.spearmanr(df_sort[sort_value], df_sort['idpb']))


   # {{http: // www.mpi - sws.mpg.de / ~babaei / result - fig / growth - rate / Daily_RT1.gif}} | |

   # print(df['img'])

def my_split(df_in, col_n, thr):
   df_in
   total_c = 0
   train_rating_all = []
   train_user_all = []
   train_news_all=[]
   user_test=[]
   rating_test=[]
   news_test=[]
   for i in range(52):
      list_user=[]
      list_rating=[]
      news_train=[]
      lln = df_in.index.tolist()
      if i==col_n:
         # rnd_list = range(len(lln))
         rnd_list = list(lln[:])
         rnd_list = shuffle(rnd_list)
         rnd_l = rnd_list[:int(thr*100)]

         # for x in range(len(lln)):
         for x in lln:
            if x in rnd_l:
               user_test.append(x)
               rating_test.append(df_in['s_label_' + str(i)][x])
               news_test.append(i)
            else:
               list_user.append(x)
               list_rating.append(df_in['s_label_' + str(i)][x])
               total_c += (len(list_user) - int(thr * 100))
               news_train.append(i)
      else:
         list_rating = list(df_in['s_label_' + str(i)])
         news_train = [i]*len(list_rating)
         list_user =df_in.index.tolist()
         total_c += len(list_user)


      train_rating_all+=list_rating[:]
      train_user_all+=list_user[:]
      train_news_all+=news_train[:]
      train_index=range(len(train_rating_all))

   df_train = pd.DataFrame({'worker_id': Series(train_user_all, index=train_index),
                      'news_id': Series(train_news_all, index=train_index),
                      'rating': Series(train_rating_all, index=train_index), })

   df_test = pd.DataFrame({'worker_id': Series(user_test, index=range(len(user_test))),
                      'news_id': Series(news_test, index=range(len(user_test))),
                      'rating': Series(rating_test, index=range(len(user_test)))})

   return df_train, df_test


def main():
   # tr_test = int(sys.argv[1])
   tr_test = 10
   # fdf_s = pd.read_csv('nmf_news_'+str(tr_test)+'.csv', sep="\t")
   fdf_s = pd.read_csv('nmf_news_dem_'+str(tr_test)+'.csv', sep="\t")
   img_inf = 'fig_real_predication_fn/'
   # sort_write_data(fdf_s, img_inf)
   # exit()
   gt_list = []
   tpb_list = []
   var_list = []
   idPb_list = []
   rmse_list = []
   acc_list = []
   acc_3_list = []
   diff1_list = []
   newsid_list = []
   newstext_list = []

   acc_0_1_list_g = []
   acc_3_list_g = []
   acc_less_list_g = []
   rmse_list_g = []

   for fileInd in [1,2,3]:

      news_id = collections.defaultdict()
      news_text = collections.defaultdict()
      news_gt = collections.defaultdict()
      news_tpb = collections.defaultdict()
      news_var = collections.defaultdict()
      news_pol = collections.defaultdict()

      fdf_s = pd.read_csv('Workers-perception-' + str(fileInd) + '.csv', sep="\t")
      for ind in fdf_s.index.tolist():
         news_id[ind] = fdf_s['text_id'][ind]
         news_text[ind] = fdf_s['text'][ind]

      ff = pd.read_csv('pred_real_labaling_'+ str(fileInd) + '.csv', sep="\t")
      ff_dem = ff[ff['political_view'].isin(['liberal', 'veryliberal'])]
      ff_rep = ff[ff['political_view'].isin(['conservative', 'veryconservative'])]

      ff_dem_index = ff_dem.index.tolist()
      ff_rep_index = ff_rep.index.tolist()

      ff_n = ff[['s_label_' + str(x) for x in range(0, 52)]]
      ff_n_dem = ff_n.iloc[ff_dem_index[:]]
      ff_n_rep = ff_n.iloc[ff_rep_index[:]]
      # ind_t = df_tmp.index.tolist()[0]

      for i in range(52):

         news_gt[i] = list(ff['real_label_' + str(i)])[0]
         tpb_t = 0
         for j in range(len(ff.index.tolist())):
            tpb_t += np.abs(ff['s_label_' + str(i)][j] - news_gt[i])
         news_tpb[i] = float(tpb_t) / float(j)
         news_var[i] = np.var(list(ff['s_label_' + str(i)]))

         groupby_ftr = 'political_view'
         df_tmp = ff.copy()  # ['s_label_' + str(i)]
         # ind_t = df_tmp.index.tolist()[0]
         dem_df = df_tmp[df_tmp['political_view'].isin(['liberal', 'veryliberal'])]
         rep_df = df_tmp[df_tmp['political_view'].isin(['conservative', 'veryconservative'])]

         dem_v = np.mean(dem_df['s_label_' + str(i)])
         rep_v = np.mean(rep_df['s_label_' + str(i)])
         news_pol[i] = dem_v - rep_v



      rmse_dict= collections.defaultdict()
      acc_dict= collections.defaultdict()
      acc_dict_3= collections.defaultdict()
      acc_less1_dict= collections.defaultdict()
      ff_n_run = ff_n
      # ff_n_run = ff_n_dem
      # ff_n_run = ff_n_rep
      for news_ind in range(52):
         acc_0_1_list_local = []
         acc_3_list_local = []
         acc_less_list_local = []
         rmse_list_local = []


         for test_run in range(5):
            try:
               df_train, df_test = my_split(ff_n_run, news_ind, tr_test * float(0.01))
               # ff_n.to_csv('test_rating_'+str(tr_test)+'.csv', columns=ff_n.columns, sep="\t", index=False)


               rating_all=[]
               userId_all=[]
               newsId_all=[]


               df_train.to_csv('df_train'+str(tr_test)+'.csv', columns=df_train.columns, sep="\t", index=False)
               df_test.to_csv('df_test'+str(tr_test)+'.csv', columns=df_test.columns, sep="\t", index=False)





               # data = tc.SFrame.read_csv('test_rating.csv',
               #                           delimiter='\t')
               # # usage_data.rename({'X1': 'news_id', 'X2': 'rating', 'X3': 'worker_id'})
               # training_data, test_data = tc.recommender.util.random_split_by_user(data, 'worker_id', 'news_id',item_test_proportion=0.2)

               training_data = tc.SFrame.read_csv('df_train'+str(tr_test)+'.csv',delimiter='\t')
               test_data = tc.SFrame.read_csv('df_test'+str(tr_test)+'.csv',delimiter='\t')

               model = tc.recommender.factorization_recommender.create(training_data, user_id='worker_id', item_id='news_id',
                                                                       target='rating', num_factors=2, max_iterations=20,
                                                                       solver='sgd')

               predictions = model.predict(test_data)
               rmse_list_local.append(tc.evaluation.rmse(test_data['rating'], predictions))

               y_pred = predictions
               y_true = test_data['rating']

               val = tc.evaluation.log_loss(y_true, y_pred)
               # cr_ent = sklearn.metrics.log_loss(y_true, y_pred)

               test_list = test_data['rating']


               predictions = np.array(predictions)
               test_list = test_list.astype(int)

               acc=0
               acc_p=0
               acc_n=0
               acc_p_t=0
               acc_n_t=0
               acc_t=0
               pred_2 = 0
               test_2 = 0
               pp=0
               acc_0_1=0


               # print(predictions)
               for i in range(len(predictions)):

                  if test_list[i]==3:
                     test_2+=1
                     continue

                  if predictions[i]<3 and test_list[i]<3:
                     acc+=1

                  if predictions[i]>3 and test_list[i]>3:
                     acc+=1

                  # if np.abs(predictions[i] - test_list[i])<=1:
                  #    pp+=1


                  acc_t+=1
               acc_0_1_list_local.append(float(acc) / acc_t)

               acc_t=0
               acc=0
               pp=0
               for i in range(len(predictions)):

                  if predictions[i]==3 and test_list[i]==3:
                     acc+=1
                     # continue

                  if predictions[i]<3 and test_list[i]<3:
                     acc+=1

                  if predictions[i]>3 and test_list[i]>3:
                     acc+=1

                  if np.abs(predictions[i] - test_list[i])<=1:
                     pp+=1


                  acc_t+=1

               acc_3_list_local.append(float(acc)/acc_t)
               acc_less_list_local.append(float(pp)/acc_t)

            except:
               continue

         acc_dict[news_ind] =np.mean(acc_0_1_list_local)
         acc_dict_3[news_ind] = np.mean(acc_3_list_local)
         acc_less1_dict[news_ind] = np.mean(acc_less_list_local)

         rmse_dict[news_ind] = np.mean(rmse_list_local)

      for i in range(52):
         newsid_list.append(news_id[i])
         newstext_list.append(news_text[i])
         gt_list.append(news_gt[i])
         tpb_list.append(news_tpb[i])
         var_list.append(news_var[i])
         idPb_list.append(news_pol[i])
         rmse_list_g.append(rmse_dict[i])
         acc_0_1_list_g.append(acc_dict[i])
         acc_3_list_g.append(acc_dict_3[i])
         acc_less_list_g.append(acc_less1_dict[i])


   index_df = range(len(newsid_list))

   df_main = pd.DataFrame({'news_id': Series(newsid_list, index=index_df),
                           'news_text': Series(newstext_list, index=index_df),
                           'gt': Series(gt_list, index=index_df),
                           'tpb': Series(tpb_list, index=index_df),
                           'var': Series(var_list, index=index_df),
                           'idpb': Series(idPb_list, index=index_df),
                           'rmse': Series(rmse_list_g, index=index_df),
                           'accuracy_0_1': Series(acc_0_1_list_g, index=index_df),
                           'acc_0_1_2': Series(acc_3_list_g, index=index_df),
                           'diff_less_1': Series(acc_less_list_g, index=index_df),

                           })

   df_main.to_csv('nmf_news_cross_entropy_'+str(tr_test)+'.csv', columns=df_main.columns, sep="\t", index=False)


if __name__ == '__main__':

    main()

