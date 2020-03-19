#!/usr/bin/python
# -*- coding: ascii -*-
import string, os, sys, code, time, multiprocessing, pickle, glob, re, time
import collections, shelve, copy, itertools, math, random, argparse, warnings, getpass
import datetime
import socket

import sys
# from pandas.core import sparse

import pandas
import matplotlib
import matplotlib.pyplot as mplpl

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

def add_med_mean_features(fdf, feature_name, groupby_name):

   df = fdf.copy()

   df.loc[:, 'mean_'+feature_name] = df[feature_name] * 0.0
   df.loc[:, 'med_'+feature_name] = df[feature_name] * 0.0
   df.loc[:, 'above_' + feature_name + '_mean'] = df[feature_name] * 0.0
   df.loc[:, 'above_' + feature_name + '_med'] = df[feature_name] * 0.0
   if 'created_at' not in df:
      df.loc[:, 'created_at'] = df[feature_name] * 0.0

   groupname_list = list(set(list(df[groupby_name][:])))

   grby_mean = df.groupby(groupby_name).mean()
   grby_med = df.groupby(groupby_name).median()
   for source_id in groupname_list:
      source_mean = grby_mean[feature_name][source_id]
      source_med = grby_med[feature_name][source_id]
      index_list = df[df[groupby_name] == source_id].index.tolist()
      for index_m in index_list:
         df['mean_'+feature_name][index_m] = source_mean
         df['med_'+feature_name][index_m] = source_med
         df['tweet_text'][index_m] = fdf['tweet_text'][index_m]
         df['created_at'][index_m] = index_m
         if df[feature_name][index_m] >= source_mean:
            df['above_' + feature_name + '_mean'][index_m] = 1
         else:
            df['above_' + feature_name + '_mean'][index_m] = 0

         if source_med==0:
            if df[feature_name][index_m] > source_med :
               df['above_' + feature_name + '_med'][index_m] = 1
            else:
               df['above_' + feature_name + '_med'][index_m] = 0
         else:
            if df[feature_name][index_m] >= source_med:
               df['above_' + feature_name + '_med'][index_m] = 1
            else:
               df['above_' + feature_name + '_med'][index_m] = 0

   return df

def hellinger1(p, q):
   return scipy.linalg.norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2

def hellinger2(p, q):
   return scipy.spatial.distance.euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def hellinger3(p, q):
   return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


def preprocess( filename, dirs, fdf,fdf_t, nshalf,
   # dependentvar='above_num_retweets_med',
   # dependentvar='above_num_retweets_mean',
   dependentvar,Train_trsh,
   tm_featnm="label", num_class=2 ):

   if tm_featnm not in fdf.columns:
      fdf.loc[:,tm_featnm] = fdf.index
      fdf_t.loc[:, tm_featnm] = fdf_t.index

   # m_thresh = 0.2

   pd.options.mode.chained_assignment = None
   df_dict = collections.defaultdict(pd.DataFrame)
   for nm_cls in list(set(fdf[dependentvar][:])):
      # df_dict[nm_cls] = fdf.ix[fdf[dependentvar]==nm_cls, : ]
      df_dict[nm_cls] = fdf[fdf[dependentvar]==nm_cls]

   # dfpos = fdf.ix[ fdf[dependentvar]>m_thresh, : ]
   # dfneg = fdf.ix[ fdf[dependentvar]<=m_thresh, : ]

   # print dfpos.iloc[ 0:20 ]
   # print dfneg.iloc[ 0:20 ]

   # print "Number of samples diff classes", len(dfpos), len(dfneg)
   print ("Number of samples diff classes", str([len(df_dict[i_cls]) for i_cls in df_dict]))
   # inspect distribution of tweet times
   def inspect_timestamps( tms, description ):
      tms = list(tms)
      quantiles = np.percentile( tms, np.arange( 0, 101, 10).tolist() )
      print ("Quantiles of tweet time", description)
      def hrtime( epoch ):
         return datetime.datetime.fromtimestamp( int(epoch) ).strftime(
            '%Y-%m-%d %H:%M:%S' )
      for quantile in quantiles:
         print (hrtime(quantile))

   '''
   # inspect distribution of tweet times
   try:
      inspect_timestamps( dfpos[tm_featnm].values, "pos" )
      inspect_timestamps( dfneg[tm_featnm].values, "neg" )
   # except ValueError:
   #    print "ValueError in inspect_timestamps, ignoring"
   except:
      print "ValueError in inspect_timestamps, ignoring"
   '''
   # filtering
   featstotrain  = basics_fake_news_test.get_featnames_from_filename( filename )
   fdf = basics_fake_news_test.filter_data( fdf, verbose=True )


   return subsample_and_balance( filename, nshalf, df_dict,fdf_t, Train_trsh, tm_featnm )


   # return subsample_and_balance(filename, nshalf, dfp, dfn, "index")


# def subsample_and_balance( filename, nshalf, df_dict, tm_featnm="created_at" ):
def subsample_and_balance(filename, nshalf, df_dict,df_t, Train_trsh, tm_featnm="createdat"):

   for i_cls in df_dict:
      if tm_featnm not in df_dict[i_cls].columns:
         df_dict[i_cls].loc[:,tm_featnm] = df_dict.index
      if len(df_dict[i_cls])<1:
         return df_dict,df_dict,[],[],[]


   for i_cls in df_dict:
      df_dict[i_cls].loc[:,"rnd1"] = np.random.rand(len(df_dict[i_cls]))
   # dfp.loc[:,"label"] = 1
   # dfn.loc[:,"label"] = 0

   pd.options.mode.chained_assignment = "warn"

   if "mtest" in filename:
      trainratio = Train_trsh
      testratio = 1.0-trainratio
      ntestslices = 5

   if "stest" in filename:
      trainratio = Train_trsh
      testratio = 1.0-trainratio
      ntestslices = 1
   print(Train_trsh)
   df_dict_1 = collections.defaultdict(pd.DataFrame)
   df_dict_2 = collections.defaultdict(pd.DataFrame)

   for i_cls in df_dict:
      tms = df_dict[i_cls][tm_featnm]
      splittm = np.percentile( tms, int(trainratio*100) )
      # splittm_train = np.percentile( tms, int(100) )
      splittm_train = np.percentile( tms, int(trainratio * 100) )
      df_dict_1[i_cls] = df_dict[i_cls].iloc[ np.where( df_dict[i_cls][tm_featnm]<=splittm_train )[0], : ]
      # dict[i_cls].iloc[ np.where( df_dict[i_cls][tm_featnm]<splittm_train )[0], : ]
      df_dict_2[i_cls] = df_dict[i_cls].iloc[ np.where( df_dict[i_cls][tm_featnm]>splittm_train )[0], : ]


   print ("After time-horizon split:", str([(len(df_dict_1[i_cls]), len(df_dict_2[i_cls])) for i_cls in df_dict]))



   # prepare the output df
   dftrain = pd.concat([ df_dict_1[i_cls] for i_cls in df_dict ])
   # dftrain.sort( tm_featnm, inplace=True  )
   dftest = pd.concat([ df_dict_2[i_cls] for i_cls in df_dict ])
   # dftest.sort( tm_featnm, inplace=True  )
   df = pd.concat([ dftrain, dftest ])
   positions_train = set(range(len( dftrain )))
   positions_test = set(range(len( dftest )))

   # dftrain = df.copy()
   # dftest = df.copy()

   print ("Train and test sizes:", len(dftrain), len(dftest))
   # print "Train and test ratios:", 1.0*len(dftrain)/len(df), 1.0*len(dftest)/len(df)

   # prepare the output long balancedc df
   # nsampleslong = min( [len(dfp), len(dfn)] )
   nsampleslong = len(dftrain) + len(dftest)
   for i_cls in df_dict:
      nsampleslong = min(len(df_dict[i_cls]), nsampleslong)

   first_round = 0
   for i_cls in df_dict:
      if first_round==0:
         df1long = df_dict[i_cls].loc[random.sample(df_dict[i_cls].index.tolist(), nsampleslong), :]
         first_round=1
      else:
         df2long = df_dict[i_cls].loc[random.sample(df_dict[i_cls].index.tolist(), nsampleslong), :]
   # df2long = dfn.loc[random.sample(dfn.index, nsampleslong), :]
         df1long = df1long.append( df2long )

   dflong = df1long
   # get the positions for train and test sets
   # positions_all = set(range(len(df_t)))
   positions_all = set(range(len(df_t)))
   positions_test_all = sorted(list(positions_all - positions_train))
   # positions_test_all = sorted(list(positions_all))
   positions_train = list(positions_train)
   slicesize = len(positions_test_all)/ntestslices
   posslices_test = []
   for i in range(ntestslices):
      posslices_test += [ positions_test_all[ int(slicesize*i) : int(slicesize*(i+1)) ] ]

   return df,df_t, dflong, positions_train, posslices_test




def classify( filename, dirs, df,df_t, dflong,
   positions_train, posslices_test, nshalf,
   classifier, featset,dependentvar, tm_featnm="created_at" ):
   # ignore "DeprecationWarning: `fprob` is deprecated"
   warnings.simplefilter('ignore', DeprecationWarning)
   # ignore "Mean of empty slice.", RuntimeWarning
   warnings.simplefilter('ignore', RuntimeWarning)

   # df["rnd2"] = df["label"] + np.random.rand(len(df))
   df["rnd2"] = np.random.rand(len(df))
   dflong["rnd2"] = np.random.rand(len(dflong))


   positions_test = posslices_test[0]
   # print positions_test

   propstoclean, featstotrain  = \
      basics_fake_news_test.get_featnames_from_filename( filename )

   propdict = collections.defaultdict(list)
   # propdict["featname"] = featstotrain + [ "rnd1", "rnd2", "sima_exp_meme_rnd" ]
   propdict["featname"] = featstotrain + [ "rnd1", "rnd2" ]
   propdict["featname"] = featstotrain
   print ("Predicting feats:", propdict["featname"])




   labels = df.iloc[positions_train].loc[:,dependentvar].values
   samplesdf = df.iloc[positions_train].loc[:,tuple(propdict["featname"])]
   print ("Negative values:")
   # for col in samplesdf:
   #    if np.sum(samplesdf[col]<0):
   #       print col, np.sum(samplesdf[col]<0)
         # print samplesdf.ix[ samplesdf[col]<0, col ]
   samples = samplesdf.as_matrix()

   labels_test = df_t.iloc[positions_test].loc[:,dependentvar].values
   samples_test = \
      ( df_t.iloc[positions_test].loc[:,tuple(propdict["featname"])] ).as_matrix()
   samples_testdf_full = df_t.iloc[positions_test]
   # labslices_test = []
   # splslices_test = []
   # for myslice in posslices_test:
   #    labslices_test += [ df_t.iloc[myslice].loc[:,"label"].values ]
   #    splslices_test += [
   #       ( df_t.iloc[myslice].loc[:,propdict["featname"]] ).as_matrix() ]


   samplesdf = samplesdf.dropna()
   samples_test_df = samples_test.copy()
#########################
   df_copy = df.copy()
   count = 0
   num_feat_after_vect = collections.defaultdict(int)
   # for feat in featstotrain:
   #     # if feat=="text":
   #     #     if df_copy[feat].dtype==object:
   #         if feat=="text":
   #
   #             count_vect = CountVectorizer(max_features=500, stop_words='english')
   #             # count_vect = CountVectorizer(stop_words='english')
   #             # count_vect = CountVectorizer(stop_words='english')
   #             X_train_counts = count_vect.fit_transform(df_copy[feat])
   #             print "Shape of the matrix of counts:", X_train_counts.shape
   #
   #             tfidf_transformer = TfidfTransformer(use_idf=False)
   #             X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
   #             num_feat_after_vect[feat] = X_train_tfidf.shape[1]
   #             if count==0:
   #                 non_word_feats = X_train_tfidf
   #             else:
   #                non_word_feats = hstack([non_word_feats, X_train_tfidf])
   #         else:
   #             feat_list = df_copy[feat].reshape(len(df_copy),1)
   #             num_feat_after_vect[feat] = feat_list.shape[1]
   #
   #             if count==0:
   #                non_word_feats = coo_matrix(feat_list)
   #             else:
   #                non_word_feats = hstack([non_word_feats, coo_matrix(feat_list)])
   #         count+=1
   # non_word_feats_scr = non_word_feats.tocsr()
   # samples_csr_matrix = non_word_feats_scr[positions_train]
   # count= 0
   #
   # df_t_copy = df_t.copy()
   # count = 0
   # num_feat_after_vect = collections.defaultdict(int)
   # for feat in featstotrain:
   #     # if feat!="text":
   #         if df_t_copy[feat].dtype==object:
   #
   #             count_vect = CountVectorizer(max_features=500, stop_words='english')
   #             # count_vect = CountVectorizer(stop_words='english')
   #             # count_vect = CountVectorizer(stop_words='english')
   #             X_train_counts = count_vect.fit_transform(df_t_copy[feat])
   #             print "Shape of the matrix of counts:", X_train_counts.shape
   #
   #             tfidf_transformer = TfidfTransformer(use_idf=False)
   #             X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
   #             num_feat_after_vect[feat] = X_train_tfidf.shape[1]
   #             if count==0:
   #                 non_word_feats = X_train_tfidf
   #             else:
   #                non_word_feats = hstack([non_word_feats, X_train_tfidf])
   #         else:
   #             feat_list = df_t_copy[feat].reshape(len(df_t_copy),1)
   #             num_feat_after_vect[feat] = feat_list.shape[1]
   #
   #             if count==0:
   #                non_word_feats = coo_matrix(feat_list)
   #             else:
   #                non_word_feats = hstack([non_word_feats, coo_matrix(feat_list)])
   #         count += 1
   # non_word_feats_scr = non_word_feats.tocsr()

   # samples_tsest_csr_matrix = non_word_feats_scr[positions_test]
   # samples_test_csr_matrix = collections.defaultdict(csr_matrix)
   # for myslice in posslices_test:
   #     samples_test_csr_matrix[count] = non_word_feats_scr[myslice]
   #     count +=1


###########################


   def hrtime( epoch ):
      return datetime.datetime.fromtimestamp( int(epoch) ).strftime(
         '%Y-%m-%d %H:%M:%S' )


   # removing nans before computing correlations
   dfc = df.dropna()
   dflongc = dflong.dropna()
   if len(dfc)==0:
      print ("WARNING: using df as dfc, otherwise ValueError below")
      dfc=df


   def apply_cdfgoodtrans( samples_var, samples_ref ):
      samplesdf_var = pd.DataFrame(samples_var)
      samplesdf_ref = pd.DataFrame(samples_ref)
      for col in samplesdf_var:
         samplesdf_var[col] = basics_fake_news_test.transform_values_to_cdf(
            samplesdf_var[col].values, samplesdf_ref[col].values )
      return samplesdf_var.as_matrix()

   if "cdfgoodtrans" in classifier:
      # it has to be done in this order
      samples_test = apply_cdfgoodtrans( samples_test, samples )
      # samples_test = apply_cdfgoodtrans( samples_tsest_csr_matrix.toarray(), samples_tsest_csr_matrix.toarray())
      samples_test = csr_matrix(samples_test)
      # for i in range(len(splslices_test)):
      #    splslices_test[i] = apply_cdfgoodtrans( splslices_test[i], samples )
      samples = apply_cdfgoodtrans( samples, samples )
      # samples = apply_cdfgoodtrans(samples_csr_matrix.toarray(),samples_csr_matrix.toarray())
      samples = csr_matrix(samples)
      # print "After cdfgoodtrans (mean):", samples.mean(axis=0)
      # print "After cdfgoodtrans (min, max):", samples.min(axis=0), samples.max(axis=0)

   if "minmax" in classifier:
      scaler = sklearn.preprocessing.MinMaxScaler()
      # samples = scaler.fit_transform(samples)
      # samples_test = scaler.transform(samples_test)
      samples = scaler.fit_transform(samples_csr_matrix.toarray())
      samples_test = scaler.transform(samples_test_csr_matrix[0].toarray())
      samples_test = csr_matrix(samples_test)
      samples = csr_matrix(samples)
      # for i in range(len(splslices_test)):
      #    splslices_test[i] = scaler.transform(splslices_test[i])
      scaler = sklearn.preprocessing.MinMaxScaler()
      # print "After scaling (mean):", samples.mean(axis=0)
      # print "After scaling (std):", samples.std(axis=0)

   # if "none" in classifier:
   #    print "After scaling (mean):", samples.mean(axis=0)
   #    print "After scaling (std):", samples.std(axis=0)


   # classifier-independent standard feature importance
   sys.stdout.flush()
   # propdict["chi2"] =  sklearn.feature_selection.chi2(samples, labels)[0]
   # propdict["anova"] =  sklearn.feature_selection.f_classif(samples, labels)[0]

   ##### preprocessing
   if "std" in classifier:
      scaler = sklearn.preprocessing.StandardScaler()
      # samples = scaler.fit_transform(samples)
      # samples_test = scaler.transform(samples_test)
      samples = scaler.fit_transform(samples_csr_matrix)
      samples_test = scaler.transform(samples_test_csr_matrix)
      for i in range(len(splslices_test)):
         splslices_test[i] = scaler.transform(splslices_test[i])
      scaler = sklearn.preprocessing.StandardScaler()
      print ("After scaling (mean):", samples.mean(axis=0))
      print ("After scaling (std):", samples.std(axis=0))

   ##### classifier-dependent feature importance
   # ignore ""
   warnings.simplefilter('ignore', UserWarning)
   clfs = dict()
   ests = dict()
   timeprofiler = dict()
   nonparametric = [ "naive_bayes", "lda", "qda", "rfe_linear2" ,"naive_bayes_fit"]

   propsets = {}
   propsets = {
      # "dists": [ "featname", "hellinger", "jsdiv", "totvar" ],
      "corrs": [ "featname", "spearmanr", "spearmanr_long", "pearsonr", "pearsonr_long" ]
      }

   def fit( clfname, clf, samples=samples ):
      print (clfname,)
      starttime = time.time()
      hparams=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
      if "gridsearch" in classifier:
         if clfname not in nonparametric:
            param_grid = [{'C': [0.01, 0.1, 1, 10, 100, 1000]}]
            # param_grid = [{'estimator_C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]
            # clf = sklearn.grid_search.GridSearchCV( clf,param_grid, cv=10, scoring=sklearn.metrics.f1_score )

            # clf = sklearn.grid_search.GridSearchCV( clf, hparams, cv=5, scoring="f1", refit=True)#, score_func=sklearn.metrics.f1_score )
            # clf = GridSearchCV( clf, param_grid, cv=5)
            # clf_tt = sklearn.svm.SVC( kernel='linear')
            # clf = sklearn.grid_search.GridSearchCV( clf, param_grid)
            clf.fit( samples, labels)
            # print(clf.best_score_)
            # scaler = sklearn.preprocessing.StandardScaler().fit(samples.toarray())
            scores = sklearn.model_selection.cross_val_score(clf, samples, labels,cv=10)
            print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
            # est = clf.best_estimator_
            print ("Gridsearch:", clfname, clf.get_params())
         else:
            clf.fit( samples.toarray(), labels)
            est = clf
      else:
         clf.fit( samples.toarray(), labels)
         est = clf
      # clfs[clfname] = scores
      clfs[clfname] = clf
      # ests[clfname] = est
      timeprofiler[clfname] = time.time() - starttime

   print ("Fitting:")

   def fit_RFE( clfname, clf, samples=samples ):
      print (clfname,)
      num_feat = 3
      starttime = time.time()
      hparams=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
      if "gridsearch" in classifier:
         if clfname not in nonparametric:
            param_grid = [
               {'C': [0.01, 0.1, 1, 10, 100, 1000]},
               # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
            ]
            # param_grid = [{'estimator_C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]
            # clf = sklearn.grid_search.GridSearchCV( clf,param_grid, cv=10, scoring=sklearn.metrics.f1_score )

            # clf = sklearn.feature_selection.RFECV(clf, step=1,n_features_to_select=7, cv=10)
            # clf = sklearn.feature_selection.RFE(clf, step=1,n_features_to_select=54)

            clf = sklearn.feature_selection.RFECV(clf, step=1, cv=5)
            clf.fit(samples, labels)
            est = clf

            # clf = sklearn.feature_selection.RFE(clf)
            # param_grid = [{'estimator__C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]
            # clf = sklearn.grid_search.GridSearchCV( clf, param_grid, cv=10)#,scoring="f1", refit=True)#, score_func=sklearn.metrics.f1_score )
            # clf.fit(samples.toarray(), labels)
            clf.fit(samples, labels)
            est = clf
            # print "Gridsearch:", clfname, clf.get_params()
         else:
            # clf = sklearn.feature_selection.RFE(clf)
            clf = sklearn.feature_selection.RFECV(clf, cv=10)
            # clf = sklearn.feature_selection.Ridge(clf)
            # clf.fit(samples.toarray(), labels)
            est = clf
      else:
         # clf = sklearn.feature_selection.RFE(clf ,n_features_to_select=7)
         clf = sklearn.feature_selection.RFE(clf,n_features_to_select=10,cv=10)
         clf.fit( samples, labels)
         est = clf
      clfs[clfname] = clf
      ests[clfname] = est
      timeprofiler[clfname] = time.time() - starttime

   print ("Fitting:")

   if ("logreg" in classifier or "allclfs" in classifier or "mainclfs" in classifier)and \
      ("none" not in classifier ):#or "cdfgoodtrans" in classifier ):
      clfname = "logreg"
      # hparams = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]}
      # fit( clfname, sklearn.linear_model.LogisticRegression() )
      # fit( clfname, sklearn.linear_model.Ridge(alpha=0.1))

      fit( clfname, sklearn.linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0),cv=5) )

      # fit( clfname, sklearn.linear_model.LarsCV() )
      # fit( clfname, sklearn.linear_model.ElasticNet() )
      # print("support : " + str(ests[clfname].support_))
      # print("ranking : " + str(ests[clfname].ranking_))
      # print("scores : " + str(ests[clfname].estimator_.coef_))
      # print("importance : " + str(ests[clfname].coef_))

      # fit_RFE( clfname, sklearn.linear_model.LogisticRegression() )
      # fit( clfname, sklearn.linear_model.Ridge() )
      # fit( clfname, sklearn.linear_model.RidgeCV(alphas=[0.001,0.01,0.1,1,10,100]) )

      # fit( clfname, m.LogisticIT())

      # fit( clfname, m.OrdinalRidge())

      # fit( clfname, m.LogisticAT())
      # yhat = ests[clfname].predict(samples)
      # from sklearn.metrics import accuracy_score

      # print('acc : ' + str(accuracy_score(labels, yhat)))
      # print('corr : ' + str(np.corrcoef(labels, yhat)))
      # print('corr : ' + str(scipy.stats.spearmanr(labels, yhat)))
      # propdict[clfname] =  ests[clfname].coef_[0]

      # print("support : " + str(ests[clfname].support_))
      # print("ranking : " + str(ests[clfname].ranking_))
      # print("scores : " + str(ests[clfname].coef_))
      # print(ests[clfname].coef_[0])
      propsets["logreg"] = [ "featname", clfname ]
      # if "mainclfs" in classifier:
      #    propsets["mainclfs"] += [ clfname ]
      #    propsets["extmainclfs"] += [ clfname ]
      # print("scores : " + str(ests[clfname].coef_))
      # print("support : " + str(ests[clfname].support_))
      # print("ranking : " + str(ests[clfname].ranking_))
      # print("scores : " + str(ests[clfname].estimator_.coef_))

      # yhat = ests[clfname].predict(samples)
      # from sklearn.metrics import accuracy_score
      #
      # print('acc : ' + str(accuracy_score(labels, yhat)))
      # print('corr : ' + str(np.corrcoef(labels, yhat)))
      # print('corr : ' + str(scipy.stats.spearmanr(labels, yhat)))
      # propdict[clfname] =  ests[clfname].coef_[0]
      # # crashes on gridsearch and other places
      # clfname = "rndlogreg"
      # hparams = {'C': [1, 10, 100, 1000]}
      # fit( clfname, sklearn.linear_model.RandomizedLogisticRegression() )
      # propdict[clfname] =  ests[clfname].scores_


   if "svm-lin" in classifier or "allclfs" in classifier:
      clfname = "svm_linear1"
      hparams = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]}
      # fit( clfname, sklearn.svm.LinearSVC(class_weight='balanced') )

      # fit( clfname, sklearn.svm.LinearSVC() )
      fit( clfname, sklearn.svm.LinearSVC() )
      # fit( clfname, sklearn.svm.SVC(kernel='linear') )

      # clf_tt = sklearn.svm.SVC( kernel='linear')
      # clf = sklearn.grid_search.GridSearchCV( clf, param_grid)
      # clf_tt.fit(samples, labels)



      # fit_RFE( clfname, sklearn.svm.LinearSVC() )
      # propdict[clfname] =  ests[clfname].coef_[0]
      # propsets["svm-lin"] = [ "featname", clfname ]

      # fit_RFE( clfname, sklearn.svm.LinearSVC(class_weight='balanced') )
      # print("support : " + str(ests[clfname].support_))
      # print("ranking : " + str(ests[clfname].ranking_))
      # print("scores : " + str(ests[clfname].estimator_.coef_))
      # print("Predicting feats: " + str(propdict["featname"]))

      # index_l = range(len(ests[clfname].support_))
      #
      # df_s = pd.DataFrame({'supp': Series(ests[clfname].support_, index=index_l),
      #                      'rank': Series(ests[clfname].ranking_, index=index_l),
      #                      'scores': Series(ests[clfname].estimator_.coef_[0], index=index_l),
      #                      'feat': Series(propdict["featname"], index=index_l)})


      # index_l = range(len(clf_tt.coef_.toarray()[0]))
      #
      # df_s = pd.DataFrame({#'supp': Series(clf_tt.support_, index=index_l),
      #                      # 'rank': Series(clf_tt.ranking_, index=index_l),
      #                      'scores': Series(clf_tt.coef_.toarray()[0], index=index_l),
      #                      'feat': Series(propdict["featname"], index=index_l)})
      #
      #
      # df_s = df_s.sort_values('scores',ascending=False)
      # # print("support : " + str(list(df_s['supp'])))
      # # print("ranking : " + str(list(df_s['rank'])))
      #
      # for ind in df_s.index.tolist():
      #
      #    print("scores : " + str(df_s['scores'][ind]))
      #    print("Predicting feats: " + str(df_s['feat'][ind]))



      # print('acc : ' + str(accuracy_score(labels, yhat)))
      # print('corr : ' + str(np.corrcoef(labels, yhat)))
      # print('corr : ' + str(scipy.stats.spearmanr(labels, yhat)))
      # propdict[clfname] =  ests[clfname].coef_[0]

      # print("support : " + str(ests[clfname].support_))
      # print("ranking : " + str(ests[clfname].ranking_))
      # print("scores : " + str(ests[clfname].coef_))
      # print(ests[clfname].coef_[0])



   if "naive_bayes" in classifier:
      clfname = "naive_bayes"
      # fit_RFE(clfname, sklearn.naive_bayes.MultinomialNB())
      fit(clfname, sklearn.naive_bayes.MultinomialNB())
      # from sklearn.model_selection import KFold, cross_val_score
      # k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
      # clf = sklearn.naive_bayes.GaussianNB()
      # print cross_val_score(clf, samples.toarray(), labels, cv=5, n_jobs=1)


      # fit_RFE( clfname, clf )
      # print("support : " + str(ests[clfname].support_))
      # print("ranking : " + str(ests[clfname].ranking_))
      # print("scores : " + str(ests[clfname].estimator_.coef_))
      # print("scores : " + str(ests[clfname].coef_))

      # yhat = ests[clfname].predict(samples)
      # from sklearn.metrics import accuracy_score
      #
      # print('acc : ' +  str(accuracy_score(labels, yhat)))
      # print('corr : ' + str(np.corrcoef(labels, yhat)))
      # print('corr : ' + str(scipy.stats.spearmanr(labels, yhat)))


   def print_f1_dict(f1_dict):
      for key in sorted( f1_dict.keys(), key=lambda x: float(x.split(", ")[0][1:]) ):
         print ("%.2f" % f1_dict[key],)
      for key in sorted( f1_dict.keys(), key=lambda x: float(x.split(", ")[0][1:]) ):
         print (key,)
      return ""

   # gridsearch and cross-validation
   print ("Testing:")
   printlist = []
   useravgf1s = dict()
   memeavgf1s = dict()
   pubtable = dict()
   train_liklihood = []
   test_liklihood = []
   acc_t_f = {}
   for clfname in clfs:
      # if clfname!="naive_bayes": continue
      printitem = [ clfname ]
      starttime = time.time()
      samples_tmp = samples_test
      print ("Testing", clfname)
      coef_dict = {}
      feat_dict = {}
      ind_cc=0
      # for i_c in range(len(clfs[clfname].coef_)):
      #     coef_dict[ind_cc] = clfs[clfname].coef_[i_c]
      #     feat_dict[ind_cc] = featstotrain[i_c]
      #     ind_cc+=1

      # sort_coef_list = sorted(coef_dict, key=coef_dict.get, reverse=False)

      # for i_c in sort_coef_list:
      #   print(str(feat_dict[i_c]) + ' : ' + str(coef_dict[i_c]))

      # featstotrain

      #
      # samples_test = df_gl[''][]
      # labels_test = df_gl['label']

      labels_pred = clfs[clfname].predict( samples_test)
      from sklearn.metrics import accuracy_score
      return labels_pred, labels_test
      print('acc test : ' +  str(accuracy_score(labels_test, labels_pred)))

      # print('corr test : ' + str(np.corrcoef(labels_test, labels_pred)))
      # print('corr test : ' + str(scipy.stats.spearmanr(labels_test, labels_pred)))
      # labels_test=[-1,1]


      for label_tmp in set(labels_test):
          # print('label : ' + str(label_tmp))
          total_c=0; correct_c=0
          for lable_i in range(len(labels_test)):
              if labels_test[lable_i]==label_tmp:
                  total_c+=1
                  if labels_pred[lable_i]==label_tmp:
                      correct_c+=1

          print('accuracy : ' +str(label_tmp) + ' : ' + str(float(correct_c)/total_c))
          acc_t_f[label_tmp] = float(correct_c)/total_c


      # acc_gt = collections.defaultdict(dict)
      # for tt in set(labels_test):
      #    acc_gt[tt] = collections.defaultdict(int)
      # for label_tmp in set(labels_test):
      #    total_c = 0;
      #    correct_c = 0
      #    for lable_i in range(len(labels_pred)):
      #       if labels_test[lable_i]==label_tmp:
      #          acc_gt[label_tmp][labels_pred[lable_i]]+=1
         # print('accuracy : ' + str(float(correct_c) / total_c))

      outp = {}
      outp_var = {}
      # news_cat_list = ['pants-fire', 'false', 'mostly_false', 'half-true', 'mostly-true', 'true']
      # news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
      dataset = 'snopes'
      data_n = 'sp'
      if dataset == 'snopes' or dataset == 'snopes_nonpol':
         # col_l = ['b', 'g', 'c', 'y', 'r']
         col_l = ['red', 'orange', 'gray', 'lime', 'green']
         news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
         news_cat_list_v = [-2, -1, 0, 1, 2]
         # news_cat_list_v = [1, 2, 3, 4, 5]

         news_cat_list_n = ['FALSE', 'MOSTLY\nFALSE', 'MIXTURE', 'MOSTLY\nTRUE', 'TRUE']
      if dataset == 'politifact':
         # col_l = ['grey','b', 'g', 'c', 'y', 'r']
         col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']

         news_cat_list_n = ['PANTS ON\nFIRE', 'FALSE', 'MOSTLY\nFALSE', 'HALF\nTRUE', 'MOSTLY\nTRUE', 'TRUE']
         news_cat_list_t_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
         news_cat_list_v = [-2, -2, -1, 0, 1, 2]

      if dataset == 'mia':
         # col_l = ['b', 'r']
         col_l = ['red', 'green']
         news_cat_list_n = ['RUMORS', 'NON RUMORS']
         news_cat_list_t_f = ['rumor', 'non-rumor']
         news_cat_list_v = [-1, 1]




      #
      # count = 0
      # Y = [0] * len(news_cat_list_v)
      # # Y1 = [0] * len(thr_list)
      # mplpl.rcParams['figure.figsize'] = 6.8, 5
      # mplpl.rc('xtick', labelsize='large')
      # mplpl.rc('ytick', labelsize='large')
      # mplpl.rc('legend', fontsize='small')
      # cat_num_st = 0
      #
      # width = 0.05
      # for cat_v in news_cat_list_v:
      #    count += 1
      #    outp[cat_v] = []
      #    for ii in news_cat_list_v:
      #       outp[cat_v].append(acc_gt[cat_v][ii])
      #    # for i in range(len(gt_acc[cat_v])):
      #    #     outp[i].append(gt_acc[cat_v][i])
      #    if dataset == 'snopes' or dataset == 'snopes_nonpol':
      #       mplpl.bar([0.09, 0.18, 0.28, 0.38, 0.48], outp[cat_v], width, bottom=np.array(Y), color=col_l[count - 1],
      #                 label=news_cat_list_n[count - 1])
      #    elif dataset == 'politifact':
      #       mplpl.bar([0.09, 0.18, 0.28, 0.38, 0.48], outp[cat_v], width, bottom=np.array(Y), color=col_l[count - 1],
      #                 label=news_cat_list_n[count - 1])
      #    Y = np.array(Y) + np.array(outp[cat_v])
      #
      # mplpl.xlim([0.08, 0.58])
      # mplpl.ylim([0, 60])
      # mplpl.ylabel('Composition of labeled news stories', fontsize=14, fontweight='bold')
      # # mplpl.xlabel('Top k news stories reported by negative PTL', fontsize=13.8,fontweight = 'bold')
      # mplpl.xlabel('Users\' perception', fontsize=14,
      #              fontweight='bold')
      # # mplpl.xlabel('Top k news stories ranked by NAPB', fontsize=18)
      #
      # mplpl.legend(loc="upper right", ncol=3, fontsize='small')
      #
      # mplpl.subplots_adjust(bottom=0.2)
      #
      # mplpl.subplots_adjust(left=0.18)
      # mplpl.grid()
      # # mplpl.title(data_name, fontsize='x-large')
      # labels = news_cat_list_n
      # x = [0.1, 0.2, 0.3, 0.4, 0.5]
      # mplpl.xticks(x, labels)
      # pp = remotedir + '/fig/fig_exp1/news_based/initial/' + data_n + '_vote_composition_gt'
      # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_vote_composition_gt'
      # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_napb_composition_gt'
      # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_gt_accuracy_reg_weight'
      pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_gt_accuracy_ordinalreg_weight'
      # mplpl.savefig(pp + '.pdf', format='pdf')
      # mplpl.savefig(pp + '.png', format='png')
      # acc=clfs[clfname]
      # acc_std=clfs[clfname].std()
      # labels_pred = clfs[clfname].predict( samples_csr_matrix.toarray())
      # train_liklihood = clfs[clfname].predict_proba(samples_test)
      # test_liklihood = clfs[clfname].predict_proba(samples_test)
      # acc = accuracy_score(labels_test, labels_pred)
      acc = accuracy_score(labels_test, labels_pred)
      # acc = clfs[clfname].best_score_
      corr_pearson = 0
      corr_spearman = 0
      # corr_pearson = np.corrcoef(labels_test, labels_pred)
      # corr_spearman = scipy.stats.spearmanr(labels_test, labels_pred)

      print(corr_pearson)
   # return train_liklihood, test_liklihood, labels_pred
   return acc, acc_t_f, corr_pearson,corr_spearman


def sort_write_data(df1, df2,fdf1, fdf2,fdf3, img_inf1, img_inf2 ):
   img_list=[]
   for i in [1,2,3]:
      for j in range(52):
         img_list.append(str(i) + '_zoomout_fake_'+str(j))

      # news_id = collections.defaultdict()
      # news_text = collections.defaultdict()
      # fdf_s = pd.read_csv('Workers-perception-' + str(i) + '.csv', sep="\t")
      # for ind in fdf_s.index.tolist():
      #    news_id[ind] = fdf_s['text_id'][ind]
      #    news_text[ind] = fdf_s['text'][ind]


   df1.loc[:,'img'] = df1['news_id']*0.0
   df1['img'] = img_list[:]

   # sort_value = 'tpb'
   sort_value = 'var'
   for ind in df1.index.tolist():
      df1['idpb'][ind] = np.abs(df1['idpb'][ind])
   # sort_value = 'rmse'
   # sort_value = 'accuracy_0_1'
   # sort_value = 'acc_0_1_2'

   df_sort = df1.sort_values(sort_value, ascending=0)
   # outF = open('outputFile/sort_rmse.text', 'w')
   # outF = open('outputFile/sort_'+sort_value+'_dem.text', 'w')
   outF = open('outputFile/shift_sort_'+sort_value+'.text', 'w')
   # outF.write('|| news id || news text|| Ground Truth || Disputability || TPB || Idelogical Disp '
   #            '||Distribution 1 cell||\n')
   outF.write('|| news id || news text|| Ground Truth || Disputability || TPB || Idelogical Disp ||Distribution 1 cell||'
              ' Shift distribution ||  Shift CDF || Democrats and publicans shift distribution || posetive and negative shift distribution|| '
              'shift dist. of different group who assess 1, 2,3, 4, or 5 seprately|| CDF || scatter plot||\n')
   for ind in df_sort.index.tolist():
      outF.write('||' + str(df_sort['news_id'][ind]) + '||' + str(df_sort['news_text'][ind]) + '||' +
                 str(df_sort['gt'][ind]) + '||' + str(df_sort['var'][ind]) + '||' + str(df_sort['tpb'][ind])+'||'+
                 str(df_sort['idpb'][ind]) + '||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/nfm_pre_'+str(df_sort['news_id'][ind])+'}} + ||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/shift_nfm_pre_'+str(df_sort['news_id'][ind])+'}} + ||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/shift_cdf_'+str(df_sort['news_id'][ind])+'}} + ||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/shift_dem_rep_nfm_pre_'+str(df_sort['news_id'][ind])+'}} + ||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/shift_pos_neg_nfm_pre_'+str(df_sort['news_id'][ind])+'}} + ||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/shift_5_dist_'+str(df_sort['news_id'][ind])+'}} + ||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/shift_5_cdf_'+str(df_sort['news_id'][ind])+'}} + ||' +
                 '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/fig_real_predication_nmf/shift_scatter_pre_'+str(df_sort['news_id'][ind])+'}} ||\n')

   idpb_sort=df1.sort_values('idpb', ascending=0)

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













def main():

   def ff(i):
      return i

   for i in range (2,12):
      print(ff(i%10),ff((i-1)%10))

   exit()

   tr_test=10
   df1 = pd.read_csv('nmf_news_'+str(tr_test)+'.csv', sep="\t")
   # fdf_s = pd.read_csv('nmf_news_dem_'+str(tr_test)+'.csv', sep="\t")
   img_inf1 = 'fig_real_predication_fn/'
   img_inf2 = 'fig_real_predication_nmf/'

   fdf_1 = pd.read_csv('nmf_prediction_news_total_1.csv', sep="\t")
   fdf_2 = pd.read_csv('nmf_prediction_news_total_2.csv', sep="\t")
   fdf_3 = pd.read_csv('nmf_prediction_news_total_3.csv', sep="\t")



   frames = [fdf_1, fdf_2, fdf_3]
   df2 = pd.concat(frames, ignore_index=True, sort=False)

   # df2 = fdf_3.copy()
   # outpp =  df2['nmf_p_label_0'].plot.kde(bw_method=0.5)
   # outpp =  df2['nmf_p_label_0'].plot(kind='hist')#kde(bw_method=0.5)
   # outpp =  df2['nmf_p_label_0'].hist(bins=20)#kde(bw_method=0.5)

   # df2.loc[:,'shift_var'] = df2['worker_id']*0.0
   # df2.loc[:,'err_var'] = df2['worker_id']*0.0
   # df2.loc[:,'err_mean'] = df2['worker_id']*0.0
   # # df2['img'] = img_list[:]
   #
   # shift_l = collections.defaultdict(list)
   # err_l = collections.defaultdict(list)
   # for ind in df2.index.tolist():
   #    # df2['idpb'][ind] = np.abs(df2['idpb'][ind])
   #    w_id = df2['worker_id'][ind]
   #    for i in range(52):
   #       # if df2['real_label_'+ str(i)][ind] != 3:
   #       shift_l[w_id].append(df2['s_label_'+ str(i)][ind] - df2['nmf_p_label_' + str(i)][ind])
   #       err_l[w_id].append(np.abs(df2['s_label_'+ str(i)][ind] - df2['real_label_' + str(i)][ind]))
   #
   #    df2['shift_var'][ind] = np.var(shift_l[w_id])
   #    df2['err_var'][ind] = np.var(err_l[w_id])
   #    df2['err_mean'][ind] = np.mean(err_l[w_id])
   # # sort_write_data(df1,df2,fdf_1, fdf_2,fdf_3, img_inf1, img_inf2)
   #
   # print(np.mean(df2['shift_var'][:]))
   #
   # bad_users_l=[]
   # good_users_l = []
   # for ind in df2.index.tolist():
   #    w_id = df2['worker_id'][ind]
   #    if df2['shift_var'][ind]>=2:
   #       bad_users_l.append(w_id)
   #    else:
   #       good_users_l.append(w_id)
   #
   #
   # print( ('corr : ' + str(np.corrcoef(df2['shift_var'][:], df2['err_var'][:]))))
   # print(scipy.stats.spearmanr(df2['shift_var'][:], df2['err_var'][:]))
   #
   # print( ('corr : ' + str(np.corrcoef(df2['shift_var'][:], df2['err_mean'][:]))))
   # print(scipy.stats.spearmanr(df2['shift_var'][:], df2['err_mean'][:]))
   #
   # # print(np.mean(df2['shift_var']))
   # # print(np.max(df2['shift_var']))
   # # print(np.min(df2['shift_var']))
   # # print(df2['shift_var'])
   #
   # final_acc = 0
   # total_n = 0
   # for ff in [fdf_1, fdf_2, fdf_3]:
   # # for ff in [fdf_2]:
   #    ff = ff[ff['worker_id'].isin(good_users_l)]
   #    for nn in range(52):
   #       acc=0
   #       t_acc=0
   #       if ff['real_label_' + str(nn)][ff.index.tolist()[0]] == 3:
   #          continue
   #       total_n += 1
   #
   #       assess_sum = 0
   #       cc=0
   #
   #       for ind in ff.index.tolist():
   #          # w_id = ff['worker_id'][ind]
   #
   #          assess_sum += ff['s_label_' + str(nn)][ind]
   #          cc+=1
   #          if assess_sum/float(cc)>3 and ff['real_label_' + str(nn)][ind]>3:
   #             acc+=1
   #          elif assess_sum/float(cc) < 3 and ff['real_label_' + str(nn)][ind] < 3:
   #             acc += 1
   #          # else:
   #          t_acc += 1
   #       try:
   #          if float(acc)/t_acc > 0.5:
   #             final_acc+=1
   #       except:
   #          continue
   #
   # print(final_acc)
   # print(total_n)
   # print(final_acc/float(total_n))
   #
   #
   #
   # final_acc = 0
   # total_n = 0
   # for ff in [fdf_1, fdf_2, fdf_3]:
   #    ff = ff[ff['worker_id'].isin(bad_users_l)]
   #
   #    # for ff in [fdf_2]:
   #    # if len(ff)<2:
   #    #    continue
   #    for nn in range(52):
   #       acc=0
   #       t_acc=0
   #       if ff['real_label_' + str(nn)][ff.index.tolist()[0]] == 3:
   #          continue
   #       total_n += 1
   #
   #       for ind in ff.index.tolist():
   #
   #          if ff['s_label_' + str(nn)][ind]>3 and ff['real_label_' + str(nn)][ind]>3:
   #             acc+=1
   #          elif ff['s_label_' + str(nn)][ind] < 3 and ff['real_label_' + str(nn)][ind] < 3:
   #             acc += 1
   #          # else:
   #          t_acc += 1
   #       try:
   #          if float(acc)/t_acc > 0.5:
   #             final_acc+=1
   #       except:
   #          continue
   #
   # print(final_acc)
   # print(total_n)
   # print(final_acc/float(total_n))
   #

   # exit()





   global df_gl
   # A = coo_matrix([[1, 2], [3, 4]])
   # B = coo_matrix([[5], [6]])
   # hstack([A,B]).toarray()
   # exit()
   # fdf = basics.load_data( "~/Desktop/icwsm_svn/code/recsys-for-posting/data/extend_uidposts.csv", nrows=100)
   hostname = socket.gethostname()
   parser = argparse.ArgumentParser(prog='PROG')
   parser.add_argument( '-t', default="all",
      help='taskname' )
   parser.add_argument( '-pre', nargs='+', default=None,
      help='what preprocessing to apply' )
   parser.add_argument( '-featsets', nargs='+', default=None,
      help='what feature sets to use' )
   parser.add_argument( '-transforms', nargs='+', default=None,
      help='what transforms to use' )
   parser.add_argument( '-clf', nargs='+', type=str, default=["mainclfs"],#"svm-nonlin"],
      help='apply all classifiers in one thread or in seperate threads' )
   parser.add_argument( '-nr', type=int, default=None,
      help='how many rows to load' )
   parser.add_argument( '-ns', type=int, default=None,
      help='how many samples to use' )
   parser.add_argument( '-nu', default=None,
      help='how many users (corresponding input file needed)' )
   parser.add_argument( '--timeout', type=int, default=60,
      help='timeout per process (the number of seconds)' )
   parser.add_argument( '-gs', action='store_true',
      help='run grid search' )
   parser.add_argument( '-mp', action='store_true',
      help='use multiprocessing?' )
   parser.add_argument( '-meme', default="hashtag",
      help='the type of memes to analyze' )
   parser.add_argument( '-seeds', default=1, help='run for how many seeds?' )
   parser.add_argument( '-topicvec', default="joints-lda-avg",
      help='the name of topical vectors' )
   args = parser.parse_args()
   # print args
   # return

   tm_featnm = "created-at"

   featsets = [
      "all_feats"
      # "allfeats"
      # "all_disputability_income_disputability_political_view_disputability_age_disputability" #-->tpb:%84, gtl:57%    85%train : gtl:58%
      # "all_disputability_income_disputability_degree_disputability" #-->tpb:%79, gtl:57%        85%train : gtl:67%
      # "all_disputability_income_disputability_political_view_disputability" #-->tpb:86%, gtl:56   85%train : gtl:61%
      # "income_disputability_political_view_disputability" #-->tpb:82%, gtl:55%
      # "all_disputability_income_disputability_political_view_disputability_employment_disputability_degree_disputability_age_disputability" #-->tpb:74%, gtl-->56%
      # "income_disputability_political_view_disputability_employment_disputability" #-->tpb:74% gtl:64
      # "all_disputability_income_disputability_political_view_disputability_employment_disputability" #-->tpb:78%, gtl:63%
      # "all_disputability_income_disputability_political_view_disputability_employment_disputability_gender"  #-->tpb:80%, gtl:64%
      # "all_disputability_income_disputability_political_view_disputability_employment_disputability_gender_pt_count" #-->76%, gtl:64%
      # "all_disputability_political_view_disputability_degree_disputability_gender_disputability" #-->tpb:60%, gtl:54%
      # "all_disputability_income_disputability_political_view_disputability_degree_disputability_age_disputability" #-->tpb:76%, gtl:55%
      # "all_disputability_income_disputability_political_view_disputability_employment_disputability_gender_disputability_age_disputability_degree_disputability_race_disputability_marital_status_disputability" #-->tpb:73%, gtl:56%
      # "all_disputability_income_disputability_political_view_disputability_employment_disputability_gender_disputability_age_disputability_degree_disputability_race_disputability_marital_status_disputability_pt_count"#71%,gtl:56%
      # "all_disputability"#tpb:75%, gtl:58%, gtl:61%
      # "all_disputability"
      # "pt_count"
      # "gender_disputability" #-->tpb:70%, gtl:57%
      # "leaning_disputability" #-->tpb:43%, gtl:36%
      # "age_disputability" #-->tpb:80%, gtl:55%
      # "degree_disputability" #-->tpb:64%, gtl:60%
      # "employment_disputability" #-->tpb:72%, gtl:56%
      # "income_disputability"#-->tpb:78%, gtl:64%
      # "political_view_disputability" #-->tpb:71%,gtl:53%
      # "race_disputability"
      # "marital_status_disputability"#-->tpb:69%, gtl:55%
      # "political_view, income"
      # 'age', 'degree', 'employment', 'gender', 'income', 'marital_status', 'political_view', 'race', 'worker_id',
      # 's_label', 'real_label'
      # "worker_pt"
      # "worker_leaning"
      # "pt_count"
      # "worker_leaning_pt_count"
      # "worker_pt_count"



      # "political_view"
       ]
   preprocessings = [ "std", "minmax", "none" ]
   nusers = "all"
   transforms = [ #"notrans",
      # "cdftrans", "cdfalltrans", "cdflatetrans",
      # "cdflatealltrans", "cdflateaddtrans",
      "cdfgoodtrans",
       # "minmax"
   ]

   wdpath, remotedir = get_dirs()


   localdir, remotedir = get_dirs()
   if is_local_machine():
      muzzled = True
      rawdir = remotedir + "/crawled-tw/"
      # rawdir = remotedir+"crawled-tw/"
   else:
      muzzled = False
      # muzzled = True
      rawdir = "/NS/twitter_archive1/work/pms/crawled-tw/"









   if is_local_machine():
      nrows = 1e5
      nsamples = int(1e4)
      args.gs = True
   else:
      nrows = None
      nsamples = int(1e5)
      nsamples = int(2e4)

   meme = args.meme
   if args.nr!=None: nrows = args.nr
   if args.ns!=None: nsamples = args.ns
   if args.nu!=None: nusers = args.nu
   if args.pre!=None: preprocessings = args.pre
   if args.featsets!=None: featsets = args.featsets
   if args.transforms!=None: transforms = args.transforms
   if args.topicvec!=None: topicvec = args.topicvec
   if args.clf=="separate": classifiers = [ "trees", "logreg", "svm-lin",
      "svm-nonlin", "svm-sigm", "other", "sgd", "rfe" ]
   else:
      classifiers = args.clf


   methods = []
   for preprocessing in preprocessings:
      for classifier in classifiers:
         for transform in transforms:
            if args.gs: methods += \
               [transform+"-"+preprocessing+"-"+classifier+"-gridsearch"]
            else: methods += \
               [transform+"-"+preprocessing+"-"+classifier]

   if "mainclfs" in classifiers:
      print ("Using main methods!")
      methods = [

         # "cdfgoodtrans-mainclfs-stest-gridsearch",########
         # "cdfgoodtrans-mainclfs-stest-naive_bayes",########
         # "cdfgoodtrans-logreg-gridsearch-stest",########
         "cdfgoodtrans-svm-lin-gridsearch-stest",########
         #       # "minmax-none-svm-lin-gridsearch-stest-rndimp",

      ]

   # if "svm-lin" in classifiers:
   #    print "Using main methods!"
   #    methods = [
   #       # "notrans-std-mainclfs-gridsearch-stest-rndimp",
   #       # "cdfgoodtrans-none-svm-lin-gridsearch-stest-rndimp",
   #       # "cdfgoodtrans-none-gridsearch-stest-rndimp",
   #       "cdfgoodtrans-none-stest-rndimp",
   #       # "minmax-none-svm-lin-gridsearch-stest-rndimp",
   #       # "notrans-std-mainclfs-gridsearch-mtest",
   #       # "cdfgoodtrans-none-mainclfs-gridsearch-mtest",
   #       # "notrans-minmax-mainclfs-gridsearch"
   #    ]


   dirs={ "png": os.path.expanduser(wdpath+"figs_similarity/"),
          "eps": os.path.expanduser(wdpath+"figs_similarity/"),
          "pdf": os.path.expanduser(wdpath+"figs_similarity/"),
          "data": os.path.expanduser(wdpath+"out_similarity/"),
          "include": os.path.expanduser(wdpath+"draft_similarity/include/"),}
          # "stdout": os.path.expanduser(wdpath+"sp_nonpol_binary/") }
          # "stdout": os.path.expanduser(wdpath+"sp_binary/") }
          # "stdout": os.path.expanduser(wdpath+"sp_nonpol/") }
          # "stdout": os.path.expanduser(wdpath+"sp/") }
          # "stdout": os.path.expanduser(wdpath+"sp_ssi/") }
          # "stdout": os.path.expanduser(wdpath+"pf/") }
          # "stdout": os.path.expanduser(wdpath+"mia/") }
          # "stdout": os.path.expanduser(wdpath+"pf_binary/") }
          # "stdout": os.path.expanduser(wdpath+"mia_binary/") }
   for mydir in dirs.values():
      if not os.path.exists(mydir): os.makedirs(mydir)

   manager = multiprocessing.Manager()
   jobs = []
   jobnames = []

   # wdpath = '/NS/twitter-7/work/Reza/fake_news/data/preprocess/new_data_set/'
   # wdpath = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
   # wdpath_test = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'

   wdpath = ''
   wdpath_test = ''

   pattern = "fake_news_features_sp_ssi.csv"
   pattern_test = "fake_news_features_sp_ssi.csv"
   # pattern = "fake_news_features_sp.csv"
   # pattern_test = "fake_news_features_sp.csv"


   # pattern_3 = "fake_news_features_sp_ssi_3.csv"

   # pattern_1 = "fake_news_features_sp_1.csv"
   # pattern_2 = "fake_news_features_sp_2.csv"
   # pattern_3 = "fake_news_features_sp_3.csv"
   # pattern_1 = "fake_news_features_sp_ssi_1.csv"
   # pattern_2 = "fake_news_features_sp_ssi_2.csv"
   # pattern_3 = "fake_news_features_sp_ssi_3.csv"


   pattern_1 = "fake_news_features_snopes_gt_1.csv"
   pattern_2 = "fake_news_features_snopes_gt_2.csv"
   pattern_3 = "fake_news_features_snopes_gt_3.csv"



   # pattern_1 = "user_selected_fake_news_features_snopes_gt_1_political_view_2.csv"
   # pattern_2 = "user_selected_fake_news_features_snopes_gt_2_political_view_2.csv"
   # pattern_3 = "user_selected_fake_news_features_snopes_gt_3_political_view_2.csv"
   # pattern_2 = "fake_news_features_snopes_gt_2.csv"
   # pattern_3 = "fake_news_features_snopes_gt_3.csv"


   # pattern = "fake_news_features_pf.csv"
   pattern_test = "fake_news_features_snopes_gt_3.csv"

   # pattern = "fake_news_features_mia.csv"
   # pattern_test = "fake_news_features_mia.csv"

   # pattern = "fake_news_features_sp_nonpol.csv"
   # pattern_test = "fake_news_features_sp_nonpol.csv"

   print ("Pattern:", pattern)
   # inputpaths = glob.glob(wdpath+pattern)
   inputpaths = [wdpath+pattern]

   # pre= [1.0,  4.32, 1.11, 1.68, 2.56, 4.5,  1.0,  1.0,  5.0,  1.54, 2.8,  2.06, 3.12, 1.0, 1.0,  1.0,  1.0,  1.0,  1.54, 3.64, 1.0,  3.12, 4.82, 1.3,  4.0,  2.1,  1.0,  1.0,  1.0,  1.36, 1.0,  3.24, 4.62, 4.37, 1.0,  1.0,  1.0,  4.6,  1.0,  5.0,  4.78, 1.0,  1.81, 5.0, 3.85, 5.0,  1.0,  1.42, 5.0,  4.28, 5.0, 1.0]
   # test=[1.57, 3.61, 2.21, 2.86, 2.92, 3.12, 1.88, 1.58, 4.27, 2.45, 2.81, 2.7,  3.0,  2.0, 2.19, 1.92, 1.57, 1.84, 2.66, 3.25, 2.59, 3.21, 3.57, 2.33, 3.54, 2.72, 1.59, 1.72, 1.98, 2.4,  2.29, 3.01, 3.19, 3.48, 2.52, 2.59, 2.18, 3.42, 2.46, 4.16, 3.43, 2.44, 2.85, 4.2, 3.24, 3.88, 2.13, 2.77, 3.67, 3.3,  4.7, 1.3]
   # gt=  [2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  4.0,  4.0,  4.0,  4.0, 4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  5.0, 1.0]



   acc_p=0
   acc_n=0
   # for i in range(len(pre)):
   #    if (pre[i]+test[i])/2<3 and gt[i]<3:
   #       acc_p+=1
   #    if (pre[i]+test[i])/2>3 and gt[i]>3:
   #       acc_n+=1
   # print(acc_p)
   # print(acc_n)
   # exit()
   primes = [ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
      61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137 ]
   seeds = primes[ 0:int(args.seeds) ]

   print ("Files:", inputpaths)
   print ("Featsets:", featsets)
   print ("Methods:", methods)
   for inputpath in inputpaths:
      inputpathbase = "_".join( inputpath.split("_")[:-1] )
      inputfilename = os.path.splitext(os.path.basename(inputpath))[0]
      inputfilenamebase = "_".join( inputfilename.split("_")[:-1] )



      # fdf = pd.read_csv(wdpath+pattern, sep="\t")
      fdf_1 = pd.read_csv(wdpath+pattern_1, sep="\t")
      fdf_2 = pd.read_csv(wdpath+pattern_2, sep="\t")
      fdf_3 = pd.read_csv(wdpath+pattern_3, sep="\t")

      frames = [fdf_1, fdf_2, fdf_3]
      # frames = [fdf_2, fdf_3]
      # frames = [fdf_1]
      fdf = pd.concat(frames,ignore_index=True, sort=False)
      fdf = fdf.copy()
      fdf_test = fdf_3.copy()
      # fdf_test = pd.read_csv(wdpath_test+pattern_test, sep="\t")



      if ".rar" in inputpath: lzmaflag=True
      else: lzmaflag=False
      # fdf = basics.load_data( inputpath, nrows=nrows )
      # fdf = pd.read_csv(inputpath, sep="\t")
      # fdf_test = pd.read_csv(wdpath_test+pattern_test, sep="\t")
      # fdf_test = pd.read_csv(inputpath, sep="\t")
      print ("Loaded",len(fdf),"sample sets.")

      #grouping tweets by users id
      df_gl = fdf.copy()
      fdf_dic = {}
      fdf_c = fdf.copy()

      output_extra_name = '_all-media'


      # dependentvar = 'above_fake_rep_med'
      # dependentvar = 'consensus_value'
      dependentvar = 'label'
      dependentvar = 'tpb'




      fdf_test = fdf.copy()

      range_l = [0.99]

      #
      fdf.loc[:,'label_tpb'] = fdf['tweet_id']*0.0
      tr_tpb_med = np.median(list(fdf['tpb']))

      tmp_df = fdf.sort_values('tpb',ascending=False)
      for index in tmp_df.index.tolist():
          if tmp_df['tpb'][index]>tr_tpb_med:
                fdf.loc[index,'label_tpb'] = 1
          else:
                fdf.loc[index, 'label_tpb'] = 0



      fdf_test.loc[:,'label_tpb'] = fdf_test['tweet_id']*0.0
      tr_tpb_med = np.median(list(fdf_test['tpb']))

      tmp_df = fdf_test.sort_values('tpb',ascending=False)
      for index in tmp_df.index.tolist():
          if tmp_df['tpb'][index]>tr_tpb_med:
                fdf_test.loc[index, 'label_tpb'] = 1
          else:
                fdf_test.loc[index, 'label_tpb'] = 0

      # df_main = pd.read_csv('nmf_news_users_10.csv', sep="\t")
      ind_file = 1
      fdf_s = pd.read_csv('Workers-perception-' + str(ind_file) + '.csv', sep="\t")
      news_id= collections.defaultdict()
      news_text= collections.defaultdict()
      news_idpb= collections.defaultdict()
      news_var= collections.defaultdict()
      for ind in fdf_s.index.tolist():
         news_id[ind] = fdf_s['text_id'][ind]
         news_text[ind] = fdf_s['text'][ind]
         df1_ind = df1[df1['news_id']==news_id[ind]].index.tolist()[0]
         news_var[ind] = df1['var'][df1_ind]
         news_idpb[ind] = df1['idpb'][df1_ind]

      mean_var = np.mean(news_var.values())
      mean_idpb = np.mean(np.absolute(news_idpb.values()))

      ff = pd.read_csv('pred_real_labaling_'+ str(ind_file) + '.csv', sep="\t")
      ff_dem = ff[ff['political_view'].isin(['liberal', 'veryliberal'])]
      ff_rep = ff[ff['political_view'].isin(['conservative', 'veryconservative'])]





      ff = pd.read_csv('nmf_prediction_news_total_'+str(ind_file)+'.csv', sep="\t")


      for nn in range(52):
         ff.loc[:,'shift_'+str(nn)] = ff['real_label_0']*0.0

      for index in ff.index.tolist():
         for nn in range(52):
             ff['shift_'+str(nn)][index] = ff['s_label_'+str(nn)][index] - ff['nmf_p_label_'+str(nn)][index]

      # print(len(ff))
      # p_labels = list(ff['p_label_0'])
      gt_label = list(ff['real_label_0'])[0]
      # w_labels = list(ff['s_label_0'])
      # ff['real_label_0'].plot.kde(bw_method=0.5)
      # mplpl.figure()
      # mplpl.hist(p_labels, normed=0, histtype='step', cumulative=False, color='r')
      # mplpl.hist(w_labels, normed=0, histtype='step', cumulative=False, color='b')

      yh=1
      acc = 0
      t_acc=0
      for fig_ind in ff.index.tolist():
         if fig_ind==52:
            break
         print(fig_ind)
         # fig_ind=16
         gt_label = list(ff['real_label_'+str(fig_ind)])[0]
         p_labels_mean = np.mean(list(ff['nmf_p_label_' + str(fig_ind)]))
         w_labels_mean = np.mean(list(ff['s_label_'+ str(fig_ind)]))
         ll_nmf_p_lable = list(ff['nmf_p_label_' + str(fig_ind)])
         ll_shift = list(ff['shift_' + str(fig_ind)])
         shift_mean = np.mean(list(ff['shift_'+ str(fig_ind)]))
         ll_s_lable= list(ff['s_label_' + str(fig_ind)])
         p_lable_sk= scipy.stats.skew(ff['nmf_p_label_' + str(fig_ind)])
         s_lable_sk= scipy.stats.skew(ff['s_label_' + str(fig_ind)])
         shift_lable_sk= scipy.stats.skew(ff['shift_' + str(fig_ind)])
         big_c=0
         less_c=0
         for ind_l in range(len(ll_nmf_p_lable)):
            # if ll_nmf_p_lable[ind_l]>ll_s_lable[ind_l]:
            if ll_shift[ind_l]>0:
                  big_c+=1
            elif ll_shift[ind_l]<0:
               less_c+=1

         if news_var[fig_ind] > mean_var:# and news_idpb[fig_ind] > mean_idpb:
            # if gt_label<3:
               ptl= w_labels_mean + s_lable_sk/2
            # elif gt_label>3:
            #    ptl= w_labels_mean + s_lable_sk/2
         else:
            ptl = w_labels_mean + s_lable_sk/2

         if news_var[fig_ind] < mean_var:# and news_idpb[fig_ind] > mean_idpb:
            if gt_label!=3:
               if gt_label<3 and ptl<3:
                  acc+=1
               elif gt_label>3 and ptl>3:
                  acc+=1
               t_acc+=1

         ff_dem = ff[ff['political_view'].isin(['liberal', 'veryliberal'])]
         ff_rep = ff[ff['political_view'].isin(['conservative', 'veryconservative'])]
         shift_mean_dem = np.mean(list(ff_dem['shift_'+ str(fig_ind)]))
         shift_mean_rep = np.mean(list(ff_rep['shift_'+ str(fig_ind)]))

         ff_neg = ff[ff['shift_'+ str(fig_ind)]<0]
         ff_pos = ff[ff['shift_'+ str(fig_ind)]>0]
         shift_mean_neg = np.mean(list(ff_neg['shift_'+ str(fig_ind)]))
         shift_mean_pos = np.mean(list(ff_pos['shift_'+ str(fig_ind)]))

         ff_s = ff.sort_values('shift_'+ str(fig_ind), inplace=True)


         ff_F = ff[ff['s_label_' + str(fig_ind)]==1]
         ff_MF = ff[ff['s_label_' + str(fig_ind)]==2]
         ff_M = ff[ff['s_label_' + str(fig_ind)]==3]
         ff_MT = ff[ff['s_label_' + str(fig_ind)]==4]
         ff_T = ff[ff['s_label_' + str(fig_ind)]==5]

         shift_mean_F = np.mean(list(ff_F['shift_'+ str(fig_ind)]))
         shift_mean_MF = np.mean(list(ff_MF['shift_'+ str(fig_ind)]))
         shift_mean_M = np.mean(list(ff_M['shift_'+ str(fig_ind)]))
         shift_mean_MT = np.mean(list(ff_MT['shift_'+ str(fig_ind)]))
         shift_mean_T = np.mean(list(ff_T['shift_'+ str(fig_ind)]))



         # print(shift_mean_neg)
         # print(shift_mean_pos)

         ff_n = ff.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
                                   "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
                                   "shift_"+str(fig_ind): "Shift " + str(shift_mean),})
         # ff_dem_n = ff_dem.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
         #                           "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
         #                           "shift_"+str(fig_ind): "Shift " + str(shift_mean_dem),})
         # ff_rep_n = ff_rep.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
         #                           "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
         #                           "shift_"+str(fig_ind): "Shift " + str(shift_mean_rep),})
         #
         # ff_neg_n = ff_neg.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
         #                           "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
         #                           "shift_"+str(fig_ind): "Shift " + str(shift_mean_neg),})
         # ff_pos_n = ff_pos.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
         #                           "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
         #                           "shift_"+str(fig_ind): "Shift " + str(shift_mean_pos),})



         ff_F_n = ff_F.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
                                   "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
                                   "shift_"+str(fig_ind): "Shift " + str(shift_mean_F) + " , " + str(len(ff_F)),})

         ff_MF_n = ff_MF.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
                                   "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
                                   "shift_"+str(fig_ind): "Shift " + str(shift_mean_MF) + " , " + str(len(ff_MF)),})

         ff_M_n = ff_M.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
                                   "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
                                   "shift_"+str(fig_ind): "Shift " + str(shift_mean_M) + " , " + str(len(ff_M)),})

         ff_MT_n = ff_MT.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
                                   "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
                                   "shift_"+str(fig_ind): "Shift " + str(shift_mean_MT) + " , " + str(len(ff_MT)),})

         ff_T_n = ff_T.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
                                   "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
                                   "shift_"+str(fig_ind): "Shift " + str(shift_mean_T) + " , " + str(len(ff_T)),})

         # try:
         mplpl.figure()
            # ff_n[["Workers Judgments "+str(w_labels_mean), "Predictions " + str(p_labels_mean)]].plot.kde(bw_method=0.5, color=['r', 'b'],lw=4)
            # ff_n["Shift "+str(shift_mean)].plot.kde(bw_method=0.5, color=['c'],lw=4)
            # ff_dem_n["Shift "+str(shift_mean_dem)].plot.kde(bw_method=0.5, color=['b'],lw=4)
            # ff_rep_n["Shift "+str(shift_mean_rep)].plot.kde(bw_method=0.5, color=['r'],lw=4)
            # ff_pos_n["Shift "+str(shift_mean_pos)].plot.kde(bw_method=0.5, color=['b'],lw=4)
            # ff_neg_n["Shift "+str(shift_mean_neg)].plot.kde(bw_method=0.5, color=['r'],lw=4)
            # mplpl.scatter(ff_s['s_label_' + str(fig_ind)],color='c', label='Assessment')
            # mplpl.scatter(ff_s['nmf_p_label_' + str(fig_ind)],color='purple', label='Prediction')

            # mplpl.scatter(range(len(ff)), ff['nmf_p_label_' + str(fig_ind)][:], c='purple', label='Prediction')
            # mplpl.scatter(range(len(ff)), ff['s_label_' + str(fig_ind)][:], c='c', label='Assessment')

         # try:
         #    # ff_F_n["Shift "+str(shift_mean_F)+ " , " + str(len(ff_F))].plot.kde(bw_method=0.5, color=['c'],lw=4)
         #    data = ff_n["Shift "+str(shift_mean)][:]
         #    num_bins = 20
         #    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
         #    cdf = np.cumsum(counts)
         #    mplpl.plot(bin_edges[1:], cdf / cdf[-1],color='c', lw=4)
         #    # mplpl.grid()
         # except:
         #    continue


         # try:
         #    # ff_F_n["Shift "+str(shift_mean_F)+ " , " + str(len(ff_F))].plot.kde(bw_method=0.5, color=['c'],lw=4)
         #    data = ff_F_n["Shift "+str(shift_mean_F)+ " , " + str(len(ff_F))][:]
         #    num_bins = 20
         #    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
         #    cdf = np.cumsum(counts)
         #    mplpl.plot(bin_edges[1:], cdf / cdf[-1],color='c', lw=4, label='assessment=1')
         # except:
         #    continue
         # try:
         #    # ff_MF_n["Shift "+str(shift_mean_MF)+ " , " + str(len(ff_MF))].plot.kde(bw_method=0.5, color=['b'],lw=4)
         #    data = ff_MF_n["Shift "+str(shift_mean_MF)+ " , " + str(len(ff_MF))][:]
         #    num_bins = 20
         #    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
         #    cdf = np.cumsum(counts)
         #    mplpl.plot(bin_edges[1:], cdf / cdf[-1],color='b', lw=4, label='assessment=2')
         # except:
         #    continue
         # try:
         #    # ff_M_n["Shift "+str(shift_mean_M)+ " , " + str(len(ff_M))].plot.kde(bw_method=0.5, color=['g'],lw=4)
         #    data = ff_M_n["Shift "+str(shift_mean_M)+ " , " + str(len(ff_M))][:]
         #    num_bins = 20
         #    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
         #    cdf = np.cumsum(counts)
         #    mplpl.plot(bin_edges[1:], cdf / cdf[-1],color='g', lw=4, label='assessment=3')
         # except:
         #    continue
         # try:
         #    # ff_MT_n["Shift "+str(shift_mean_MT)+ " , " + str(len(ff_MT))].plot.kde(bw_method=0.5, color=['r'],lw=4)
         #    data = ff_MT_n["Shift "+str(shift_mean_MT)+ " , " + str(len(ff_MT))][:]
         #    num_bins = 20
         #    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
         #    cdf = np.cumsum(counts)
         #    mplpl.plot(bin_edges[1:], cdf / cdf[-1],color='r', lw=4, label='assessment=4')
         # except:
         #    continue
         # try:
         #    # ff_T_n["Shift "+str(shift_mean_T)+ " , " + str(len(ff_T))].plot.kde(bw_method=0.5, color=['orange'],lw=4)
         #    data = ff_T_n["Shift "+str(shift_mean_T)+ " , " + str(len(ff_T))][:]
         #    num_bins = 20
         #    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
         #    cdf = np.cumsum(counts)
         #    mplpl.plot(bin_edges[1:], cdf / cdf[-1],color='orange', lw=4, label='assessment=5')
         # except:
         #    continue

         ff_n = ff.rename(columns={"s_label_"+str(fig_ind): "Workers Judgments "+str(w_labels_mean),
                                   "nmf_p_label_"+str(fig_ind): "Predictions " + str(p_labels_mean),
                                   "shift_"+str(fig_ind): "Shift " + str(shift_mean),})
         ff_n["Shift " + str(shift_mean)].plot.kde(bw_method=0.2, color=['b'],lw=4)
         #
         ff_n["Shift " + str(shift_mean)].plot.hist(bins=20, normed=True, edgecolor='black',color='c')  # kde(bw_method=0.5)
         # ff_n["Predictions " + str(p_labels_mean)].plot.kde(bw_method=0.2, color='b',lw=4)
         # ff_n["Workers Judgments "+str(w_labels_mean)].plot.kde(bw_method=0.2,color='r',lw=4)
         # # ff_n[["Workers Judgments "+str(w_labels_mean), "Predictions "+str(p_labels_mean)]].plot.kde(bw_method=0.2, secondary_y=True,color=['r', 'b'], lw=4)
         #
         mplpl.plot([gt_label, gt_label], [0,yh], color='g', label='Ground Truth : ' + str(gt_label), lw=4)
         #
         mplpl.xlim([-5, 7])
         # # mplpl.ylabel('CDF')
         mplpl.xlabel('Shift (assessment - prediction)')
         mplpl.grid()
         # # mplpl.ylim([0, yh])
         #
         labels = ['','','','','-1','0', 'False', 'Mostly False', 'Mixture', 'Mostly True', 'True', '', '']
         X = [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5, 6 ,7]
         mplpl.xticks(X, labels,rotation='90')
         legend_properties = {'weight': 'bold'}
         mplpl.legend(loc="upper right", prop=legend_properties, fontsize='large', ncol=1)  # , fontweight = 'bold')
         mplpl.subplots_adjust(bottom=0.2)
         # mplpl.title('skew real: ' + str(s_lable_sk)+', skew MF: ' + str(p_lable_sk)+'\n'+
         #                'Number of shift > 0 :' + str(big_c) + ', Number of shift<0 '+str(less_c))
         # # legend_properties = {'weight': 'bold'}
         mplpl.legend(loc="upper left", prop=legend_properties, fontsize='large', ncol=1)  # , fontweight = 'bold')
         mplpl.subplots_adjust(bottom=0.2)
         # # mplpl.savefig('fig_real_predication_nmf/shift_pos_neg_nfm_pre_'+str(news_id[fig_ind]), format='png')
         # # mplpl.savefig('fig_real_predication_nmf/shift_cdf_'+str(news_id[fig_ind]), format='png')
         mplpl.savefig('fig_real_predication_fn/shift_nfm_pre_'+str(news_id[fig_ind]), format='png')
         # mplpl.savefig('fig_real_predication_fn/nfm_pre_'+str(news_id[fig_ind]), format='png')

         # except:
         #    continue



         # ff_n[["Workers Judgments "+str(w_labels_mean), "Predictions "+str(w_labels_mean)]].plot.hist(color=['r', 'b'], lw=4)
         # mplpl.plot([gt_label, gt_label], [0,yh], color='g', label='Ground Truth', lw=4)
         #
         # mplpl.xlim([-1, 7])
         # mplpl.ylim([0, yh])
         #
         # labels = ['','', 'False', 'Mostly False', 'Mixture', 'Mostly True', 'True', '', '']
         # X = [-1, 0, 1, 2, 3, 4, 5, 6 ,7]
         # mplpl.xticks(X, labels,rotation='90')
         # legend_properties = {'weight': 'bold'}
         # mplpl.legend(loc="upper right", prop=legend_properties, fontsize='large', ncol=1)  # , fontweight = 'bold')
         # mplpl.subplots_adjust(bottom=0.2)
         # mplpl.savefig('fig_real_predication_nmf/'+str(ind_file)+'_nmf_pre_hist_'+str(fig_ind), format='png')


      # fdf_s = pd.read_csv('Workers-perception-3.csv', sep="\t")
      # print(len(fdf_s))


      print("acc: " +str(acc))
      print("total acc: " +str(t_acc))

      exit()
      worker_list = []
      # fdf_workers = pd.read_csv('Amt-workers-features.csv', sep="\t")
      # for worker_id in fdf_workers['worker_id']:
      #    if str(worker_id) in fdf_s.columns:
      #       worker_list.append(worker_id)
      # fdf_workers = fdf_workers[fdf_workers['worker_id'].isin(worker_list)]
      # print(len(fdf_workers))
      #
      # for ind in fdf_s.index.tolist():
      #    fdf_workers.loc[:, 's_label_'+str(ind)] = fdf_workers['worker_id'] * 0.0
      #    fdf_workers.loc[:, 'real_label_'+str(ind)] = fdf_workers['worker_id'] * 0.0
      #
      # index_s = 0
      # g_val_d = 0
      # countt=0
      # for index_s in fdf_s.index.tolist():
      #    print(index_s)
      #    countt+=1
      #    print(countt)
      #    for worker_id in worker_list:
      #       ind = fdf_workers[fdf_workers['worker_id']==worker_id].index.tolist()[0]
      #       fdf_workers['s_label_'+str(index_s)][ind] = fdf_s[str(worker_id)][index_s]
      #       g_val = fdf_s['ground_truth'][index_s]
      #       if g_val=='FALSE':
      #          g_val_d = 1
      #       elif g_val=='MOSTLY FALSE':
      #          g_val_d=2
      #       elif g_val=='MIXTURE':
      #          g_val_d=3
      #       elif g_val=='MOSTLY TRUE':
      #          g_val_d=4
      #       elif g_val=='TRUE':
      #          g_val_d=5
      #       fdf_workers['real_label_'+str(index_s)][ind] = g_val_d
      #    print(np.mean(list(fdf_workers['s_label_'+str(index_s)])))
      # fdf['label_tpb'][index] = tmp_df['tpb'][index]
      # fdf = fdf_workers.copy()
      # fdf_test = fdf_workers.copy()
      # print(fdf.columns)
      # fdf_workers.to_csv('workers_labeled_stories_3.csv', columns=fdf_workers.columns,sep="\t", index=False)
      fdf_workers = pd.read_csv('workers_labeled_stories_3.csv', sep="\t")
      fdf = fdf_workers.copy()
      fdf_test = fdf_workers.copy()
      for index_ss in fdf_s.index.tolist():
         fdf.loc[:, 'p_label_'+str(index_ss)] = fdf['worker_id'] * 0.0

      for index_s in fdf_s.index.tolist():
         print(index_s)
         print(np.mean(list(fdf_workers['s_label_'+str(index_s)])))

      # exit()
      # top_h_c = list(tmp_df['tweet_id'][:50])
      # df_h_c = fdf[fdf['tweet_id'].isin(top_h_c)]
      # for index in df_h_c.index.tolist():
      #    fdf['label_tpb'][index] = int(1)
      # #
      # top_l_c = list(tmp_df['tweet_id'][100:])
      # df_l_c = fdf[fdf['tweet_id'].isin(top_l_c)]
      # for index in df_l_c.index.tolist():
      #    fdf['label_tpb'][index] = int(-1)
      #
      #
      # mid_l_c = list(tmp_df['tweet_id'][50:100])
      # df_l_c = fdf[fdf['tweet_id'].isin(mid_l_c)]
      # for index in df_l_c.index.tolist():
      #    fdf['label_tpb'][index] = int(0)

      # fdf = fdf[fdf['tweet_id'].isin(list(top_h_c)+list(top_l_c))]

      dependentvar = 's_label'
      dependentvar_learning = 's_label'
      # dependentvar = 'label_tpb'
      # dependentvar_learning = 'label_tpb'


      # main_user_list = list(set(fdf['tweet_source_id']))
      # print(len(main_user_list))

      acc_t_f_d_l=collections.defaultdict(list)
      acc_diff_trsh_dict = collections.defaultdict()
      corr_pearson_diff_trsh_dict = collections.defaultdict()
      corr_spearman_diff_trsh_dict = collections.defaultdict()
      corr_pearson_diff_trsh_std_dict = collections.defaultdict()
      corr_spearman_diff_trsh_std_dict = collections.defaultdict()
      acc_diff_trsh_std_dict = collections.defaultdict()
      publisher_classification = {}

      labels_pred_list_all = []
      labels_test_list_all = []
      labels_real_list_all = []

      print('prediction started')

      # fdf_m = fdf_s.copy()
      fdf_m = fdf.copy()
      fdf_mt = fdf_test.copy()
      for index_s in fdf_s.index.tolist():
         # fdf.loc[:, 'p_label_' + str(index_s)] = fdf['tweet_id'] * 0.0
         labels_pred_list = []
         labels_test_list = []
         print(index_s)
         dependentvar = 's_label_' + str(index_s)
         dependentvar_learning = 's_label_'+str(index_s)
         print(np.mean(list(fdf_workers['s_label_' + str(index_s)])))
         for method in methods:
            for featset in featsets:
               for seed in seeds:
                   for Train_trsh in range_l:
                   # for Train_trsh in [0.5]:
                       print('treshold is :     ' + str(Train_trsh))
                       acc_list=[]
                       acc_true_list=[]
                       acc_false_list=[]
                       corr_list=[]
                       corr_pearson_list=[]
                       corr_spearman_list = []
                       outfilename = featset+"_"+inputfilenamebase+\
                         "_n%dk"%(nsamples/1000)+"-seed"+str(seed) + output_extra_name + '_based-on_'+dependentvar_learning + '_all_publishers'
                       # print(outfilename)
                       np.random.seed(seed)
                       random.seed(seed)
                       print ("Processing:", method, outfilename)
                       # sys.stdout = open(dirs["stdout"] + "/classify_" + method + "_" + outfilename, 'w')
                       count_user = 0
                       # for user_m in main_user_list:
                       # Train_trsh=.5
                       for predict_c in range(1):
                           print("########################################################################")

                           print('kth round : ' + str(predict_c))
                           # fdf= fdf.iloc[np.random.permutation(len(fdf))]
                           fdf_m= fdf_m.iloc[shuffle(range(len(fdf_m)),random_state=1)]
                           # print(fdf.index.tolist())

                                 # fdf_c = fdf[fdf["tweet_source_id"] == user_m]
                           fdf_c = fdf_m.copy()
                           fdf_t = fdf_mt.copy()
                           # fdf_t = fdf.copy()
                           # fdf_t = fdf_test.copy()

                           df, df_t, dflong, positions_train, posslices_test = preprocess(
                             method+"_"+outfilename, dirs, fdf_c,fdf_t, nsamples,dependentvar, Train_trsh, tm_featnm=tm_featnm )
                           if len(df)==0 or len(dflong)==0 or len(positions_train)==0 or len(posslices_test)==0:
                             continue
                           print("prediction for publisher " + str(count_user) + 'th')
                           count_user += 1
                           for user_ind in fdf_c.index.tolist():
                              print(user_ind)
                              posslices_test=[[user_ind]]
                              position_train=fdf_c.index.tolist()
                              position_train.remove(user_ind)
                              arguments = ( outfilename, dirs, fdf_c,fdf_c, fdf_c, position_train,
                                posslices_test, nsamples, method, featset,dependentvar_learning, tm_featnm )

                                # publisher_classification[user_m] = classify( *arguments )
                              # train_liklihood, test_liklihood, publisher_classification = classify( *arguments )
                              # acc, corr, corr_pearson = classify( *arguments )
                              labels_pred, labels_test = classify( *arguments )

                              labels_pred_list.append(labels_pred)
                              labels_test_list.append(labels_test)
                              fdf.loc[user_ind, 'p_label_'+str(index_s)] = labels_pred
                           # corr_pearson_list.append(corr_pearson[0][1])
                           # corr_spearman_list.append(corr_spearman[0])

                           # for ky in acc_t_f.keys():
                           #     acc_t_f_d_l[ky].append(acc_t_f[ky])
                       labels_pred_list_all.append(np.mean(labels_pred_list))
                       labels_test_list_all.append(np.mean(labels_test_list))
                       labels_real_list_all.append(fdf_workers['real_label_'+str(index_s)][index_s])
                       # print(np.mean(labels_pred_list))
                       # print(np.mean(labels_test_list))
                       # print(fdf_workers['real_label'][0])
                    # exit()
                    # import scipy.stats as st
                    #
                    # test = st.t.interval(0.90, len(acc_list) - 1, loc=np.mean(acc_list), scale=st.sem(acc_list))
                    # print("conf interval 90 : " + str(test))
                    # acc_diff_trsh_dict[Train_trsh] = np.mean(acc_list)
                    # acc_diff_trsh_std_dict[Train_trsh] = np.std(acc_list)
                    # print('avg result : ' + str(acc_diff_trsh_dict[Train_trsh]))
                    # print(acc_diff_trsh_std_dict[Train_trsh])
                    # corr_pearson_diff_trsh_dict[Train_trsh] = np.mean(corr_pearson_list)
                    # corr_spearman_diff_trsh_dict[Train_trsh] = np.mean(corr_spearman_list)
                    # corr_pearson_diff_trsh_std_dict[Train_trsh] = np.std(corr_pearson_list)
                    # corr_spearman_diff_trsh_std_dict[Train_trsh] = np.std(corr_spearman_list)
                 # if featsets[0] == "tweet_text":
              # df_t.to_csv(output, columns=df_t.columns, sep="\t", index=False)
         fdf_m = fdf.copy()
         fdf_mt = fdf_test.copy()

      print(labels_pred_list_all)
      print(labels_test_list_all)
      print(labels_real_list_all)
      pre_acc = 0
      wis_acc = 0
      for i in range(len(labels_pred_list_all)):
         if labels_real_list_all[i] == 3:
            continue
         if labels_real_list_all[i] < 3:
            if labels_pred_list_all[i] < 3:
               pre_acc += 1
            if labels_test_list_all[i] < 3:
               wis_acc += 1
         if labels_real_list_all[i] > 3:
            if labels_pred_list_all[i] > 3:
               pre_acc += 1
            if labels_test_list_all[i] > 3:
               wis_acc += 1
      print(wis_acc)
      print(pre_acc)

      acc_p = 0
      acc_n = 0
      for i in range(len(labels_test_list_all)):
         if (labels_pred_list_all[i] - labels_test_list_all[i])  < 0 and labels_real_list_all[i] < 3:
            acc_p += 1
         if (labels_pred_list_all[i] - labels_test_list_all[i])  > 0 and labels_real_list_all[i] > 3:
            acc_n += 1
      print(acc_p)
      print(acc_n)
      fdf.to_csv('pred_real_labaling_3.csv', columns=fdf.columns, sep="\t", index=False)
      exit()
      xtic=[]
      ytic_corr_spearman=[]
      ytic_corr_pearson=[]
      ytic_corr_pearson_interval = []
      ytic_corr_spearman_interval = []
      ytic_acc = []
      ytic_acc_interval = []
      print("########################################################################")
      print("########################################################################")
      # print("########################################################################")
      for trsh in range_l:
      # for trsh in [0.5]:
          xtic.append(trsh)
          # ytic_corr_pearson.append(corr_pearson_diff_trsh_dict[trsh])
          # ytic_corr_spearman.append(corr_spearman_diff_trsh_dict[trsh])
          # ytic_corr_pearson_interval.append(scipy.stats.norm.interval(0.9, loc=corr_pearson_diff_trsh_dict[trsh], scale=corr_pearson_diff_trsh_std_dict[trsh]))
          # ytic_corr_spearman_interval.append(scipy.stats.norm.interval(0.9, loc=corr_spearman_diff_trsh_dict[trsh], scale=corr_spearman_diff_trsh_std_dict[trsh]))

          ytic_acc.append(acc_diff_trsh_dict[trsh])
          ytic_acc_interval.append(scipy.stats.norm.interval(0.9, loc=acc_diff_trsh_dict[trsh], scale=acc_diff_trsh_std_dict[trsh]))

          print(acc_diff_trsh_dict[trsh])
          # print(corr_pearson_diff_trsh_dict[trsh])
          # print(corr_spearman_diff_trsh_dict[trsh])

      print("########################################################################")
      print("########################################################################")
      print("########################################################################")
      slowdown = 1.0*args.timeout/len(methods)/5.0
      print ("Slowdown by:", slowdown)
      # time.sleep(slowdown)

      mplpl.clf()
      mplpl.rc('xtick', labelsize='medium')
      mplpl.rc('ytick', labelsize='medium')
      mplpl.rc('xtick.major', size=3, pad=3)
      mplpl.rc('xtick.minor', size=2, pad=3)
      mplpl.rc('legend', fontsize='small')

      mplpl.plot(xtic, ytic_acc, c='r',linewidth=3.0, label='')
      # mplpl.plot(xtic, ytic_corr_pearson, c='b',linewidth=3.0, label='Pearson')
      count = 0
      x_conf_int = range_l
      x_conf_int1 = [x+0.01 for x in range_l]
      for conf_int in ytic_acc_interval[:]:
          mplpl.plot([x_conf_int[count],x_conf_int[count]], [conf_int[0], conf_int[1]], c='r')
          count+=1
      count = 0
      # for conf_int in ytic_corr_pearson_interval[:]:
      #     mplpl.plot([x_conf_int1[count],x_conf_int1[count]], [conf_int[0], conf_int[1]], c='b')
      #     count+=1
      # mplpl.plot(xtic, ytic_corr_pearson, c='c',linewidth=3.0, label='Pearson')
      # count = 0
      # x_conf_int = frange(0.1, 1, 0.05)
      # for conf_int in ytic_corr_pearson_interval[:]:
      #     mplpl.plot([x_conf_int[count],x_conf_int[count]], [conf_int[0], conf_int[1]], c='c')
      #     count+=1
      #
      # mplpl.plot(xtic, ytic_corr_spearman, c='g',linewidth=3.0, label='Spearman(Ranked)')
      print(ytic_corr_pearson_interval[:])
      mplpl.grid()


      mplpl.xlim([0,1])
      mplpl.ylim([0,1])

      mplpl.xlabel('Train set ratio (Cost)', fontsize='large')
      # mplpl.ylabel('Accuracy of Prediction of GT', fontsize='large')
      mplpl.ylabel('Accuracy of Prediction of binary TPB', fontsize='large')
      # mplpl.ylabel('Correlation of Prediction of TPB', fontsize='large')
      mplpl.title(featsets[0], fontsize='small')
      mplpl.legend(loc="upper left")


      # mplpl.savefig('fake_news_data/fig/Corr_TPB_sp_incentive_50claims', format='png')
      # mplpl.savefig('fake_news_data/fig/ACC_binary_TPB_sp_ssi', format='png')
      # mplpl.savefig('fake_news_data/fig/ACC_TPB_binary_sp_gt_1_50claims_svm_'+featsets[0], format='png')
      # mplpl.savefig('fake_news_data/fig/ACC_TPB_binary_sp_gt_150claims_svm_all_rfe_'+featsets[0], format='png')
      # mplpl.savefig('ACC_TPB_binary_sp_gt_150claims_reg_all_'+featsets[0], format='png')
      mplpl.savefig('ACC_TPB_binary_sp_gt_150claims_svm_all_'+featsets[0], format='png')
      # mplpl.savefig('ACC_TPB_binary_sp_gt_150claims_svm_all1_'+'all_features', format='png')
      # mplpl.savefig('ACC_TPB_binary_sp_gt_150claims_svm_all_chi_'+featsets[0], format='png')
      # mplpl.savefig('ACC_TPB_binary_sp_gt_150claims_svm_all_chi_allfeat', format='png')
      # mplpl.savefig('ACC_TPB_binary_sp_gt_150claims_reg_rfe_all_allfeat', format='png')
      # mplpl.savefig('fake_news_data/fig/Corr_GT_sp', format='png')
      # mplpl.savefig('fake_news_data/fig/Corr_GT_sp_LogisticIT', format='png')
      # mplpl.savefig('fake_news_data/fig/Acc_GT_sp_LogisticIT', format='png')
      # mplpl.savefig('fake_news_data/fig/Acc_GT_sp', format='png')


      # mplpl.figure()
      #
      # mplpl.clf()
      # mplpl.rc('xtick', labelsize='medium')
      # mplpl.rc('ytick', labelsize='medium')
      # mplpl.rc('xtick.major', size=3, pad=3)
      # mplpl.rc('xtick.minor', size=2, pad=3)
      # mplpl.rc('legend', fontsize='small')
      #
      # mplpl.plot(xtic, ytic_corr_spearman, c='g',linewidth=2.0)
      # mplpl.grid()
      #
      #
      # mplpl.xlim([0,1])
      # mplpl.ylim([0,1])
      #
      # mplpl.xlabel('Train set ratio', fontsize='large')
      # mplpl.ylabel('Spearman(Ranked) Corr between \nActual_TPB and Predicted_TPB', fontsize='large')
      # mplpl.title('', fontsize='small')
      # mplpl.legend(loc="upper right")
      #
      #
      # mplpl.savefig('fake_news_data/fig/spearman_corr_TPB_sp', format='png')


      mplpl.show()

if __name__ == '__main__':

    main()

