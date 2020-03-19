#!/usr/bin/python
# -*- coding: ascii -*-
import string, os, sys, code, time, multiprocessing, pickle, glob, re, time
import collections, shelve, copy, itertools, math, random, argparse, warnings
import bisect, subprocess
import StringIO
import csv
import pandas as pd
import numpy as np
from pandas import (Index, Series, TimeSeries, DataFrame, isnull, notnull,
                    bdate_range, date_range, concat)
# terminal config
terminal_columns = 300
# this below is just to check if python is in interactive mode
import __main__
if not hasattr(__main__, '__file__'):
   terminal_size = os.popen('stty size', 'r').read().split()
   terminal_rows = int(terminal_size[0])
   terminal_columns = int(terminal_size[1])

import pandas as pd
terminal_columns, terminal_rows = pd.util.terminal.get_terminal_size()
pd.set_option( "display.width", terminal_columns )
pd.set_option( 'max_columns', terminal_columns/9 )

import numpy as np
np.set_printoptions( linewidth=terminal_columns )
np.random.seed(1234567890)

import scipy
import scipy.stats
import scipy.optimize

sys.path.append(os.path.expanduser("/home/pms/Dropbox/repo/py/pms"))
# import basictools
# import basicfigs
# import averages
# import mypylzma


def load_htnames( inputfilename ):
   """Load hashtag - id mapping"""
   htnames = dict()
   with open( inputfilename ) as inputfile:
      for line in inputfile:
         fields = line.split(" ")
         htnames[ int(fields[1]) ] = fields[0]
   return htnames

def load_data( inputpath, nrows=None ):
   """Load samples and their features for classification problem"""
   #pass
   tweet_id_l=[];user_id_l=[];class_ave_id_l=[];class_mid_l=[];ntweets_l=[]
   perc_en_l=[];retweet_count_avg_l=[];retweet_count_med_l=[];followers_count_l=[];
   friends_count_l=[];retweet_count_l=[];favorite_count_l=[];
   in_reply_to_uid_l=[];retweeted_uid_l=[];created_at_l=[];lang_l=[];text_l=[];in_reply_to_tid_l=[]

   if inputpath.endswith(".dat.rar"):
      inputfilename = os.path.splitext(os.path.basename(inputpath))[0]
      if not os.path.isfile(inputpath):
         if not os.path.isfile(inputpath):
             # continue
            print("somthing is wrong")
      print "Loading file: "+inputfilename+".",
      # sio = mypylzma.lzmafile(inputpath).getsio()
      cmd = ["7z", "e", "-so", inputpath]
      fnull = open(os.devnull, "w")
      try: sio = StringIO.StringIO( subprocess.check_output(cmd,stderr=fnull) )
      except subprocess.CalledProcessError:print("somthing is wrong") #continue
      readdf = pd.read_csv( sio, sep="\t", nrows=nrows, na_values=[-1],
         index_col=False )
      del sio
   else:
        inputfilename = os.path.splitext(os.path.basename(inputpath))[0]
        if not os.path.isfile(inputpath):
         if not os.path.isfile(inputpath):
            # continue
              print("somthing is wrong")

        print "Loading file: "+inputfilename+".",
        # if "csv" in inputpath:
        # readdf = pd.read_csv( inputpath, sep="\t", nrows=nrows, na_values=[-1],index_col=None )
        # readdf = pd.read_csv(inputpath, sep="\t",error_bad_lines=False, index_col=0)
        readdf = pd.read_csv(inputpath, sep="\t", encoding='utf8', engine='python',index_col=None)
        # readdf = pd.read_csv(inputpath, parse_dates=True, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
        # else:
        #     inputfile = open(inputpath,'r')
        #     header_line = inputfile.readline()
        #     header_list = header_line.split('\t')
        #     for line in inputfile:
        #        line_split = line.split('\t')
        #        tweet_id_l.append(int(line_split[0]))
        #        user_id_l.append(int(line_split[1]))
        #        class_ave_id_l.append(int(line_split[2]))
        #        class_mid_l.append(int(int(line_split[3])))
        #        ntweets_l.append(int(line_split[4]))
        #        perc_en_l.append(float(line_split[5]))
        #        retweet_count_avg_l.append(float(line_split[6]))
        #        retweet_count_med_l.append(float(line_split[7]))
        #        followers_count_l.append(int(line_split[8]))
        #        friends_count_l.append(int(line_split[9]))
        #        retweet_count_l.append(int(line_split[10]))
        #        favorite_count_l.append(int(line_split[11]))
        #        in_reply_to_uid_l.append(int(line_split[12]))
        #        in_reply_to_tid_l.append(int(line_split[13]))
        #        retweeted_uid_l.append(int(line_split[14]))
        #        created_at_l.append(int(line_split[15]))
        #        lang_l.append(line_split[16])
        #        text_l.append(line_split[17])
        #     readdf = DataFrame({'tweet_id':Series(tweet_id_l, index=tweet_id_l),
        #                        'user_id':Series(user_id_l, index=tweet_id_l),
        #                        'class_ave_id':Series(class_ave_id_l, index=tweet_id_l),
        #                        'class_mid':Series(class_mid_l, index=tweet_id_l),
        #                        'ntweets':Series(ntweets_l, index=tweet_id_l),
        #                        'perc_en':Series(perc_en_l, index=tweet_id_l),
        #                        'retweet_count_avg':Series(retweet_count_avg_l, index=tweet_id_l),
        #                        'retweet_count_med':Series(retweet_count_med_l, index=tweet_id_l),
        #                        'followers_count':Series(followers_count_l, index=tweet_id_l),
        #                        'friends_count':Series(friends_count_l, index=tweet_id_l),
        #                        'retweet_count':Series(retweet_count_l, index=tweet_id_l),'favorite_count':Series(favorite_count_l, index=tweet_id_l),
        #                        'in_reply_to_uid':Series(in_reply_to_uid_l, index=tweet_id_l),'in_reply_to_tid':Series(in_reply_to_tid_l, index=tweet_id_l),
        #                        'retweeted_uid':Series(retweeted_uid_l,index=tweet_id_l),'created_at':Series(created_at_l, index=tweet_id_l),
        #                        'lang':Series(lang_l, index=tweet_id_l),'text':Series(text_l, index=tweet_id_l),})



   # print "Loaded",len(readdf)
   # print "Loaded following columns:", " ".join( readdf.columns )
   # readdf.iloc[-1]=0
   # print readdf.iloc[-3:]
   # correctrows = readdf["adopters_so_far"]>=0
   # correctrows = readdf["adopters_so_far"].notnull()
   # print "Loaded: %d / %d samples."%(
   #    np.sum(correctrows), len(readdf) )
   # readdf = readdf.ix[ correctrows, :]

   # print "Samples:"
   # print readdf.iloc[215:217,:]

   return readdf


def pubname( codename ):
   # return codename
   # social
   if codename=="social_exposure": return r"Social exposure $\kappa$"
   # user
   if codename=="adoptions_sofar": return r"$n_m(t)$"
   if codename=="adopter_outdeg": return r"$k^{out}$"
   if codename=="adopter_indeg": return r"$k^{in}$"
   if codename=="sim_avgexp": return r"$S($user$,\langle$user$\rangle)$"
   # meme
   if codename=="adopters_so_far": return r"$n_u(t)$"
   if codename=="tot_outdeg_sofar": return r"$k^{out}$"
   if codename=="tot_indeg_sofar": return r"$k^{in}$"
   if codename=="sim_avgpastmeme": return r"$S($meme$,\langle$meme$\rangle)$"
   if codename=="entropy_pastmeme": return r"$H_t($meme$)$"
   if codename=="entropy_exp": return r"$H($user$)$"
   if codename=="entropy_meme": return r"$H($meme$)$"
   if codename=="sim_pastmeme_prev": return r"$S($meme$(t),$meme$(t-1))$"
   if codename=="sim_pastmeme_first": return r"$S($meme$(t),$meme$(0))$"
   # topic - dynamic
   if codename=="sima_exp_pastmeme": return r"$S($user,meme$)$"
   if codename=="sima_exp_exp": return r"$S($user,exposer$)$"
   if codename=="sima_exposerexp_pastmeme": return r"$S($exposer,meme$)$"
   # topic - static
   if codename=="sima_exp_meme": return r"$S($user,meme$)$"
   if codename=="sima_exposerexp_meme": return r"$S($exposer,meme$)$"
   return codename

feats_exposer = [
   # social
   # influence
   # "exposer_tot_outdeg", "exposer_tot_indeg",
   "exposer_avg_outdeg", "exposer_avg_indeg",
   # topic
   "sima_int_int", "sima_exp_exp", "sima_int_exp", "sima_exp_int",
   "sima_exposerint_meme", "sima_exposerexp_meme",
   "sima_exposerint_pastmeme", "sima_exposerexp_pastmeme",
]

feats_staticmeme = [
   "sim_avgmeme", "entropy_meme",
   "sima_int_meme", "sima_exp_meme",
   "sima_exposerint_meme", "sima_exposerexp_meme"
]

feats_dynamicmeme = [
   "sim_avgpastmeme", "entropy_pastmeme",
   "jsdiv_pastmeme_prev", "jsdiv_pastmeme_first",
   "sim_pastmeme_prev", "sim_pastmeme_first",
   "pearson_pastmeme_prev", "pearson_pastmeme_first",
   "sima_int_pastmeme", "sima_exp_pastmeme",
   "sima_exposerint_pastmeme", "sima_exposerexp_pastmeme"
]

featsall = [
   # social
   # influence
   "social_exposure",
   # "exposer_tot_outdeg", "exposer_tot_indeg",
   "exposer_avg_outdeg", "exposer_avg_indeg",

   # user
   "adoptions_sofar",
   "adopter_outdeg", "adopter_indeg",
   "sim_avgint", "sim_avgexp",

   # meme
   "adopters_so_far",
   "avg_outdeg_sofar", "avg_indeg_sofar",
   "tot_outdeg_sofar", "tot_indeg_sofar",

   "sim_avgmeme", "entropy_meme",
   # "sim_avgpastmeme", "entropy_pastmeme",

   # # dynamic-only
   # "jsdiv_pastmeme_prev", "jsdiv_pastmeme_first",
   # "sim_pastmeme_prev", "sim_pastmeme_first",
   # "pearson_pastmeme_prev", "pearson_pastmeme_first",

   # topic
   "sima_int_int", "sima_exp_exp", "sima_int_exp", "sima_exp_int",

   "sima_int_meme", "sima_exp_meme",
   "sima_exposerint_meme", "sima_exposerexp_meme",
   # "sima_int_pastmeme", "sima_exp_pastmeme",
   # "sima_exposerint_pastmeme", "sima_exposerexp_pastmeme"

   # other
   # "first_adopt_time",
]

featsall_intvec = [
   # user
   "sim_avgint",
   # topic
   "sima_int_int", "sima_int_exp", "sima_exp_int",
   "sima_int_meme",
   "sima_exposerint_meme",
   "sima_int_pastmeme",
   "sima_exposerint_pastmeme",
   "entropy_int"
]

featsall_expvec = [
   # user
   "sim_avgexp",
   # topic
   "sima_exp_exp", "sima_int_exp", "sima_exp_int",
   "sima_exp_meme",
   "sima_exposerexp_meme",
   "sima_exp_pastmeme",
   "sima_exposerexp_pastmeme",
   "entropy_exp"
]

def get_featnames_from_filename( filename, verbose=False ):
   if verbose: print "Retriving filtered featnames:", filename

   # featstotrain = [ "first_adopt_time" ]
   featstotrain = [ ]

   # if  "social" in filename or "allfeats" in filename:
   #    featstotrain += ['ret_chi', 'f_ret_chi', 'rep_chi', 'f_rep_chi','ret_rep_chi','ret_skew',
   #                     'ret_mean', 'ret_median', 'rep_skew', 'rep_mean', 'rep_median', 'cluster_num', 'num_replyes', 'num_retweets']

   if "social" in filename or "allfeats" in filename:
      featstotrain += ['dem_num_ret', 'rep_num_ret','neut_num_ret','num_retweets','dem_num_ret_mean','rep_num_ret_mean'
         ,'neut_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi', 'tweet_source_score']
      #
      # featstotrain += ['dem_num_ret', 'rep_num_ret','neut_num_ret','dem_num_ret_mean','rep_num_ret_mean'
      #    ,'neut_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi', 'tweet_source_score','ret_skew',
      #                  'ret_mean', 'ret_median', 'rep_skew', 'rep_mean', 'rep_median']

      # featstotrain += ['dem_num_ret', 'rep_num_ret','neut_num_ret','dem_num_ret_mean','rep_num_ret_mean'
      #    ,'neut_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi', 'tweet_source_score']
      #
      # featstotrain += ['dem_num_ret_weight_mean', 'rep_num_ret_weight_mean', 'dem_num_ret', 'rep_num_ret','neut_num_ret','dem_num_ret_mean','rep_num_ret_mean'
      #    ,'neut_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi', 'tweet_source_score']

      # featstotrain += ['dem_num_ret', 'rep_num_ret','neut_num_ret','num_retweets','dem_num_ret_mean','rep_num_ret_mean'
      #    ,'neut_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi', 'tweet_source_score',
      #                  'dem_num_ret_weight_mean', 'rep_num_ret_weight_mean','neut_num_ret_weight_mean', 'dem_sum_abs','rep_sum_abs','neut_sum_abs']


      # featstotrain += ['dem_num_ret','neut_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi']
      # featstotrain += ['dem_num_ret','dem_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi']
      # featstotrain += ['dem_num_ret','dem_num_ret_mean','neut_num_ret','neut_num_ret_mean','rep_chi', 'ret_chi','ret_rep_chi']













      # featstotrain += ['dem_num_ret_weight_mean', 'rep_num_ret_weight_mean', 'neut_num_ret_weight_mean',
      #                  'dem_sum_abs','rep_sum_abs','neut_sum_abs','rep_chi', 'ret_chi','ret_rep_chi']

      # featstotrain += ['dem_num_ret_weight_mean', 'rep_num_ret_weight_mean', 'neut_num_ret_weight_mean',
      #                  'dem_sum_abs', 'rep_sum_abs', 'neut_sum_abs']
      # featstotrain += ['dem_num_ret_weight_mean', 'neut_num_ret_weight_mean', 'rep_num_ret_weight_mean']





      # featstotrain += ['dem_num_ret_weight_mean', 'rep_num_ret_weight_mean',
      #                  'dem_sum_abs','rep_sum_abs','rep_chi', 'ret_chi','ret_rep_chi']
      # featstotrain += ['ret_rep_chi', 'ret_chi', 'rep_chi']


      # featstotrain += ['dem_sum_abs','rep_sum_abs','rep_chi', 'ret_chi', 'ret_rep_chi', 'ret_skew', 'rep_skew','tweet_source_score']
      # featstotrain += ['dem_sum_abs','rep_sum_abs','rep_chi', 'ret_chi', 'ret_rep_chi']
      # featstotrain += ['rep_chi', 'ret_chi', 'ret_rep_chi','tweet_source_score']
      # featstotrain += ['dem_num_ret', 'neut_num_ret','rep_chi', 'ret_chi', 'ret_rep_chi','tweet_source_score']


      # pattern = "sp_all_replyers_retweeters_polarization_consensus_metrics_ATM_offensive.csv"


      # featstotrain += ['rep_chi', 'ret_chi', 'ret_rep_chi']

      # featstotrain += ['dem_num_ret_weight_mean', 'rep_num_ret_weight_mean',
      #                  'dem_sum_abs','rep_sum_abs']

      # featstotrain += ['consensus_measure_weight_1','consensus_measure_weight_2','consensus_measure_weight',
      #                  'consensus_measure','consensus_measure_1','consensus_measure_2']
      #'dem_ave_ret','rep_ave_ret']#, 'neut_num_ret','neut_ave_ret']
   # if "social" in filename or "allfeats" in filename:
   #    featstotrain += ['ret_chi', 'f_ret_chi', 'rep_chi', 'f_rep_chi', 'ret_rep_chi', 'ret_skew',
   #                     'ret_mean', 'ret_median', 'rep_skew', 'rep_mean', 'rep_median', 'cluster_num']

         # if  "social" in filename or "allfeats" in filename:
      # featstotrain += ['ret_chi', 'rep_chi','ret_rep_chi','ret_skew',
                       # 'ret_mean', 'ret_median', 'rep_skew', 'rep_mean', 'rep_median']
   # if  "social" in filename or "allfeats" in filename:
   #    featstotrain += ['cluster_num']#,


   # if  "user" in filename or "allfeats" in filename:
   #    featstotrain += [
   #       "adoptions_sofar",
   #       "adopter_outdeg", "adopter_indeg"
   #       # "sim_avgint", "sim_avgexp"
   #       ]

   return [],featstotrain


def filter_data( apdfs, featstoclean=None, verbose=True, filtering="" ):
   """
   clean data
   filtering - added lately just for tht gt option
      so it does rather little
   """
   apdfscopy = apdfs.copy()
   return apdfscopy



def filter_by_topicality( apdfs, topicality, quartilenm, nbins=10,
   splitbystd=False, verbose=True ):
   apdfscopy = dict()

   if "double" not in topicality:
      quantiles = []
      if type(quartilenm) == list: quantiles = quartilenm
      # this overlaps with the code in the other parts
      if "f" in quartilenm: quantiles += [0]
      if "s" in quartilenm: quantiles += [1]
      if "t" in quartilenm: quantiles += [2]
      if "r" in quartilenm: quantiles += [3]
      if "i" in quartilenm: quantiles += [4]
      if "x" in quartilenm: quantiles += [5]
      if "e" in quartilenm: quantiles += [6]
      if "l" in quartilenm: quantiles += [nbins-1]
      if "all" in quartilenm: quantiles = range(nbins)
      if "notf" in quartilenm: quantiles = [0, "notf"]
      if "notl" in quartilenm: quantiles = ["notl", nbins-1]
      if "jnotf" in quartilenm: quantiles = ["notf"]
      if "jnotl" in quartilenm: quantiles = ["notl"]
      # if "f" not in quartilenm and "l" not in quartilenm:
      #    quantiles += [ int(quartilenm) ]

      print "Filtering for topicality quantiles:", quantiles

      for key in apdfs:
         for quantile in quantiles:
            if verbose: print "Filtering for:", topicality, quantile
            try:
               dim = apdfs[key][topicality].values
               if splitbystd:
                  dimstd = apdfs["std"][topicality].values
                  binedges = np.percentile( dimstd, np.linspace(0, 100, nbins+1).tolist() )
               else:
                  binedges = np.percentile( dim, np.linspace(0, 100, nbins+1).tolist() )
               # not very precise, but doesn't matter at this point
               if type(quantile)==int:
                  chosenrows = ( (dim>=binedges[quantile]) & (dim<=binedges[quantile+1]) )
               else:
                  if quantile=="notf":
                     chosenrows = ( (dim>=binedges[1]) & (dim<=binedges[nbins]) )
                  if quantile=="notl":
                     chosenrows = ( (dim>=binedges[0]) & (dim<=binedges[nbins-1]) )
               newkey = key+"-top"+str(quantile)
               if len(quantiles)==1: newkey = key
               apdfscopy[newkey] = copy.deepcopy( apdfs[key].ix[ chosenrows, : ] )
            except ValueError:
               print "ERROR: ValueError error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
               print key, topicality, len(dim)
               print apdfs[key][topicality]
               print dim
               print min(dim), max(dim)
               print vals
               sys.exit()
   else:
      quantiles = []
      # if "f" in quartilenm: quantiles += [0]
      # if "l" in quartilenm: quantiles += [nbins-1]
      # quartiles1 = quartiles2 = quantiles
      if quartilenm=="fl": quartiles1 = quartiles2 = [0, nbins-1]
      if quartilenm=="f": quartiles1 = quartiles2 = [0]
      if quartilenm=="s": quartiles1 = [0]; quartiles2 = [nbins-1]
      if quartilenm=="t": quartiles1 = [nbins-1]; quartiles2 = [0]
      if quartilenm=="l": quartiles1 = quartiles2 = [nbins-1]

      print "Filtering for topicality quantiles:", quartiles1, quartiles2

      for key in apdfs:
         for q1 in quartiles1:
            for q2 in quartiles2:
               if verbose: print "Filtering for:", topicality, q1, q2
               try:
                  dim1 = apdfs[key]["entropy_meme"].values
                  dim2 = apdfs[key]["entropy_exp"].values

                  binedges1 = np.percentile( dim1, np.linspace(0, 100, nbins+1).tolist() )
                  binedges2 = np.percentile( dim2, np.linspace(0, 100, nbins+1).tolist() )
                  chosenrows = \
                     ( (dim1>=binedges1[q1]) & (dim1<=binedges1[q1+1]) ) &\
                     ( (dim2>=binedges2[q2]) & (dim2<=binedges2[q2+1]) )

                  newkey = key+"-top"+str(q1)+str(q2)
                  if len(quartiles1)==1 and len(quartiles2)==1: newkey = key
                  apdfscopy[newkey] = copy.deepcopy( apdfs[key].ix[ chosenrows, : ] )
               except ValueError:
                  print "ERROR: ValueError error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
                  print key, topicality, len(dim)
                  print apdfs[key][topicality]
                  print dim
                  print min(dim), max(dim)
                  print vals
                  sys.exit()

   return apdfscopy




def transform_values_to_cdf( totrans, reference=None ):
   if not type(reference).__module__ == np.__name__:
      reference=totrans

   # first, sort the values
   indices = np.argsort( reference )
   reverse = np.argsort( indices )
   reference = reference[indices]

   # this method is fast, exactly cdf
   transformed = []
   for v in totrans:
      i = bisect.bisect_right( reference, v )
      transformed += [ 1.0*i/len(reference) ]
   transformed = np.array(transformed)

   return transformed

def transform_df_to_cdf( df, featstotrans=None ):
   """transforms each column of a data frame to a cdf"""
   # print "Running CDF transform"
   if featstotrans==None:
      feats = [ col for col in df.columns if col.startswith("sim") ]
   else: feats = featstotrans
   # print "CDF transfrom for columns:", feats
   for feat in feats:
      correctrows = df[feat].notnull()
      df.ix[correctrows,feat] = \
         transform_values_to_cdf( df.ix[correctrows,feat].values )

def add_transform_df_to_cdf( df, featstotrans=None ):
   """transforms each column of a data frame to a cdf"""
   # print "Running CDF transform"
   if featstotrans==None:
      feats = [ col for col in df.columns if col.startswith("sim") ]
   else: feats = featstotrans
   # print "CDF transfrom for columns:", feats
   for feat in feats:
      correctrows = df[feat].notnull()
      df.ix[correctrows,"cdf_"+feat] = \
         transform_values_to_cdf( df.ix[correctrows,feat].values )

def transform_apdfs_to_cdf( apdfs, featstotrans=None ):
   """transforms for each df of apdfs each column starting with 'sim' to a cdf"""
   # print "Running CDF transform"
   print "Transforming for:",
   apdfscopy = copy.deepcopy(apdfs)
   for key in apdfscopy:
      print key,
      transform_df_to_cdf( apdfscopy[key], featstotrans=featstotrans )
   print "Finished"
   return apdfscopy


def pubname_featset( featset ):
   pubfeatset = featset
   # pubfeatset = featset.replace( "exptopic", "topic" )
   pubfeatset = featset.replace( "exttopic", "topic" )
   pubfeatset = featset.replace( "eetopic", "topic" )
   pubfeatset = featset.replace( "etopic", "topic" )
   pubfeatset = pubfeatset.replace( "allfeats", "all" )
   pubfeatset = re.findall(
      "(user|meme|social|topic|all|simee|simem|socexp|extsim|exttopic|extmeme"+\
      "|extuser|extallfeats|etopic|eallfeats|eetopic|eeallfeats)",
      pubfeatset)
   pubfeatset = "+".join( pubfeatset )
   return pubfeatset.title()


# # testing, to remove
# df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
# apdfs=dict()
# apdfs["b"]=df
# df.columns = ["sim_", "asim_" , "sim_add", "da" ]
# df["sim_add"]+=2

# # test1
# print apdfs
# apdfs = transform_apdfs_to_cdf( apdfs )

# print apdfs
# # test2
# print df["sim_"].values
# print df["sim_add"].values
# print "Transformed:"
# print transform_values_to_cdf( df["sim_"].values, df["sim_add"].values )
# print transform_values_to_cdf( df["asim_"].values, df["sim_add"].values )
# print transform_values_to_cdf( df["sim_add"].values, df["sim_add"].values )
# print transform_values_to_cdf( df["da"].values, df["sim_add"].values )

# # test3
# print np.apply_along_axis( transform_values_to_cdf, 0, df.values,
#    df["sim_add"].values )

# # # test4
# print df
# for col in df:
#    df[col] = transform_values_to_cdf( df[col].values, df["sim_add"].values )
# print df
