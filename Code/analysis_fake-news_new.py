import string, json, os, sys, code, time, multiprocessing, pickle, glob, re, cPickle
import collections, shelve, copy, itertools, math, random, argparse, warnings
import sqlite3, gzip, datetime, socket, getpass, csv, gc, traceback
import operator, gc
import pylab as Plab
from scipy.stats import expon

sys.path.append(os.path.expanduser("/home/babaei/Desktop/SVN/streaming_api/with_tweepy/dataset-master"))
# import dataset
# import Image
import psycopg2 as pgsql

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplpl
import pickle
import cPickle

import urllib
# import mysql.connector as conn
from scipy import spatial
from scipy.stats.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
import scipy
import scipy.stats
import scipy.optimize
from scipy.sparse import coo_matrix, hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from textblob import TextBlob


# from ifa.distribution import Distribution
# from ifa.divergence import jsd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import tweetstxt_basic
from pandas import Series
from numpy.testing import assert_allclose
# import seaborn as sns
from dateutil.parser import parse

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# import base_prediction
# import basictools

import entropy_estimator

import sklearn
import basics_prediction

from numpy import zeros, array
from math import sqrt, log

import gzip, json, pickle, numpy
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
# from textblob import TextBlob
# from lxml import html
import requests
# from ehp import *
import twokenize

import nltk
from nltk.cluster.kmeans import KMeansClusterer

from HTMLParser import HTMLParser
# from htmlentitydefs import name2codepoint
#
class MyHTMLParser(HTMLParser):
    globflag = False
    def handle_starttag(self, tag, attrs):
        # out_list = []
        # return "Start tag:", tag
        try:
            for attr in attrs:
                try:
                    if 'article-link-category' in attr or 'claimReviewed' in attr or 'datePublished' in attr or 'author' in attr:
                        out_list.append(self.get_starttag_text())
                        # print(self.get_starttag_text())
                except:
                    continue
        except:
            print('something is wrong')


    def handle_decl(self, data):
        return "Decl     :", data
#
#


def plot_density_mean_median(dataframe_data, m_xlim=(-0.5, 0.5), m_ylim=[0, 10], m_color='r',
                             m_xlable='', m_ylable='',m_title='no_title', saving_path=''):

    mplpl.clf()
    mplpl.rc('xtick', labelsize='medium')
    mplpl.rc('ytick', labelsize='medium')
    mplpl.rc('xtick.major', size=3, pad=3)
    mplpl.rc('xtick.minor', size=2, pad=3)
    mplpl.rc('legend', fontsize='small')

    dataframe_data.plot(kind="density", xlim= m_xlim, color=m_color)
    # mplpl.plot(frange(-0.45, .5, 0.05), ave_ret_dist_cluster_hist, '*', color='orange')
    mplpl.vlines(dataframe_data.mean(), ymin=0, ymax=2, linewidth=2.0)
    mplpl.vlines(dataframe_data.median(), ymin=0, ymax=2, linewidth=2.0, color="red")


    mplpl.xlim([-1,1])
    mplpl.ylim(m_ylim)

    mplpl.xlabel(m_xlable, fontsize='large')
    mplpl.ylabel(m_ylable, fontsize='large')
    mplpl.title(m_title, fontsize='small')
    mplpl.legend(loc="upper right")


    mplpl.savefig(saving_path, format='png')

    # mplpl.show()


def plot_2_density_mean_median(dataframe_data, m_xlim=(-0.5, 0.5), m_ylim=[0, 10],
                             m_xlable='', m_ylable='',m_title='no_title', saving_path=''):

    mplpl.clf()
    mplpl.rc('xtick', labelsize='medium')
    mplpl.rc('ytick', labelsize='medium')
    mplpl.rc('xtick.major', size=3, pad=3)
    mplpl.rc('xtick.minor', size=2, pad=3)
    mplpl.rc('legend', fontsize='small')

    # col_1, col_2 = dataframe_data.columns
    # print(col_1)
    # print(col_2)
    #
    # df = dataframe_data.copy()
    # df_1 = df.drop(col_2,1)
    # df_2 = df.drop(col_2,1)
    try:
        dataframe_data.plot(kind="density", xlim= m_xlim)
        mplpl.vlines(dataframe_data.mean(), ymin=0, ymax=2, linewidth=2.0)
        mplpl.vlines(dataframe_data.median(), ymin=0, ymax=2, linewidth=2.0, color="red")



        mplpl.xlim([-1,1])
        mplpl.ylim(m_ylim)

        mplpl.xlabel(m_xlable, fontsize='large')
        # mplpl.ylabel(m_ylable, fontsize='large')
        mplpl.title(m_title, fontsize='small')
        mplpl.legend(loc="upper right")


        mplpl.savefig(saving_path, format='png')
    except:
        print ("there is something wrong with plotting")
    # mplpl.show()


def modify_demographic(dem_name):
    if dem_name == 'prefernotrespond':
        return 'prefer not respond'
    elif dem_name == 'living_with_partner':
        return 'living with partner'

    elif dem_name == 'collegegraduatebsbaorother4yeardegree':
        # return 'college graduate bs ba or other 4 year degree'
        return 'college graduate bs/ba'
    elif dem_name == 'technicaltradeorvocationalschoolafterhighschool':
        # return 'technical trade or vocational school after highschool'
        return 'technical trade or vocational school'
    elif dem_name == 'postgraduatetrainingorprofessionalschoolingaftercollegeegtowardamastersdegreeorphdlawormedicalschool':
        # return 'post graduate training or professional schooling after college (e.g.toward a masters degree or phd law or medical school'
        return 'post graduate training or professional schooling'
    elif dem_name == 'highschoolgraduategrade12orgedcertificate':
        # return 'highschool graduate grade 12 or e.g. dcertificate'
        return 'highschool graduate'
    elif dem_name == 'somecollegeassociatedegreeno4yeardegree':
        # return 'some college associate degree no 4 year degree'
        return 'some college associate'
    elif dem_name == '100001ndash150000':
        return '100001-150000'
    elif dem_name == '40001ndash50000':
        return '40001-50000'
    elif dem_name == '70001ndash100000':
        return '70001-100000'
    elif dem_name == '60001ndash70000':
        return '60001-70000'
    elif dem_name == '150001ormore':
        return '150001 or more'
    elif dem_name == '10000ndash20000':
        return '10000-20000'
    elif dem_name == '50001ndash60000':
        return '50001-60000'
    elif dem_name == '20001ndash30000':
        return '20001-30000'
    elif dem_name == 'under10000':
        return 'under 10000'
    elif dem_name == '30001ndash40000':
        return '30001-40000'
    elif dem_name == 'hispanicorlatino':
        return 'hispanic or latino'
    elif dem_name == 'americanindianoralaskanative':
        return 'american-indian or alaska-native'
    elif dem_name == 'other':
        return 'other'
    elif dem_name == 'asian':
        return 'asian'
    elif dem_name == 'blackorafricanamerican':
        return 'black or african-american'
    elif dem_name == 'prefernotrespond':
        return 'prefer not respond'
    elif dem_name ==  'nativehawaiianorotherpacificislander':
        return 'native hawaiian or other pacificis lander'
    elif dem_name == 'white':
        return 'white'
    elif dem_name == 'unitedstates':
        return 'united states'
    elif dem_name == 'canada':
        return 'canada'
    elif dem_name == 'unitedkingdom':
        return 'united kingdom'
    elif dem_name == 'unitedstates':
        return 'unitedstates'
    elif dem_name == '0.0':
        return '0.0'
    elif dem_name == '1.0':
        return '1.0'
    elif dem_name == '-1.0':
        return '-1.0'
    elif dem_name == '-10.0':
        return '-10.0'
    elif dem_name == '65-74':
        return '65-74'
    elif dem_name == '55-64':
        return '55-64'
    elif dem_name == '25-34':
     return '25-34'
    elif dem_name == '18-24':
     return '18-24'
    elif dem_name == '45-54':
        return '45-54'
    elif dem_name == '35-44':
        return '35-44'

    elif dem_name == 'male':
        return 'male'
    elif dem_name == 'female':
        return 'female'
    elif dem_name == 'liberal':
        return 'liberal'
    elif dem_name == 'veryconservative':
        return 'very conservative'
    elif dem_name == 'conservative':
        return 'conservative'
    elif dem_name == 'other':
        return 'other'
    elif dem_name == 'veryliberal':
        return 'very liberal'
    elif dem_name == 'moderate':
        return 'moderate'
    elif dem_name == 'separated':
        return 'separated'
    elif dem_name == 'widowed':
        return 'widowed'
    elif dem_name == 'divorced':
        return 'divorced'
    elif dem_name == 'married':
        return 'married'
    elif dem_name == 'living_with_partner':
        return 'living with partner'
    elif dem_name == 'single':
        return 'single'
    elif dem_name == 'infulltimeworkpermanent':
        return 'infull time work permanent'
    elif dem_name == 'retired':
        return 'retired'
    elif dem_name == 'infulltimeworktempcontract':
        return 'infull time work temp contract'
    elif dem_name == 'unemployed':
        return 'unemployed'
    elif dem_name == 'inparttimeworkpermanent':
        return 'inpart time work permanent'
    elif dem_name == 'inpart timeworktempcontract':
        return 'inparttime work temp contract'
    elif dem_name == 'studentonly':
        return 'student only'
    elif dem_name == 'parttimeworkparttimestudent':
        return 'part time work part time student'

    elif dem_name == 'selfemployed':
        return 'self employed'

def converting_demographic_num(dem_name):
    if dem_name == 'prefernotrespond':
        return 'prefer not respond'
    elif dem_name == 'living_with_partner':
        return 'living with partner'

    elif dem_name == 'collegegraduatebsbaorother4yeardegree':
        return 1 #'college graduate bs ba or other 4 year degree'
    elif dem_name == 'technicaltradeorvocationalschoolafterhighschool':
        return 2#'technical trade or vocational school after highschool'
    elif dem_name == 'postgraduatetrainingorprofessionalschoolingaftercollegeegtowardamastersdegreeorphdlawormedicalschool':
        return 3#'post graduate training or professional schooling after college (e.g.toward a masters degree or phd law or medical school'
    elif dem_name == 'highschoolgraduategrade12orgedcertificate':
        return 4#'highschool graduate grade 12 or e.g. dcertificate'
    elif dem_name == 'somecollegeassociatedegreeno4yeardegree':
        return 5#'some college associate degree no 4 year degree'
    elif dem_name == '100001ndash150000':
        return 1#'100001-150000'
    elif dem_name == '40001ndash50000':
        return 2#'40001-50000'
    elif dem_name == '70001ndash100000':
        return 3#'70001-100000'
    elif dem_name == '60001ndash70000':
        return '60001-70000'
    elif dem_name == '150001ormore':
        return '150001 or more'
    elif dem_name == '10000ndash20000':
        return '10000-20000'
    elif dem_name == '50001ndash60000':
        return '50001-60000'
    elif dem_name == '20001ndash30000':
        return '20001-30000'
    elif dem_name == 'under10000':
        return 'under 10000'
    elif dem_name == '30001ndash40000':
        return '30001-40000'
    elif dem_name == 'hispanicorlatino':
        return 'hispanic or latino'
    elif dem_name == 'americanindianoralaskanative':
        return 'american-indian or alaska-native'
    elif dem_name == 'other':
        return 'other'
    elif dem_name == 'asian':
        return 'asian'
    elif dem_name == 'blackorafricanamerican':
        return 'black or african-american'
    elif dem_name == 'prefernotrespond':
        return 'prefer not respond'
    elif dem_name ==  'nativehawaiianorotherpacificislander':
        return 'native hawaiian or other pacificis lander'
    elif dem_name == 'white':
        return 'white'
    elif dem_name == 'unitedstates':
        return 'united states'
    elif dem_name == 'canada':
        return 'canada'
    elif dem_name == 'unitedkingdom':
        return 'united kingdom'
    elif dem_name == 'unitedstates':
        return 'unitedstates'
    elif dem_name == '0.0':
        return '0.0'
    elif dem_name == '1.0':
        return '1.0'
    elif dem_name == '-1.0':
        return '-1.0'
    elif dem_name == '-10.0':
        return '-10.0'
    elif dem_name == '65-74':
        return '65-74'
    elif dem_name == '55-64':
        return '55-64'
    elif dem_name == '25-34':
     return '25-34'
    elif dem_name == '18-24':
     return '18-24'
    elif dem_name == '45-54':
        return '45-54'
    elif dem_name == '35-44':
        return '35-44'

    elif dem_name == 'male':
        return 'male'
    elif dem_name == 'female':
        return 'female'
    elif dem_name == 'liberal':
        return 'liberal'
    elif dem_name == 'veryconservative':
        return 'very conservative'
    elif dem_name == 'conservative':
        return 'conservative'
    elif dem_name == 'other':
        return 'other'
    elif dem_name == 'veryliberal':
        return 'very liberal'
    elif dem_name == 'moderate':
        return 'moderate'
    elif dem_name == 'separated':
        return 'separated'
    elif dem_name == 'widowed':
        return 'widowed'
    elif dem_name == 'divorced':
        return 'divorced'
    elif dem_name == 'married':
        return 'married'
    elif dem_name == 'living_with_partner':
        return 'living with partner'
    elif dem_name == 'single':
        return 'single'
    elif dem_name == 'infulltimeworkpermanent':
        return 'infull time work permanent'
    elif dem_name == 'retired':
        return 'retired'
    elif dem_name == 'infulltimeworktempcontract':
        return 'infull time work temp contract'
    elif dem_name == 'unemployed':
        return 'unemployed'
    elif dem_name == 'inparttimeworkpermanent':
        return 'inpart time work permanent'
    elif dem_name == 'inpart timeworktempcontract':
        return 'inparttime work temp contract'
    elif dem_name == 'studentonly':
        return 'student only'
    elif dem_name == 'parttimeworkparttimestudent':
        return 'part time work part time student'

    elif dem_name == 'selfemployed':
        return 'self employed'

def my_KS_distance(u, v):
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is
    equal to 1 - (u.v / |u||v|).
    """
    # output = 1 - (numpy.dot(u, v) / (
    #             sqrt(numpy.dot(u, u)) * sqrt(numpy.dot(v, v))))
    output = float(scipy.stats.ks_2samp(u, v)[0])
    return output


def m_clusterDemographics(demographicsDict, filename):
    for i in demographicsDict:
        total = sum(demographicsDict[i])
        demographicsDict[i] = [j / float(total) for j in demographicsDict[i]]

    demDist = []
    for i in sorted(demographicsDict.keys()):
        demDist.append(demographicsDict[i])

    X = numpy.array(demDist)

    # # To know the optimum number of clusters K,
    # # Use Silhouette Score (https://en.wikipedia.org/wiki/Silhouette_(clustering))
    #
    fp = open("silhouette_score_" + filename + ".txt", "w")
    fp.write("Number_of_Clusters (K) \\t Silhouette_Score \n")
    for numCluster in range(2, 11):
        kmeans = KMeans(n_clusters=numCluster, n_jobs=10).fit(X)
        # fp.write("{0}\t{1}\n".format(numCluster, metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')))
        fp.write("{0}\t{1}\n".format(numCluster, metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')))
    fp.close()
    #

    # """
    numCluster = 2
    # kclusterer = KMeansClusterer(numCluster, distance= np.mean, repeats=25)
    # kclusterer = KMeansClusterer(numCluster, distance= scipy.stats.ks_2samp, repeats=25)
    # kclusterer = KMeansClusterer(numCluster, distance= my_KS_distance, repeats=10)
    kclusterer = KMeansClusterer(numCluster, distance=my_KS_distance, conv_test=100000, avoid_empty_clusters=True,
                                 repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    clusterMemberDict = {}
    for i in range(numCluster):
        clusterMemberDict[i] = []
        for j in range(len(assigned_clusters)):
            if assigned_clusters[j] == i:
                clusterMemberDict[i].append(sorted(demographicsDict.keys())[j])
    return clusterMemberDict




def normalized_max_min_funct(demographicsDict):
    output = []
    min_value = np.min(demographicsDict)
    max_value = np.max(demographicsDict)
    m = scipy.interpolate.interp1d([min_value, max_value], [-1,1] )
    for el in demographicsDict:
        output.append(m(el))
        # output.append((el - min_value) / float(max_value - min_value))
    return (output)


def normalized_max_min_funct_sikitlearn(demographicsDict):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(demographicsDict)

    return (X_train_minmax)


def rec_value_set(features_init_num=[0] * 3, value=[0, 1]):
    out_lists = []
    if len(features_init_num) == 1:
        for val_e in value:
            out_lists.append([val_e])
        return out_lists
    tmp_out_list = []
    for val in value:
        # tmp_list = []
        tmp_list = [val]
        len_list = len(features_init_num) - 1
        out_lists = rec_value_set(features_init_num=[0] * len_list, value=[0, 1])
        for m_list in out_lists:
            tmp_out_list.append(tmp_list + m_list)
    return tmp_out_list


def initial_manually(num_feat, features_init_num, value):
    initial_list = []
    res_lists = rec_value_set(features_init_num, value)
    num_arr = math.pow(len(value), len(features_init_num))
    for res_list in res_lists:
        tmp_list = [0] * num_feat
        for el in features_init_num:
            tmp_list[el - 1] = res_list[el - 1]
        initial_list.append(tmp_list)

    return initial_list


def clusterDemographics(demographicsDict, filename, number_clusters, initial_list=[], normalization_clustering=''):
    # for i in demographicsDict:
    #     total = sum(demographicsDict[i])
    #     demographicsDict[i] = [j / float(total) for j in demographicsDict[i]]
    #
    # demographicsDict = normalized_max_min_funct(demographicsDict)


    demDist = []
    for i in sorted(demographicsDict.keys()):
        demDist.append(demographicsDict[i])

    if normalization_clustering == 'minamax':
        demDist = normalized_max_min_funct_sikitlearn(demDist)

    X = numpy.array(demDist)

    # To know the optimum number of clusters K,
    # Use Silhouette Score (https://en.wikipedia.org/wiki/Silhouette_(clustering))

    # fp = open("publisher_minmax_initiallist_silhouette_score_"+filename+".txt", "w")
    # # fp = open("publisher_silhouette_score_"+filename+".txt", "w")
    # print('best k for k-means : ' + "publisher_silhouette_score_"+filename+".txt")
    # fp.write("Number_of_Clusters (K) \\t Silhouette_Score \n")
    # for numCluster in range(2, 11):
    #     kmeans = KMeans(n_clusters=numCluster, n_jobs=10).fit(X)
    #     # fp.write("{0}\t{1}\n".format(numCluster, metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')))
    #     fp.write("||{0}||\t{1}||\n".format(numCluster, metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')))
    # fp.close()
    #

    # """
    numCluster = number_clusters
    if len(initial_list) == 0:
        print('initial_list is empty')
        kmeans = KMeans(n_clusters=numCluster, n_init=100, n_jobs=100).fit(X)
        precision = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
        print(precision)
    else:
        kmeans = KMeans(n_clusters=numCluster, n_init=1, init=numpy.array(initial_list), n_jobs=10).fit(X)
        print(metrics.silhouette_score(X, kmeans.labels_, metric='euclidean'))

    clusterMemberDict = {}
    # print(kmeans.cluster_centers_)
    for center in kmeans.cluster_centers_ :
        print('||')
        for cent_i in center:
            print(str(cent_i) + '||')
        # print('\n')
    for i in range(numCluster):
        clusterMemberDict[i] = []
        for j in range(len(kmeans.labels_)):
            if kmeans.labels_[j] == i:
                clusterMemberDict[i].append(sorted(demographicsDict.keys())[j])
    return clusterMemberDict, kmeans.cluster_centers_, precision


def computeDistanceBetweenDistributions(dist1, dist2):
    chiSquareDistance = 0.0
    for i in range(len(dist1)):
        val1 = dist1[i]
        val2 = dist2[i]
        chiSquareDistance += ((val1 - val2) ** 2) / (val1 + val2)
    chiSquareDistance /= 2.0

    return chiSquareDistance


def chi_sqr(dist1, dist2):
    chiSquareDistance = 0.0
    for i in range(len(dist1)):
        val1 = dist1[i]
        val2 = dist2[i]
        if val1 + val2 == 0:
            chiSquareDistance += 0
        else:
            chiSquareDistance += ((val1 - val2) ** 2) / (val1 + val2)
    chiSquareDistance /= 2.0

    return chiSquareDistance


class JSD(object):
    def __init__(self):
        self.log2 = log(2)

    def KL_divergence(self, p, q):
        """ Compute KL divergence of two vectors, K(p || q)."""
        return sum(p[x] * log((p[x]) / (q[x])) for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)

    def Jensen_Shannon_divergence(self, p, q):
        """ Returns the Jensen-Shannon divergence. """
        self.JSD = 0.0
        weight = 0.5
        average = zeros(len(p))  #Average
        for x in range(len(p)):
            average[x] = weight * p[x] + (1 - weight) * q[x]
            self.JSD = (weight * self.KL_divergence(array(p), average)) + (
            (1 - weight) * self.KL_divergence(array(q), average))
        return 1 - (self.JSD / sqrt(2 * self.log2))


def preparing_output_replys_for_wiki(inp_txt, tweet_source):
    if len(inp_txt) > 2:
        output_str = '[[https://twitter.com/' + inp_txt.replace('\n', " ").split('<<**>>')[1] + '|' \
                     + inp_txt.replace('\n', "").split('<<**>>')[1] + ']]:' + '[[https://twitter.com/' \
                     + tweet_source + '/status/' \
                     + inp_txt.replace('\n', "").split('<<**>>')[2] + '|*' \
                     + inp_txt.replace('\n', "").split('<<**>>')[3].replace('\n', "") + ']]'
    else:
        output_str = ''
    return output_str


def frange(start, stop, step):
    i = start
    res_list = []
    while i < stop:
        res_list.append(i)
        i += step
    return res_list[:]


def plot_diff_dist(inp_list1, inp_list2, label_1, label_2, name):
    if 'controversial' in name:
        print"test"
    mplpl.clf()
    mplpl.rc('xtick', labelsize='medium')
    mplpl.rc('ytick', labelsize='medium')
    mplpl.rc('xtick.major', size=3, pad=3)
    mplpl.rc('xtick.minor', size=2, pad=3)
    mplpl.rc('legend', fontsize='small')  # for tweet_id in df_num_retweet_sorted['tweet_id']:

    # target_users_list = [factual_tweet_list, opinion_tweet_list]#, controversial_tweet_list,noncontroversial_tweet_list,
    #informative_tweet_list, noninformative_tweet_list]
    if len(inp_list1) > 1:
        if 'KL' not in name and 'js' not in name:
            counts, bin_edges = np.histogram(inp_list1, bins=np.arange(0, 1, 0.05), normed=True)
        else:
            counts, bin_edges = np.histogram(inp_list1, bins=np.arange(0, max(inp_list1), max(inp_list1) / 20),
                                             normed=True)

        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1]
        Plab.plot(bin_edges[0:-1], cdf, color='r', label=label_1, lw=4)

        if 'KL' not in name and 'js' not in name:
            counts, bin_edges = np.histogram(inp_list2, bins=np.arange(0, 1, 0.05), normed=True)
        else:
            counts, bin_edges = np.histogram(inp_list2, bins=np.arange(0, max(inp_list2), max(inp_list2) / 20),
                                             normed=True)
        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1]

        Plab.plot(bin_edges[0:-1], cdf, color='b', label=label_2, lw=4)

        # mplpl.hist(inp_list1, normed=1, histtype='step', cumulative=True, color='r',label=label_1)
        # mplpl.hist(inp_list2, normed=1, histtype='step', cumulative=True, color='b', label=label_2)

        mplpl.xlabel('', fontsize='large')
        mplpl.ylabel('CDF', fontsize='large')
        mplpl.title('', fontsize='large')

        mplpl.legend(loc="upper left")
        mplpl.ylim([0, 1])
        if 'KL' not in name and 'js' not in name:
            mplpl.xlim([0, 1])
        else:
            mplpl.xlim(0, max(bin_edges))

        pp = remotedir + '/new_data_set/fig/distribution_fig/'
        if not os.path.exists(pp): os.makedirs(pp)

        mplpl.savefig(pp + name, format='png')



def replace_outliers_mean(n_dist):
    out = []
    tmp_dict = collections.defaultdict()
    i = 0
    for el in n_dist:
        tmp_dict[i] = el
        i += 1

    previous_v = 0
    dist_mean = np.mean(n_dist)
    Thresh = 5
    count = 0
    previous_el = 0
    for dist_el in range(0, len(n_dist)):
        if tmp_dict[dist_el] / dist_mean > Thresh:
            tmp_dict[dist_el] = np.max([previous_v, dist_mean])
        out.append(tmp_dict[dist_el])
        previous_v = tmp_dict[dist_el]
    return out




def remove_outliers_dist(n_dist):
    out = []
    tmp_dict = collections.defaultdict()

    k_top = 2

    i = 0
    for el in n_dist:
        tmp_dict[i] = el
        i += 1
    i = 0

    dist_sorted = sorted(tmp_dict, key=tmp_dict.get, reverse=True)
    dist_mean = np.mean(n_dist)
    count = 0
    previous_el = 0
    dist_replace_val = np.max([tmp_dict[x] for x in dist_sorted[k_top:]])
    for dist_el in dist_sorted:
        tmp_dict[dist_el] = dist_replace_val
        if count >= k_top:
            break
        count += 1

    for dist_el in range(0, len(n_dist)):
        out.append(tmp_dict[dist_el])

    return out


def AMT_tweet_cont_fact_scores():
    processing_data = 'media_sites'

    local_dir_saving = '/preprocess/new_data_set/'
    remotedir = '/home/babaei/Desktop/SVN/streaming_api/with_tweepy/fake_news_data'

    pp = remotedir + local_dir_saving
    inpFtmp = open(pp + 'wiki_output_diff_ks_with_results.txt', 'r')
    outputFile = open(pp + 'set_0', 'w')
    count = 0
    cc = 0
    out_list = []
    ks_dict_tweet = collections.defaultdict()
    tweet_source_dict = collections.defaultdict()
    tweet_text_dict = collections.defaultdict()
    AMT_tweet_id_list = []
    for line in inpFtmp:
        try:
            if '==' in line or '||Index||tweet||tweet_id||source||' in line:
                # outputFile.write(line)
                count += 1
                continue
            if count < 7:
                continue
            cc += 1
            line = line.replace('\n', '')
            line_splt = line.split('||')
            tweet = line_splt[2]
            tweet_txt = tweet.split('<<BR>>')[0]
            tweet_link = tweet.split('<<BR>>')[1]
            tweet_link = tweet_link.replace('[[', '')
            tweet_link = tweet_link.replace(']]', '')
            tweet_link = tweet_link.split('|')[0]
            tweet_id = int(line_splt[3])
            source = tweet_link.split('/')[3].replace('@', '')
            ks_value = float(line_splt[6])

            tmp_list = [tweet_id, source, tweet_txt, tweet_link]
            ks_dict_tweet[tweet_id] = ks_value
            tweet_source_dict[tweet_id] = source
            tweet_text_dict[tweet_id] = tweet_txt
            AMT_tweet_id_list.append(tweet_id)
            # out_list.append(tmp_list)
        except:
            continue

    tweet_sorted_ks = sorted(ks_dict_tweet, key=ks_dict_tweet.get, reverse=False)
    tweets_ks_values_sorted = []
    for tweet_e in tweet_sorted_ks:
        tweets_ks_values_sorted.append(np.round(ks_dict_tweet[tweet_e], 3))

    # valid users dict
    Vuser_dict = collections.defaultdict(int)
    # query1 = "SELECT text from all_expert_tweets_parsed WHERE tweet_id = %d"%tweet_id
    query1 = "select workerid, count(*) from mturk_crowd_signals_tweet_response_1 group by workerid;"

    cursor.execute(query1)
    res_all = cursor.fetchall()
    for el in res_all:
        if el[1] == 21:
            Vuser_dict[el[0]] = 1

    query2 = "select workerid, tweet_id, ra, rb, rc, text from mturk_crowd_signals_tweet_response_1;"

    cursor.execute(query2)
    res_all = cursor.fetchall()

    query3 = "select workerid, ra from mturk_crowd_signals_tweet_response_1 where tweet_id=1;"

    cursor.execute(query3)
    res_leaning = cursor.fetchall()
    leaning_dict = collections.defaultdict()
    for el in res_leaning:
        leaning_dict[el[0]] = el[1]

    workerid_list = collections.defaultdict(list)
    tweetid_list = collections.defaultdict(list)
    ra_list = collections.defaultdict(list)
    rb_list = collections.defaultdict(list)
    rc_list = collections.defaultdict(list)
    txt_list = collections.defaultdict(list)
    index_list = collections.defaultdict(list)

    workerid_list_all = collections.defaultdict(list)
    tweetid_list_all = collections.defaultdict(list)
    ra_list_all = collections.defaultdict(list)
    rb_list_all = collections.defaultdict(list)
    rc_list_all = collections.defaultdict(list)
    txt_list_all = collections.defaultdict(list)
    index_list_all = collections.defaultdict(list)

    workerid_list_m = []
    tweetid_list_m = []
    ra_list_m = []
    rb_list_m = []
    rc_list_m = []
    txt_list_m = []
    index_list_m = []
    user_leaning_list_m = []
    count = 0
    # df_category = [0,1,2,3]*[1 as with author and 2 without author]
    category_author_laning = ['withoutauthor_lean_1', 'withauthor_lean_1', 'withourauthor_lean2', 'withauthor_lean2',
                              'withoutauthor_lean3', 'withauthor_lean3', 'withoutauthor_lean4', 'withauthor_lean4']
    for el in res_all:
        if el[0] in Vuser_dict:
            workerid = int(el[0])
            tweetid = int(el[1])
            ra = int(el[2])
            rb = int(el[3])
            rc = int(el[4])
            if tweetid != 1:
                ra = ra % 2
                rb = rb % 2
                rc = rc % 2
            txt = el[5]

            user_leaning = leaning_dict[el[0]]

            if workerid % 2 == 1:
                var_sum = 1
            else:
                var_sum = 0
            user_leaning_1 = user_leaning * 2 + var_sum
            workerid_list[user_leaning_1].append(workerid)
            tweetid_list[user_leaning_1].append(tweetid)
            ra_list[user_leaning_1].append(ra)
            rb_list[user_leaning_1].append(rb)
            rc_list[user_leaning_1].append(rc)
            txt_list[user_leaning_1].append(txt)
            index_list[user_leaning_1].append(count)

            workerid_list_all[user_leaning].append(workerid)
            tweetid_list_all[user_leaning].append(tweetid)
            ra_list_all[user_leaning].append(ra)
            rb_list_all[user_leaning].append(rb)
            rc_list_all[user_leaning].append(rc)
            txt_list_all[user_leaning].append(txt)
            index_list_all[user_leaning].append(count)

            workerid_list_m.append(workerid)
            tweetid_list_m.append(tweetid)
            ra_list_m.append(ra)
            rb_list_m.append(rb)
            rc_list_m.append(rc)
            txt_list_m.append(txt)
            index_list_m.append(count)
            user_leaning_list_m.append(user_leaning)
            count += 1

    index_list_m = range(len(workerid_list_m))
    df = pd.DataFrame({'workerid': Series(workerid_list_m, index=index_list_m),
                       'tweetid': Series(tweetid_list_m, index=index_list_m),
                       'ra': Series(ra_list_m, index=index_list_m),
                       'rb': Series(rb_list_m, index=index_list_m),
                       'rc': Series(rc_list_m, index=index_list_m),
                       'text': Series(txt_list_m, index=index_list_m),
                       'leaning': Series(user_leaning_list_m, index=index_list_m)})

    df_cat = collections.defaultdict()
    for i in range(0, 8):
        index_list[i] = range(len(workerid_list[i]))
        df_cat[i] = pd.DataFrame({'workerid': Series(workerid_list[i], index=index_list[i]),
                                  'tweetid': Series(tweetid_list[i], index=index_list[i]),
                                  'ra': Series(ra_list[i], index=index_list[i]),
                                  'rb': Series(rb_list[i], index=index_list[i]),
                                  'rc': Series(rc_list[i], index=index_list[i]),
                                  'text': Series(txt_list[i], index=index_list[i])})
        # 'leaning': Series(user_leaning_list[i], index=index_list_m)})

    df_all_aut_cat = collections.defaultdict()
    for i in range(1, 4):
        index_list_all[i] = range(len(workerid_list_all[i]))
        df_all_aut_cat[i] = pd.DataFrame({'workerid': Series(workerid_list_all[i], index=index_list_all[i]),
                                          'tweetid': Series(tweetid_list_all[i], index=index_list_all[i]),
                                          'ra': Series(ra_list_all[i], index=index_list_all[i]),
                                          'rb': Series(rb_list_all[i], index=index_list_all[i]),
                                          'rc': Series(rc_list_all[i], index=index_list_all[i]),
                                          'text': Series(txt_list_all[i], index=index_list_all[i])})

    with_author = 0;
    without_author = 0;
    workerid_set = set(df['workerid'])
    with_author_list = []
    without_author_list = []
    for wid in workerid_set:
        if wid % 2 == 1:
            with_author += 1
            with_author_list.append(wid)
        else:
            without_author += 1
            without_author_list.append(wid)
    print('number of workers who sees the surveys without author is : ' + str(without_author))
    print('number of workers who sees the surveys with author is : ' + str(with_author))

    df['ra'][df['ra'] == 2] = 0
    df['rb'][df['rb'] == 2] = 0
    df['rc'][df['rc'] == 2] = 0

    df_author = df[df['workerid'] % 2 == 1]
    df_w_author = df[df['workerid'] % 2 == 0]

    groupby_ftr = 'tweetid'
    grouped = df.groupby(groupby_ftr, sort=False)
    sum_feat = df.groupby(groupby_ftr).sum()

    category_author_laning = ['withoutauthor_dem', 'withauthor_dem', 'withourauthor_rep', 'withauthor_rep',
                              'withoutauthor_neut', 'withauthor_neut', 'withoutauthor_dont_know',
                              'withauthor_dont_know']

    tweet_id_ra_dict = collections.defaultdict(float)
    tweet_id_rb_dict = collections.defaultdict(float)
    tweet_id_rc_dict = collections.defaultdict(float)

    groupby_ftr = 'tweetid'
    grouped = df.groupby(groupby_ftr, sort=False)
    sum_feat = df.groupby(groupby_ftr).sum()
    count_feat = df.groupby(groupby_ftr).count()
    outF = open(remotedir + local_dir_saving + 'amt_output_wiki_clustering.txt', 'w')
    for tweet_id_e in tweet_sorted_ks:
        relative_ra = float(sum_feat['ra'][tweet_id_e]) / count_feat['ra'][tweet_id_e]
        relative_rb = float(sum_feat['rb'][tweet_id_e]) / count_feat['rb'][tweet_id_e]
        relative_rc = float(sum_feat['rc'][tweet_id_e]) / count_feat['rc'][tweet_id_e]
        tweet_id_ra_dict[tweet_id_e] = relative_ra
        tweet_id_rb_dict[tweet_id_e] = relative_rb
        tweet_id_rc_dict[tweet_id_e] = relative_rc

        ##############################
    author = 0
    ##############################


    tweet_id_ra_dict_w = collections.defaultdict(float)
    tweet_id_rb_dict_w = collections.defaultdict(float)
    tweet_id_rc_dict_w = collections.defaultdict(float)

    tweet_id_ra_dict_nw = collections.defaultdict(float)
    tweet_id_rb_dict_nw = collections.defaultdict(float)
    tweet_id_rc_dict_nw = collections.defaultdict(float)

    groupby_ftr = 'tweetid'
    grouped = df_author.groupby(groupby_ftr, sort=False)
    sum_feat = df_author.groupby(groupby_ftr).sum()
    count_feat = df_author.groupby(groupby_ftr).count()
    for tweet_id_e in tweet_sorted_ks:
        tweet_id_ra_dict_w[tweet_id_e] = float(sum_feat['ra'][tweet_id_e]) / count_feat['ra'][tweet_id_e]
        tweet_id_rb_dict_w[tweet_id_e] = float(sum_feat['rb'][tweet_id_e]) / count_feat['rb'][tweet_id_e]
        tweet_id_rc_dict_w[tweet_id_e] = float(sum_feat['rc'][tweet_id_e]) / count_feat['rc'][tweet_id_e]

    groupby_ftr = 'tweetid'
    grouped = df_w_author.groupby(groupby_ftr, sort=False)
    sum_feat = df_w_author.groupby(groupby_ftr).sum()
    count_feat = df_w_author.groupby(groupby_ftr).count()

    for tweet_id_e in tweet_sorted_ks:
        tweet_id_ra_dict_nw[tweet_id_e] = float(sum_feat['ra'][tweet_id_e]) / count_feat['ra'][tweet_id_e]
        tweet_id_rb_dict_nw[tweet_id_e] = float(sum_feat['rb'][tweet_id_e]) / count_feat['rb'][tweet_id_e]
        tweet_id_rc_dict_nw[tweet_id_e] = float(sum_feat['rc'][tweet_id_e]) / count_feat['rc'][tweet_id_e]

    return [tweet_id_ra_dict, tweet_id_rb_dict, tweet_id_rc_dict, tweet_id_ra_dict_w, tweet_id_rb_dict_w,
            tweet_id_rc_dict_w, tweet_id_ra_dict_nw, tweet_id_rb_dict_nw, tweet_id_rc_dict_nw, AMT_tweet_id_list]

# ================================ main

if __name__ == '__main__':
    global globflag
    global out_list
    out_list = []
    # currently we are testing these arguments
    # for simplicity and accesibility we list them here
    if len(sys.argv) == 1:
        # sys.argv += ["-t", "AMT_dataset_reliable_news_processing_all_dataset_weighted"]
        # sys.argv += ["-t", "AMT_dataset_reliable_news_processing_all_dataset_leaning"]

        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_PDF_CDF_MPB_fig"]
        # sys.argv += ["-t", "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_true-false_accuracy"]
        # sys.argv += ["-t", "AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_mpb_cdf_toghether"]
        # sys.argv += ["-t","AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_apb_cdf_toghether"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_PDF_CDF_APB_fig"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_true-false(gt-pt)_news_ktop_nptl_scatter_fig1"]
        # sys.argv += ["-t","AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_disp_cdf_toghether"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_PDF_CDF_disp_fig"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_DISP_TPB_scatter_fig"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_time_analysis"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_time_fig"]
        # sys.argv += ["-t", "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_true-false(gt-pt)_news_ktop_nptl_scatter"]
        # sys.argv += ["-t", "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_labeld_news_ktop_nptl_together"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_comparing_ssi_amt"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_effect_10claims"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_dist_judgement"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_creating_features_prediction_old"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_creating_features_prediction"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_writing_wiki_tweets_dist_judgments"]
        # sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_writing_wiki_workers_dist_judgments"]
        # sys.argv += ["-t",
        #              "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_writing_wiki_workers_dist_judgments_diff_dataset"]

        # sys.argv += ["-t","AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_tpb_FPB_FNB_cdf_toghether"]
        # sys.argv += ["-t",
        #              "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_analysis_plan"]
        #
        sys.argv += ["-t","AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_analysis_plan_individual_tweet"]


        # sys.argv += ["-t","scratch_bounus"]
        print "The list od arguments modified to include:", sys.argv[1:]


    parser = argparse.ArgumentParser()
    parser.add_argument('inputpaths', nargs='*',
                        default=glob.glob("tweets-bilal/20*.gz"),
                        help='paths to gzipped files with tweets in json format')
    parser.add_argument('-t',
                        default="extract-userinfo-usercrawl",
                        help='task name')
    parser.add_argument('-c',
                        default="set3-en",
                        # default="set2-all",
                        # default="nov3m",
                        # default="all",
                        help='crawlname')
    parser.add_argument('-ct',
                        # default="tweets-retweeters",
                        default="tweets-expall",
                        help='crawltype')
    parser.add_argument('-usn',
                        default="set3-en",
                        help="usersetnm")
    parser.add_argument('-muzzled',
                        default=False,
                        action='store_true')
    parser.add_argument('-nj',
                        default=None,
                        help='n_jobs_remote')
    parser.add_argument('-pn',
                        default="before-feb2016-1m",
                        help='periodname')
    parser.add_argument('-f',
                        default="0",
                        help='num_followers_bin')

    args = parser.parse_args()
    crawlname = os.path.basename(args.c)
    crawltype = os.path.basename(args.ct)
    usersetnm = os.path.basename(args.usn)
    periodname = os.path.basename(args.pn)
    f_g_bin = int(args.f)
    # print(f_g_bin)
    inputpaths = args.inputpaths

    localdir, remotedir = tweetstxt_basic.get_dirs()

    # if tweetstxt_basic.is_local_machine():
    # 	nrows = 1e5
    # 	nsamples = int(1e4)
    # 	args.gs = True
    # else:
    # 	nrows = None
    # 	nsamples = int(1e5)
    # 	nsamples = int(2e4)

    conn = pgsql.connect("host='postgresql00.mpi-sws.org' dbname='twitter_data' user='twitter_data' password='twitter@mpi'")
    cursor = conn.cursor()





    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted":

        weighted=False
        # dataset = 'snopes'
        # dataset = 'snopes_2'
        # dataset = 'snopes_incentive_10'
        # dataset = 'snopes_ssi'
        # dataset = 'snopes_incentive'
        # dataset = 'snopes_incentive_notimer'
        dataset = 'snopes_noincentive_timer'
        # dataset = 'mia'
        # dataset = 'politifact'
        # dataset = 'snopes_nonpol'
        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []



        if dataset=='mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'


            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                     + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets','r')



            exp1_list = sample_tweets_exp1


            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor= {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id)<100060:
                    tweet_lable_dict[tweet_id]='rumor'
                else:
                    tweet_lable_dict[tweet_id]='non-rumor'
                # if int(tweet_id) in [100012, 100016, 100053, 100038, 100048]:


        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        if dataset == 'snopes' or dataset == 'snopes_ssi' or dataset=='snopes_incentive'or dataset=='snopes_2'or dataset=='snopes_incentive_10'or dataset=='snopes_noincentive_timer'or dataset=='snopes_incentive_notimer':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable


        if dataset == 'snopes_nonpol':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'


            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            print(inp_all)
            source_dict = {}
            text_dict = {}
            date_dict = {}
            # outF = open(remotedir + 'politifact_last_100_news.txt', 'w')
            # F = open(remotedir + 'snopes_latest_20_news_per_lable_non_politics.txt', 'r')


            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable


        # run = 'plot'
        # run = 'analysis'
        run = 'second-analysis'
        exp1_list = sample_tweets_exp1

        if run == 'analysis':
            experiment = 1
            # experiment = 2
            # experiment = 3

            dem_list = []
            rep_list = []
            neut_list = []
            # valid users dict
            Vuser_dict = collections.defaultdict(int)
            if dataset=='snopes':
                query1 = "select workerid, count(*) from mturk_sp_claim_response_exp1_"+str(experiment)+"_recovery group by workerid;"
            elif dataset == 'snopes_2':
                query1 = "select workerid, count(*) from mturk_sp_claim_response_exp2_" + str(
                    experiment) + " group by workerid;"
            elif dataset == 'snopes_incentive':
                query1 = "select workerid, count(*) from mturk_sp_claim_incentive_response_exp1_" + str(experiment) + " group by workerid;"
            elif dataset == 'snopes_incentive_notimer':
                query1 = "select workerid, count(*) from mturk_sp_claim_incentive_notimer_response_exp1_" + str(
                    experiment) + " group by workerid;"
            elif dataset == 'snopes_noincentive_timer':
                query1 = "select workerid, count(*) from mturk_sp_claim_noincentive_timer_response_exp1_" + str(
                    experiment) + " group by workerid;"
            elif dataset == 'snopes_incentive_10':
                query1 = "select workerid, count(*) from mturk_sp_claim_incentive_10_response_exp1_" + str(
                    experiment) + " group by workerid;"
            elif dataset == 'snopes_ssi':
                # query1 = "select workerid, count(*) from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_recovery_full group by workerid;"
                query1 = "select workerid, count(*) from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_new_recovery group by workerid;"
            elif dataset=='snopes_nonpol':
                query1 = "select workerid, count(*) from mturk_sp_claim_nonpol_response_exp1_recovery group by workerid;"
            elif dataset=='politifact':
                query1 = "select workerid, count(*) from mturk_pf_claim_response_exp1_" + str(experiment) + "_recovery group by workerid;"
            elif dataset=='mia':
                query1 = "select workerid, count(*) from mturk_m_claim_response_exp1_recovery group by workerid;"

            cursor.execute(query1)
            res_exp2 = cursor.fetchall()
            for el in res_exp2:
                if dataset=='snopes_ssi':
                    if experiment==2:
                        if el[0]==4:
                            continue
                    if experiment==3:
                        if el[0]==3:
                            continue
                if el[1] == 53:
                    Vuser_dict[1000*experiment + el[0]] = 1

            if dataset=='snopes':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_response_exp1_"+str(experiment)+"_recovery;"
            elif dataset=='snopes_2':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_response_exp2_"+str(experiment)+";"
            elif dataset == 'snopes_incentive':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_incentive_response_exp1_" + str(experiment) + ";"
            elif dataset == 'snopes_incentive_notimer':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_incentive_notimer_response_exp1_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_noincentive_timer':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_noincentive_timer_response_exp1_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_incentive_10':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_incentive_10_response_exp1_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_ssi':
                # query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_recovery_full;"
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_new_recovery;"
            elif dataset == 'snopes_nonpol':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_nonpol_response_exp1_recovery;"
            elif dataset=='politifact':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_pf_claim_response_exp1_"+str(experiment)+"_recovery;"

            elif dataset == 'mia':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_m_claim_response_exp1_recovery;"




            cursor.execute(query2)
            res_exp2 = cursor.fetchall()

            res_exp1_l = []
            for el in res_exp2:
                if Vuser_dict[1000*experiment + el[0]]==1:
                    res_exp1_l.append((1000*experiment + el[0], el[1], el[2], el[3], el[4], el[5], el[6]))

            if dataset=='snopes':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_sp_claim_demographics_"+str(experiment)+"_recovery"
            elif dataset == 'snopes_2':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_demographics2_" + str(
                    experiment) + ""
            elif dataset == 'snopes_incentive':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_incentive_demographics_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_incentive_notimer':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_incentive_notimer_demographics_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_noincentive_timer':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_noincentive_timer_demographics_" + str(
                    experiment) + ";"


            elif dataset == 'snopes_incentive_10':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_incentive_10_demographics_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_ssi':
                # query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                #          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                #          " demographic_political_view, demographic_race," \
                #          " demographic_marital_status from mturk_sp_claim_ssi_demographics" + str(
                #     experiment) + "_recovery_full"
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_ssi_demographics" + str(
                    experiment) + "_new_recovery"
            elif dataset == 'snopes_nonpol':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_sp_claim_nonpol_demographics1_recovery"


            elif dataset=='politifact':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_pf_claim_demographics_"+str(experiment)+"_recovery"

            elif dataset == 'mia':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_m_claim_demographics_recovery"


            cursor.execute(query3)
            res_leaning = cursor.fetchall()
            leaning_dict = collections.defaultdict()
            dem_l = [];
            rep_l = [];
            neut_l = []

            w_nationality = {};w_residence = {};w_gender = {};w_age = {}
            w_degree = {};w_employment = {};w_income = {};w_political_view = {}
            w_race = {};w_marital_status = {}

            workerid_list_m  = []
            w_nationality_l = [];w_residence_l = [];w_gender_l = []
            w_age_l = [];w_degree_l = [];w_employment_l = []
            w_income_l = [];w_political_view_l = [];w_race_l = [];w_marital_status_l = []
            tweet_id_l = []; ra_l=[]; time_l=[];txt_list_m=[]
            for el in res_leaning:
                w_nationality[1000*experiment+el[0]] = el[1]
                w_residence[1000*experiment + el[0]] = el[2]
                w_gender[1000*experiment + el[0]] = el[3]
                w_age[1000*experiment + el[0]] = el[4]
                w_degree[1000*experiment + el[0]] = el[5]
                w_employment[1000*experiment + el[0]] = el[6]
                w_income[1000*experiment + el[0]] = el[7]
                w_political_view[1000*experiment + el[0]] = el[8]
                w_race[1000*experiment + el[0]] = el[9]
                w_marital_status[1000*experiment + el[0]] = el[10]


                workerid_list_m.append(el[0]+1000*experiment)
                w_nationality_l.append(el[1])
                w_residence_l.append(el[2])
                w_gender_l.append(el[3])
                w_age_l.append(el[4])
                w_degree_l.append(el[5])
                w_employment_l.append(el[6])
                w_income_l.append(el[7])
                w_political_view_l.append(el[8])
                w_race_l.append(el[9])
                w_marital_status_l.append(el[10])

            index_list_m = range(len(w_nationality_l))
            df_w = pd.DataFrame({'worker_id': Series(workerid_list_m, index=index_list_m),
                                'nationality': Series(w_nationality_l, index=index_list_m),
                                'residence': Series(w_residence_l, index=index_list_m),
                                'gender': Series(w_gender_l, index=index_list_m),
                                'age': Series(w_age_l, index=index_list_m),
                                'degree': Series(w_degree_l, index=index_list_m),
                                'employment': Series(w_employment_l, index=index_list_m),
                                'income': Series(w_income_l, index=index_list_m),
                                'political_view': Series(w_political_view_l, index=index_list_m),
                                'race': Series(w_race_l, index=index_list_m),
                                'marital_status': Series(w_marital_status_l, index=index_list_m),})

            workerid_list_m  = []
            w_nationality_l = [];w_residence_l = [];w_gender_l = []
            w_age_l = [];w_degree_l = [];w_employment_l = []
            w_income_l = [];w_political_view_l = [];w_race_l = [];w_marital_status_l = []
            tweet_id_l = []; ra_l=[]; time_l=[];txt_list_m=[];ra_gt_l=[]


            if dataset=='snopes':
                df_w.to_csv(remotedir  +'worker_amt_answers_sp_claims_exp'+str(experiment)+'.csv',
                          columns=df_w.columns, sep="\t", index=False)
            elif dataset=='snopes_2':
                df_w.to_csv(remotedir  +'worker_amt_answers_sp2_claims_exp'+str(experiment)+'.csv',
                          columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_incentive':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_incentive_claims_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_incentive_notimer':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_incentive_notimer_claims_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)

            elif dataset == 'snopes_noincentive_timer':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_noincentive_timer_claims_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)

            elif dataset == 'snopes_incentive_10':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_incentive_10_claims_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_ssi':
                # df_w.to_csv(remotedir + 'worker_amt_answers_sp_ssi_claims_exp' + str(experiment) + '.csv',
                #             columns=df_w.columns, sep="\t", index=False)
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_ssi_claims_exp' + str(experiment) + '_new.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_nonpol':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_claims_nonpol_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset=='politifact':
                df_w.to_csv(remotedir  +'worker_amt_answers_pf_claims_exp'+str(experiment)+'.csv',
                          columns=df_w.columns, sep="\t", index=False)

            elif dataset == 'mia':
                df_w.to_csv(remotedir  +'worker_amt_answers_mia_claims_exp1.csv',
                          columns=df_w.columns, sep="\t", index=False)




            outF = open(remotedir + 'amt_st.txt' , 'w')
            for dem_att in df_w.columns:
                outF.write('=== ' +  dem_att +' ===\n')
                sum_feat = df_w.groupby(dem_att).sum()
                count_feat = df_w.groupby(dem_att).count()
                grouped = df_w.groupby(dem_att)

                for key in grouped.groups.keys():
                    outF.write('|| ' + str(key))
                outF.write('||\n')
                for key in grouped.groups.keys():
                    out_l = grouped.groups[key]
                    outF.write('|| ' + str(len(out_l)))
                outF.write('||\n')
            outF.close()
            # workerid, tweet_id, ra, rb, rc, at, text

            for el in res_exp1_l:
                if el[1]==1:
                    continue
                tweet_id_l.append(el[1])
                ra_l.append(el[2])
                time_l.append(el[5])
                txt_list_m.append(tweet_txt_dict[el[1]])
                ra_gt_l.append(tweet_lable_dict[el[1]])

                workerid_list_m.append(el[0])
                w_nationality_l.append(w_nationality[el[0]])
                w_residence_l.append(w_residence[el[0]])
                w_gender_l.append(w_gender[el[0]])
                w_age_l.append(w_age[el[0]])
                w_degree_l.append(w_degree[el[0]])
                w_employment_l.append(w_employment[el[0]])
                w_income_l.append(w_income[el[0]])
                w_political_view_l.append(w_political_view[el[0]])
                w_race_l.append(w_race[el[0]])
                w_marital_status_l.append(w_marital_status[el[0]])

            index_list_m = range(len(w_nationality_l))
            df = pd.DataFrame({'worker_id': Series(workerid_list_m, index=index_list_m),
                               'tweet_id': Series(tweet_id_l, index=index_list_m),
                               'ra': Series(ra_l, index=index_list_m),
                               'ra_gt': Series(ra_gt_l, index=index_list_m),
                               'time': Series(time_l, index=index_list_m),
                               'text': Series(txt_list_m, index=index_list_m),
                                'nationality': Series(w_nationality_l, index=index_list_m),
                                'residence': Series(w_residence_l, index=index_list_m),
                                'gender': Series(w_gender_l, index=index_list_m),
                                'age': Series(w_age_l, index=index_list_m),
                                'degree': Series(w_degree_l, index=index_list_m),
                                'employment': Series(w_employment_l, index=index_list_m),
                                'income': Series(w_income_l, index=index_list_m),
                                'political_view': Series(w_political_view_l, index=index_list_m),
                                'race': Series(w_race_l, index=index_list_m),
                                'marital_status': Series(w_marital_status_l, index=index_list_m),})

            worker_id_list = []

            print(len(df))

            if dataset == 'snopes':
                df.to_csv(remotedir + 'amt_answers_sp_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_2':
                df.to_csv(remotedir + 'amt_answers_sp_2_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_incentive':
                df.to_csv(remotedir + 'amt_answers_sp_incentive_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_incentive_notimer':
                df.to_csv(remotedir + 'amt_answers_sp_incentive_notimer_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)

            elif dataset == 'snopes_noincentive_timer':
                df.to_csv(remotedir + 'amt_answers_sp_noincentive_timer_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)

            elif dataset == 'snopes_incentive_10':
                df.to_csv(remotedir + 'amt_answers_sp_incentive_10_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_ssi':
                # df.to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(experiment) + '.csv',
                #           columns=df.columns, sep="\t", index=False)
                df.to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(experiment) + '_new.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_nonpol':
                df.to_csv(remotedir + 'amt_answers_sp_claims_nonpol_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'politifact':
                df.to_csv(remotedir + 'amt_answers_pf_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)

            elif dataset == 'mia':
                df.to_csv(remotedir + 'amt_answers_mia_claims_exp.csv',
                          columns=df.columns, sep="\t", index=False)




        else:

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            if dataset == 'snopes':
                for i in range(1,4):
                    inp1 = remotedir  +'amt_answers_sp_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")


            elif dataset == 'snopes_2':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_2_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp2_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'snopes_incentive':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_incentive_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_incentive_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'snopes_incentive_notimer':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_incentive_notimer_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_incentive_notimer_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'snopes_noincentive_timer':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_noincentive_timer_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_noincentive_timer_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'snopes_incentive_10':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_incentive_10_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_incentive_10_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'snopes_ssi':
                # for i in range(1,4):
                #     inp1 = remotedir  +'amt_answers_sp_ssi_claims_exp'+str(i)+'.csv'
                #     inp1_w = remotedir  +'worker_amt_answers_sp_claims_ssi_exp'+str(i)+'.csv'
                #     df[i] = pd.read_csv(inp1, sep="\t")
                #     df_w[i] = pd.read_csv(inp1_w, sep="\t")
                for i in range(1, 2):
                    inp1 = remotedir + 'amt_answers_sp_ssi_claims_exp' + str(i) + '_new.csv'
                    inp1_w = remotedir + 'worker_amt_answers_sp_ssi_claims_exp' + str(i) + '_new.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'snopes_nonpol':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_claims_nonpol_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_claims_nonpol_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")


            elif dataset == 'politifact':
                for i in range(1, 4):
                    inp1 = remotedir + 'amt_answers_pf_claims_exp' + str(i) + '.csv'
                    inp1_w = remotedir + 'worker_amt_answers_pf_claims_exp' + str(i) + '.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'mia':
                for i in range(1, 2):
                    inp1 = remotedir + 'amt_answers_mia_claims_exp.csv'
                    inp1_w = remotedir + 'worker_amt_answers_mia_claims_exp1.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            mia_rum_t_l = []
            mia_nrum_t_l = []
            if dataset=='politifact' or dataset=='snopes' or dataset=='snopes_ssi':
                ind_l = [1,2,3]
            else:
                ind_l = [1]
            for ind in ind_l:

                df[ind].loc[:,'rel_v'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:,'rel_v_b'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'rel_gt_v'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'delta_time'] = df[ind]['tweet_id'] * 0.0

                df[ind].loc[:, 'err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'err_b'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'vote'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'gull'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'cyn'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'susc'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'acc'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'leaning'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:,'delta_time'] = df[ind]['tweet_id'] * 0.0
                tweet_rel_dict = collections.defaultdict(list)
                tweet_intst_dict = collections.defaultdict(list)
                rel_4 = 0
                for index in df[ind].index.tolist():
                    tweet_id = df[ind]['tweet_id'][index]
                    if tweet_id == 1:
                        continue

                    ra = df[ind]['ra'][index]
                    ra_gt = df[ind]['ra_gt'][index]

                    if ra==1:
                        rel = -3
                    elif ra==2:
                        rel=-2
                    elif ra==3:
                        rel=-1


                    elif ra==4:
                        rel=0
                    elif ra==5:
                        rel = 1
                    elif ra==6:
                        rel = 2
                    elif ra==7:
                        rel = 3


                    if ra==1:
                        rel_b = -1
                    elif ra==2:
                        rel_b = -1
                    elif ra==3:
                        rel_b = -1
                    elif ra==4:
                        rel_b = 0
                    elif ra==5:
                        rel_b = 1
                    elif ra==6:
                        rel_b = 1
                    elif ra==7:
                        rel_b = 1





                    rel = rel / float(3)
                    # rel = rel / float(2)
                    df[ind]['rel_v'][index] = rel
                    if rel < 0:
                        df[ind]['vote'][index] = rel
                        df[ind]['rel_v_b'][index] = -1
                    elif rel>0:
                        df[ind]['vote'][index] = 0
                        df[ind]['rel_v_b'][index] = 1
                    else:
                        df[ind]['vote'][index] = 0
                        df[ind]['rel_v_b'][index] = 0
                    tweet_rel_dict[tweet_id].append(rel)

                    if dataset == 'snopes' or dataset=='snopes_ssi' or dataset=='snopes_nonpol'or dataset=='snopes_incentive' or dataset=='snopes_incentive_10'or dataset=='snopes_2' \
                            or dataset == 'snopes_incentive_notimer'or dataset=='snopes_noincentive_timer':
                        if ra_gt == 'FALSE':
                            rel_gt = -2
                        if ra_gt == 'MOSTLY FALSE':
                            rel_gt = -1
                        if ra_gt == 'MIXTURE':
                            rel_gt = 0
                        if ra_gt == 'MOSTLY TRUE':
                            rel_gt = 1
                        if ra_gt == 'TRUE':
                            rel_gt = 2
                        df[ind]['rel_gt_v'][index] = rel_gt / float(2)



                    elif dataset == 'politifact':
                        if ra_gt == 'pants-fire':
                            rel_gt = -2
                        if ra_gt == 'false':
                            rel_gt = -2
                        if ra_gt == 'mostly-false':
                            rel_gt = -1
                        if ra_gt == 'half-true':
                            rel_gt = 0
                        if ra_gt == 'mostly-true':
                            rel_gt = 1
                        if ra_gt == 'true':
                            rel_gt = 2
                        if rel_gt>=0:
                            df[ind]['rel_gt_v'][index] = rel_gt / float(2)
                        else:
                            df[ind]['rel_gt_v'][index] = rel_gt / float(2)




                    elif dataset == 'mia':
                        if ra_gt == 'rumor':
                            rel_gt = -1
                            mia_rum_t_l.append(tweet_id)
                        if ra_gt == 'non-rumor':
                            rel_gt = 1
                            mia_nrum_t_l.append(tweet_id)
                        # if int(tweet_id) in [100012, 100016, 100053, 100038, 100048]:
                        #     rel_gt = 0


                        df[ind]['rel_gt_v'][index] = rel_gt / float(1)


                    l_scr = df[ind]['rel_gt_v'][index]
                    l_scr_b = -10
                    if l_scr>0:
                        l_scr_b=1
                    elif l_scr<0:
                        l_scr_b=-1
                    else:
                        l_scr=0

                    err_val_b = rel_b - l_scr_b

                    err_val = rel  - l_scr

                    gull_val = rel  - l_scr
                    if gull_val==0:
                        gull_val = 0.01
                    else:
                        gull_val = gull_val * ((np.sign(gull_val) + 1) / float(2))

                    cyn_val = l_scr  - rel
                    if cyn_val==0:
                        cyn_val = 0.01
                    else:
                        cyn_val = cyn_val * ((np.sign(cyn_val) + 1) / float(2))

                    df[ind]['err'][index] = err_val
                    df[ind]['err_b'][index] = err_val_b
                    df[ind]['gull'][index] = gull_val
                    df[ind]['cyn'][index] = cyn_val
                    df[ind]['susc'][index] = df[ind]['cyn'][index] + df[ind]['gull'][index]


                    if rel > 0 and rel_gt > 0:
                        df[ind]['acc'][index] = 1
                    elif rel < 0 and rel_gt < 0:
                        df[ind]['acc'][index] = 1
                    elif rel == 0 and rel_gt == 0:
                        df[ind]['acc'][index] = 1
                    elif rel > 0 and rel_gt < 0:
                        df[ind]['acc'][index] = 0
                    elif rel < 0 and rel_gt > 0:
                        df[ind]['acc'][index] = 0

                    else:
                        df[ind]['acc'][index] = -1


                    if 'liberal' in df[ind]['political_view'][index]:
                        df[ind]['leaning'][index]=1
                    elif 'conservative' in df[ind]['political_view'][index]:
                        df[ind]['leaning'][index]=-1
                    elif 'moderate' in df[ind]['political_view'][index]:
                        df[ind]['leaning'][index] = 0
                    else:
                        df[ind]['leaning'][index] = -10




                df_gr = df[ind].groupby('worker_id', sort=False)
                worker_id_l = df_gr.groups.keys()
                cc = 0
                pr_tim = 0
                cur_tim = 0



                for workerid in worker_id_l:
                    cc = 0
                    pr_tim = 0
                    cur_tim = 0
                    ind_t = df_gr.groups[workerid]
                    df_wid = df[ind].iloc[ind_t]
                    df_wid = df_wid.sort('time', ascending=True)
                    for tmp_ind in df_wid.index.tolist():
                        cur_tim = df_wid['time'][tmp_ind]
                        if cc==0:
                            pr_tim=cur_tim
                        cc+=1
                        delta = cur_tim - pr_tim
                        pr_tim = cur_tim
                        df[ind]['delta_time'][tmp_ind] = delta
                    np.max(df_wid['time']) - np.min(df_wid['time'])

            # news_time_labling_F = open(remotedir + local_dir_saving + 'news_labling_time.txt','w')
            # news_time_labling_csv = open(remotedir + local_dir_saving + dataset +'_news_labling_time.csv','w')
            # #
            # df_gr = df[i].groupby('tweet_id', sort=False)
            # #
            # news_time_labling_csv.write('tweet_id\ttweet_text\t')
            # for w_id in worker_id_l:
            #     news_time_labling_csv.write(str(w_id) + '\t')
            # #
            # news_time_labling_csv.write('AVG\t')
            # news_time_labling_csv.write('\n')
            # #
            # my_ind = -1
            # for tweetid in df_gr.groups:
            #     ts = 0
            #     my_ind += 1
            #     if tweetid==1:
            #         continue
            #     ind_l = df_gr.groups[tweetid]
            #     df_wid = df.iloc[ind_l]
            #     # df_wid = df_wid.sort('time', ascending=True)
            #     # news_time_labling_F.write( '||' + tweet_text_dic[tweetid].replace("?", "'") +'||' + str(tweetid) + '||')
            #     news_time_labling_csv.write(str(my_ind)+'\t')
            #     m_text = tweet_text_dic[tweetid].replace("?", "'")
            #     m_text = m_text.replace("\t", " ")
            #     news_time_labling_csv.write(str(tweetid) + '\t' + m_text +'\t' )
            #     news_time_labling_csv.write(str(tweetid) + '\t' )
            # #
            #     for w_id in worker_id_l:
            #         ind_t = df_wid[df_wid['worker_id'] == w_id].index[0]
            #         ts+= int(df['delta_time'][ind_t])
            # #         news_time_labling_F.write(str(df['delta_time'][ind_t])+'||')
            #         news_time_labling_csv.write(str(df['delta_time'][ind_t]) + '\t')
            # #
            # #     news_time_labling_F.write((str(ts/float(len(worker_id_l)))) + '||')
            # #     news_time_labling_F.write('\n')
            # #
            #     news_time_labling_csv.write((str(ts/float(len(worker_id_l)))) + '\t')
            #     news_time_labling_csv.write('\n')
            #
            #
            # news_time_labling_csv.close()
            # input = remotedir + local_dir_saving  + dataset+'_news_labling_time.csv'
            # df_time = pd.read_csv(input, sep="\t")










            if dataset=='snopes':

                for ind in [1,2,3]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)


            elif dataset=='snopes_2':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_2_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)


            elif dataset=='snopes_incentive':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_incentive_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)


            elif dataset=='snopes_incentive_notimer':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_incentive_notimer_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            elif dataset=='snopes_noincentive_timer':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_noincentive_timer_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)


            elif dataset=='snopes_incentive_10':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_incentive_10_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)
                    print(len(df[ind]))

            elif dataset=='snopes_ssi':

                # for ind in [1,2,3]:
                #     # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                #     #           columns=df[ind].columns, sep="\t", index=False)
                #     df[ind].to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(ind) + '_final_weighted.csv',
                #                    columns=df[ind].columns, sep="\t", index=False)

                for ind in [1, 2, 3]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(ind) + '_final_weighted_new.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            elif dataset=='snopes_nonpol':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_nonpol_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_nonpol_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            elif dataset == 'politifact':

                for ind in [1, 2, 3]:
                    # df[ind].to_csv(remotedir + 'amt_answers_pf_claims_exp' + str(ind) + '_final.csv',
                    #                columns=df[ind].columns, sep="\t", index=False)

                    df[ind].to_csv(remotedir + 'amt_answers_pf_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            elif dataset == 'mia':
                mia_rum_t_l = np.array(list(set(mia_rum_t_l)))
                mia_nrum_t_l = list(set(mia_nrum_t_l))
                und_l = np.array([100012, 100016, 100053, 100038, 100048])

                mia_rum_t_l = list(np.setdiff1d(mia_rum_t_l,und_l))
                random.shuffle(mia_rum_t_l)
                mia_rum_t_l = mia_rum_t_l[:30]
                random.shuffle(mia_nrum_t_l)
                mia_nrum_t_l = mia_nrum_t_l[:30]
                tmp_l = mia_rum_t_l + mia_nrum_t_l

                df[1] = df[1][df[1]['tweet_id'].isin(tmp_l)]
                for ind in [1]:
                    # df[ind].to_csv(remotedir + 'amt_answers_mia_claims_exp1_fina.csv',
                    #                columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_mia_claims_exp1_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_leaning":

        weighted=False
        # dataset = 'snopes'
        # dataset = 'snopes_ssi'
        # dataset = 'snopes_incentive'
        # dataset = 'snopes_leaning'
        dataset = 'snopes_leaning_ben'
        # dataset = 'mia'
        # dataset = 'politifact'
        # dataset = 'snopes_nonpol'
        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []



        if dataset=='mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'


            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                     + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets','r')



            exp1_list = sample_tweets_exp1


            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor= {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id)<100060:
                    tweet_lable_dict[tweet_id]='rumor'
                else:
                    tweet_lable_dict[tweet_id]='non-rumor'
                # if int(tweet_id) in [100012, 100016, 100053, 100038, 100048]:


        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        if dataset == 'snopes' or dataset == 'snopes_ssi' or dataset=='snopes_incentive'or dataset=='snopes_leaning'or dataset=='snopes_leaning_ben':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable


        if dataset == 'snopes_nonpol':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'


            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            print(inp_all)
            source_dict = {}
            text_dict = {}
            date_dict = {}
            # outF = open(remotedir + 'politifact_last_100_news.txt', 'w')
            # F = open(remotedir + 'snopes_latest_20_news_per_lable_non_politics.txt', 'r')


            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable


        # run = 'plot'
        # run = 'analysis'
        run = 'second-analysis'
        exp1_list = sample_tweets_exp1

        if run == 'analysis':
            experiment = 1
            # experiment = 2
            # experiment = 3

            dem_list = []
            rep_list = []
            neut_list = []
            # valid users dict
            Vuser_dict = collections.defaultdict(int)
            if dataset=='snopes':
                query1 = "select workerid, count(*) from mturk_sp_claim_response_exp1_"+str(experiment)+"_recovery group by workerid;"
            elif dataset == 'snopes_incentive':
                query1 = "select workerid, count(*) from mturk_sp_claim_incentive_response_exp1_" + str(experiment) + " group by workerid;"
            elif dataset == 'snopes_leaning':
                query1 = "select workerid, count(*) from mturk_sp_claim_leaning_response_exp1_" + str(
                    experiment) + " group by workerid;"
            elif dataset == 'snopes_leaning_ben':
                query1 = "select workerid, count(*) from mturk_sp_claim_leaning_ben_response_exp1_" + str(
                    experiment) + " group by workerid;"
            elif dataset == 'snopes_ssi':
                # query1 = "select workerid, count(*) from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_recovery_full group by workerid;"
                query1 = "select workerid, count(*) from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_new_recovery group by workerid;"
            elif dataset=='snopes_nonpol':
                query1 = "select workerid, count(*) from mturk_sp_claim_nonpol_response_exp1_recovery group by workerid;"
            elif dataset=='politifact':
                query1 = "select workerid, count(*) from mturk_pf_claim_response_exp1_" + str(experiment) + "_recovery group by workerid;"
            elif dataset=='mia':
                query1 = "select workerid, count(*) from mturk_m_claim_response_exp1_recovery group by workerid;"

            cursor.execute(query1)
            res_exp2 = cursor.fetchall()
            for el in res_exp2:
                if dataset=='snopes_ssi':
                    if experiment==2:
                        if el[0]==4:
                            continue
                    if experiment==3:
                        if el[0]==3:
                            continue
                if el[1] == 53:
                    Vuser_dict[1000*experiment + el[0]] = 1

            if dataset=='snopes':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_response_exp1_"+str(experiment)+"_recovery;"
            elif dataset == 'snopes_incentive':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_incentive_response_exp1_" + str(experiment) + ";"
            elif dataset == 'snopes_leaning':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_leaning_response_exp1_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_leaning_ben':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_leaning_ben_response_exp1_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_ssi':
                # query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_recovery_full;"
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_new_recovery;"
            elif dataset == 'snopes_nonpol':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_sp_claim_nonpol_response_exp1_recovery;"
            elif dataset=='politifact':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_pf_claim_response_exp1_"+str(experiment)+"_recovery;"

            elif dataset == 'mia':
                query2 = "select workerid, tweet_id, ra, rb, rc, at, text from mturk_m_claim_response_exp1_recovery;"




            cursor.execute(query2)
            res_exp2 = cursor.fetchall()

            res_exp1_l = []
            for el in res_exp2:
                if Vuser_dict[1000*experiment + el[0]]==1:
                    res_exp1_l.append((1000*experiment + el[0], el[1], el[2], el[3], el[4], el[5], el[6]))

            if dataset=='snopes':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_sp_claim_demographics_"+str(experiment)+"_recovery"
            elif dataset == 'snopes_incentive':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_incentive_demographics_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_leaning':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_leaning_demographics_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_leaning_ben':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_leaning_ben_demographics_" + str(
                    experiment) + ";"
            elif dataset == 'snopes_ssi':
                # query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                #          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                #          " demographic_political_view, demographic_race," \
                #          " demographic_marital_status from mturk_sp_claim_ssi_demographics" + str(
                #     experiment) + "_recovery_full"
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                         " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                         " demographic_political_view, demographic_race," \
                         " demographic_marital_status from mturk_sp_claim_ssi_demographics" + str(
                    experiment) + "_new_recovery"
            elif dataset == 'snopes_nonpol':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_sp_claim_nonpol_demographics1_recovery"


            elif dataset=='politifact':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_pf_claim_demographics_"+str(experiment)+"_recovery"

            elif dataset == 'mia':
                query3 = "select worker_id, demographic_nationality, demographic_residence, demographic_gender," \
                          " demographic_age, demographic_degree, demographic_employment, demographic_income," \
                          " demographic_political_view, demographic_race," \
                          " demographic_marital_status from mturk_m_claim_demographics_recovery"


            cursor.execute(query3)
            res_leaning = cursor.fetchall()
            leaning_dict = collections.defaultdict()
            dem_l = [];
            rep_l = [];
            neut_l = []

            w_nationality = {};w_residence = {};w_gender = {};w_age = {}
            w_degree = {};w_employment = {};w_income = {};w_political_view = {}
            w_race = {};w_marital_status = {}

            workerid_list_m  = []
            w_nationality_l = [];w_residence_l = [];w_gender_l = []
            w_age_l = [];w_degree_l = [];w_employment_l = []
            w_income_l = [];w_political_view_l = [];w_race_l = [];w_marital_status_l = []
            tweet_id_l = []; ra_l=[]; time_l=[];txt_list_m=[]
            for el in res_leaning:
                w_nationality[1000*experiment+el[0]] = el[1]
                w_residence[1000*experiment + el[0]] = el[2]
                w_gender[1000*experiment + el[0]] = el[3]
                w_age[1000*experiment + el[0]] = el[4]
                w_degree[1000*experiment + el[0]] = el[5]
                w_employment[1000*experiment + el[0]] = el[6]
                w_income[1000*experiment + el[0]] = el[7]
                w_political_view[1000*experiment + el[0]] = el[8]
                w_race[1000*experiment + el[0]] = el[9]
                w_marital_status[1000*experiment + el[0]] = el[10]


                workerid_list_m.append(el[0]+1000*experiment)
                w_nationality_l.append(el[1])
                w_residence_l.append(el[2])
                w_gender_l.append(el[3])
                w_age_l.append(el[4])
                w_degree_l.append(el[5])
                w_employment_l.append(el[6])
                w_income_l.append(el[7])
                w_political_view_l.append(el[8])
                w_race_l.append(el[9])
                w_marital_status_l.append(el[10])

            index_list_m = range(len(w_nationality_l))
            df_w = pd.DataFrame({'worker_id': Series(workerid_list_m, index=index_list_m),
                                'nationality': Series(w_nationality_l, index=index_list_m),
                                'residence': Series(w_residence_l, index=index_list_m),
                                'gender': Series(w_gender_l, index=index_list_m),
                                'age': Series(w_age_l, index=index_list_m),
                                'degree': Series(w_degree_l, index=index_list_m),
                                'employment': Series(w_employment_l, index=index_list_m),
                                'income': Series(w_income_l, index=index_list_m),
                                'political_view': Series(w_political_view_l, index=index_list_m),
                                'race': Series(w_race_l, index=index_list_m),
                                'marital_status': Series(w_marital_status_l, index=index_list_m),})

            workerid_list_m  = []
            w_nationality_l = [];w_residence_l = [];w_gender_l = []
            w_age_l = [];w_degree_l = [];w_employment_l = []
            w_income_l = [];w_political_view_l = [];w_race_l = [];w_marital_status_l = []
            tweet_id_l = []; ra_l=[]; time_l=[];txt_list_m=[];ra_gt_l=[]


            if dataset=='snopes':
                df_w.to_csv(remotedir  +'worker_amt_answers_sp_claims_exp'+str(experiment)+'.csv',
                          columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_incentive':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_incentive_claims_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_leaning':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_leaning_claims_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_leaning_ben':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_leaning_ben_claims_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_ssi':
                # df_w.to_csv(remotedir + 'worker_amt_answers_sp_ssi_claims_exp' + str(experiment) + '.csv',
                #             columns=df_w.columns, sep="\t", index=False)
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_ssi_claims_exp' + str(experiment) + '_new.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset == 'snopes_nonpol':
                df_w.to_csv(remotedir + 'worker_amt_answers_sp_claims_nonpol_exp' + str(experiment) + '.csv',
                            columns=df_w.columns, sep="\t", index=False)
            elif dataset=='politifact':
                df_w.to_csv(remotedir  +'worker_amt_answers_pf_claims_exp'+str(experiment)+'.csv',
                          columns=df_w.columns, sep="\t", index=False)

            elif dataset == 'mia':
                df_w.to_csv(remotedir  +'worker_amt_answers_mia_claims_exp1.csv',
                          columns=df_w.columns, sep="\t", index=False)




            outF = open(remotedir + 'amt_st.txt' , 'w')
            for dem_att in df_w.columns:
                outF.write('=== ' +  dem_att +' ===\n')
                sum_feat = df_w.groupby(dem_att).sum()
                count_feat = df_w.groupby(dem_att).count()
                grouped = df_w.groupby(dem_att)

                for key in grouped.groups.keys():
                    outF.write('|| ' + str(key))
                outF.write('||\n')
                for key in grouped.groups.keys():
                    out_l = grouped.groups[key]
                    outF.write('|| ' + str(len(out_l)))
                outF.write('||\n')
            outF.close()
            # workerid, tweet_id, ra, rb, rc, at, text

            for el in res_exp1_l:
                if el[1]==1:
                    continue
                tweet_id_l.append(el[1])
                ra_l.append(el[2])
                time_l.append(el[5])
                txt_list_m.append(tweet_txt_dict[el[1]])
                ra_gt_l.append(tweet_lable_dict[el[1]])

                workerid_list_m.append(el[0])
                w_nationality_l.append(w_nationality[el[0]])
                w_residence_l.append(w_residence[el[0]])
                w_gender_l.append(w_gender[el[0]])
                w_age_l.append(w_age[el[0]])
                w_degree_l.append(w_degree[el[0]])
                w_employment_l.append(w_employment[el[0]])
                w_income_l.append(w_income[el[0]])
                w_political_view_l.append(w_political_view[el[0]])
                w_race_l.append(w_race[el[0]])
                w_marital_status_l.append(w_marital_status[el[0]])

            index_list_m = range(len(w_nationality_l))
            df = pd.DataFrame({'worker_id': Series(workerid_list_m, index=index_list_m),
                               'tweet_id': Series(tweet_id_l, index=index_list_m),
                               'ra': Series(ra_l, index=index_list_m),
                               'ra_gt': Series(ra_gt_l, index=index_list_m),
                               'time': Series(time_l, index=index_list_m),
                               'text': Series(txt_list_m, index=index_list_m),
                                'nationality': Series(w_nationality_l, index=index_list_m),
                                'residence': Series(w_residence_l, index=index_list_m),
                                'gender': Series(w_gender_l, index=index_list_m),
                                'age': Series(w_age_l, index=index_list_m),
                                'degree': Series(w_degree_l, index=index_list_m),
                                'employment': Series(w_employment_l, index=index_list_m),
                                'income': Series(w_income_l, index=index_list_m),
                                'political_view': Series(w_political_view_l, index=index_list_m),
                                'race': Series(w_race_l, index=index_list_m),
                                'marital_status': Series(w_marital_status_l, index=index_list_m),})

            worker_id_list = []

            print(len(df))

            if dataset == 'snopes':
                df.to_csv(remotedir + 'amt_answers_sp_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_incentive':
                df.to_csv(remotedir + 'amt_answers_sp_incentive_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_leaning':
                df.to_csv(remotedir + 'amt_answers_sp_leaning_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_leaning_ben':
                df.to_csv(remotedir + 'amt_answers_sp_leaning_ben_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_ssi':
                # df.to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(experiment) + '.csv',
                #           columns=df.columns, sep="\t", index=False)
                df.to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(experiment) + '_new.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'snopes_nonpol':
                df.to_csv(remotedir + 'amt_answers_sp_claims_nonpol_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)
            elif dataset == 'politifact':
                df.to_csv(remotedir + 'amt_answers_pf_claims_exp' + str(experiment) + '.csv',
                          columns=df.columns, sep="\t", index=False)

            elif dataset == 'mia':
                df.to_csv(remotedir + 'amt_answers_mia_claims_exp.csv',
                          columns=df.columns, sep="\t", index=False)




        else:

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            if dataset == 'snopes':
                for i in range(1,4):
                    inp1 = remotedir  +'amt_answers_sp_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")


            elif dataset == 'snopes_leaning':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_leaning_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_leaning_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")


            elif dataset == 'snopes_leaning_ben':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_leaning_ben_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_leaning_ben_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")


            elif dataset == 'snopes_incentive':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_incentive_claims_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_incentive_claims_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")



            elif dataset == 'snopes_ssi':
                # for i in range(1,4):
                #     inp1 = remotedir  +'amt_answers_sp_ssi_claims_exp'+str(i)+'.csv'
                #     inp1_w = remotedir  +'worker_amt_answers_sp_claims_ssi_exp'+str(i)+'.csv'
                #     df[i] = pd.read_csv(inp1, sep="\t")
                #     df_w[i] = pd.read_csv(inp1_w, sep="\t")
                for i in range(1, 2):
                    inp1 = remotedir + 'amt_answers_sp_ssi_claims_exp' + str(i) + '_new.csv'
                    inp1_w = remotedir + 'worker_amt_answers_sp_ssi_claims_exp' + str(i) + '_new.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'snopes_nonpol':
                for i in range(1,2):
                    inp1 = remotedir  +'amt_answers_sp_claims_nonpol_exp'+str(i)+'.csv'
                    inp1_w = remotedir  +'worker_amt_answers_sp_claims_nonpol_exp'+str(i)+'.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")


            elif dataset == 'politifact':
                for i in range(1, 4):
                    inp1 = remotedir + 'amt_answers_pf_claims_exp' + str(i) + '.csv'
                    inp1_w = remotedir + 'worker_amt_answers_pf_claims_exp' + str(i) + '.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            elif dataset == 'mia':
                for i in range(1, 2):
                    inp1 = remotedir + 'amt_answers_mia_claims_exp.csv'
                    inp1_w = remotedir + 'worker_amt_answers_mia_claims_exp1.csv'
                    df[i] = pd.read_csv(inp1, sep="\t")
                    df_w[i] = pd.read_csv(inp1_w, sep="\t")

            mia_rum_t_l = []
            mia_nrum_t_l = []
            if dataset=='politifact' or dataset=='snopes' or dataset=='snopes_ssi':
                ind_l = [1,2,3]
            else:
                ind_l = [1]
            for ind in ind_l:

                df[ind].loc[:,'rel_v'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:,'rel_v_b'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'rel_gt_v'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'delta_time'] = df[ind]['tweet_id'] * 0.0

                df[ind].loc[:, 'err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'err_b'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'vote'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'gull'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'cyn'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'susc'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'acc'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'leaning'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'tweet_leaning'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:,'delta_time'] = df[ind]['tweet_id'] * 0.0
                tweet_rel_dict = collections.defaultdict(list)
                tweet_intst_dict = collections.defaultdict(list)
                rel_4 = 0
                for index in df[ind].index.tolist():
                    tweet_id = df[ind]['tweet_id'][index]
                    if tweet_id == 1:
                        continue

                    ra = df[ind]['ra'][index]
                    ra_gt = df[ind]['ra_gt'][index]

                    if ra==1:
                        rel = -1
                    elif ra==2:
                        rel=0
                    elif ra==3:
                        rel=1


                    # rel = rel / float(2)
                    df[ind]['tweet_leaning'][index] = rel

                    tweet_rel_dict[tweet_id].append(rel)

                    if dataset == 'snopes' or dataset=='snopes_ssi' or dataset=='snopes_nonpol'or dataset=='snopes_incentive':
                        if ra_gt == 'FALSE':
                            rel_gt = -2
                        if ra_gt == 'MOSTLY FALSE':
                            rel_gt = -1
                        if ra_gt == 'MIXTURE':
                            rel_gt = 0
                        if ra_gt == 'MOSTLY TRUE':
                            rel_gt = 1
                        if ra_gt == 'TRUE':
                            rel_gt = 2
                        df[ind]['rel_gt_v'][index] = rel_gt / float(2)



                    elif dataset == 'politifact':
                        if ra_gt == 'pants-fire':
                            rel_gt = -2
                        if ra_gt == 'false':
                            rel_gt = -2
                        if ra_gt == 'mostly-false':
                            rel_gt = -1
                        if ra_gt == 'half-true':
                            rel_gt = 0
                        if ra_gt == 'mostly-true':
                            rel_gt = 1
                        if ra_gt == 'true':
                            rel_gt = 2
                        if rel_gt>=0:
                            df[ind]['rel_gt_v'][index] = rel_gt / float(2)
                        else:
                            df[ind]['rel_gt_v'][index] = rel_gt / float(2)




                    elif dataset == 'mia':
                        if ra_gt == 'rumor':
                            rel_gt = -1
                            mia_rum_t_l.append(tweet_id)
                        if ra_gt == 'non-rumor':
                            rel_gt = 1
                            mia_nrum_t_l.append(tweet_id)
                        # if int(tweet_id) in [100012, 100016, 100053, 100038, 100048]:
                        #     rel_gt = 0


                        df[ind]['rel_gt_v'][index] = rel_gt / float(1)


                    l_scr = df[ind]['rel_gt_v'][index]
                    l_scr_b = -10
                    if l_scr>0:
                        l_scr_b=1
                    elif l_scr<0:
                        l_scr_b=-1
                    else:
                        l_scr=0


                    if 'liberal' in df[ind]['political_view'][index]:
                        df[ind]['leaning'][index]=1
                    elif 'conservative' in df[ind]['political_view'][index]:
                        df[ind]['leaning'][index]=-1
                    elif 'moderate' in df[ind]['political_view'][index]:
                        df[ind]['leaning'][index] = 0
                    else:
                        df[ind]['leaning'][index] = -10




                df_gr = df[ind].groupby('worker_id', sort=False)
                worker_id_l = df_gr.groups.keys()
                cc = 0
                pr_tim = 0
                cur_tim = 0



                for workerid in worker_id_l:
                    cc = 0
                    pr_tim = 0
                    cur_tim = 0
                    ind_t = df_gr.groups[workerid]
                    df_wid = df[ind].iloc[ind_t]
                    df_wid = df_wid.sort('time', ascending=True)
                    for tmp_ind in df_wid.index.tolist():
                        cur_tim = df_wid['time'][tmp_ind]
                        if cc==0:
                            pr_tim=cur_tim
                        cc+=1
                        delta = cur_tim - pr_tim
                        pr_tim = cur_tim
                        df[ind]['delta_time'][tmp_ind] = delta
                    np.max(df_wid['time']) - np.min(df_wid['time'])

            # news_time_labling_F = open(remotedir + local_dir_saving + 'news_labling_time.txt','w')
            # news_time_labling_csv = open(remotedir + local_dir_saving + dataset +'_news_labling_time.csv','w')
            # #
            # df_gr = df[i].groupby('tweet_id', sort=False)
            # #
            # news_time_labling_csv.write('tweet_id\ttweet_text\t')
            # for w_id in worker_id_l:
            #     news_time_labling_csv.write(str(w_id) + '\t')
            # #
            # news_time_labling_csv.write('AVG\t')
            # news_time_labling_csv.write('\n')
            # #
            # my_ind = -1
            # for tweetid in df_gr.groups:
            #     ts = 0
            #     my_ind += 1
            #     if tweetid==1:
            #         continue
            #     ind_l = df_gr.groups[tweetid]
            #     df_wid = df.iloc[ind_l]
            #     # df_wid = df_wid.sort('time', ascending=True)
            #     # news_time_labling_F.write( '||' + tweet_text_dic[tweetid].replace("?", "'") +'||' + str(tweetid) + '||')
            #     news_time_labling_csv.write(str(my_ind)+'\t')
            #     m_text = tweet_text_dic[tweetid].replace("?", "'")
            #     m_text = m_text.replace("\t", " ")
            #     news_time_labling_csv.write(str(tweetid) + '\t' + m_text +'\t' )
            #     news_time_labling_csv.write(str(tweetid) + '\t' )
            # #
            #     for w_id in worker_id_l:
            #         ind_t = df_wid[df_wid['worker_id'] == w_id].index[0]
            #         ts+= int(df['delta_time'][ind_t])
            # #         news_time_labling_F.write(str(df['delta_time'][ind_t])+'||')
            #         news_time_labling_csv.write(str(df['delta_time'][ind_t]) + '\t')
            # #
            # #     news_time_labling_F.write((str(ts/float(len(worker_id_l)))) + '||')
            # #     news_time_labling_F.write('\n')
            # #
            #     news_time_labling_csv.write((str(ts/float(len(worker_id_l)))) + '\t')
            #     news_time_labling_csv.write('\n')
            #
            #
            # news_time_labling_csv.close()
            # input = remotedir + local_dir_saving  + dataset+'_news_labling_time.csv'
            # df_time = pd.read_csv(input, sep="\t")










            if dataset=='snopes':

                for ind in [1,2,3]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            if dataset=='snopes_incentive':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_incentive_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)


            if dataset=='snopes_leaning':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_leaning_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)


            if dataset=='snopes_leaning_ben':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_leaning_ben_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)


            if dataset=='snopes_ssi':

                # for ind in [1,2,3]:
                #     # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                #     #           columns=df[ind].columns, sep="\t", index=False)
                #     df[ind].to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(ind) + '_final_weighted.csv',
                #                    columns=df[ind].columns, sep="\t", index=False)

                for ind in [1, 2, 3]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_ssi_claims_exp' + str(ind) + '_final_weighted_new.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            elif dataset=='snopes_nonpol':

                for ind in [1]:
                    # df[ind].to_csv(remotedir  +'amt_answers_sp_nonpol_claims_exp'+str(ind)+'_final.csv',
                    #           columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_sp_nonpol_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            elif dataset == 'politifact':

                for ind in [1, 2, 3]:
                    # df[ind].to_csv(remotedir + 'amt_answers_pf_claims_exp' + str(ind) + '_final.csv',
                    #                columns=df[ind].columns, sep="\t", index=False)

                    df[ind].to_csv(remotedir + 'amt_answers_pf_claims_exp' + str(ind) + '_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

            elif dataset == 'mia':
                mia_rum_t_l = np.array(list(set(mia_rum_t_l)))
                mia_nrum_t_l = list(set(mia_nrum_t_l))
                und_l = np.array([100012, 100016, 100053, 100038, 100048])

                mia_rum_t_l = list(np.setdiff1d(mia_rum_t_l,und_l))
                random.shuffle(mia_rum_t_l)
                mia_rum_t_l = mia_rum_t_l[:30]
                random.shuffle(mia_nrum_t_l)
                mia_nrum_t_l = mia_nrum_t_l[:30]
                tmp_l = mia_rum_t_l + mia_nrum_t_l

                df[1] = df[1][df[1]['tweet_id'].isin(tmp_l)]
                for ind in [1]:
                    # df[ind].to_csv(remotedir + 'amt_answers_mia_claims_exp1_fina.csv',
                    #                columns=df[ind].columns, sep="\t", index=False)
                    df[ind].to_csv(remotedir + 'amt_answers_mia_claims_exp1_final_weighted.csv',
                                   columns=df[ind].columns, sep="\t", index=False)

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_PDF_CDF_MPB_fig":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # dataset = 'snopes'
        # dataset = 'snopes_nonpol'
        dataset = 'snopes_ssi'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        if dataset == 'mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                  + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

            exp1_list = sample_tweets_exp1
            tweet_id = 100010
            publisher_name = 110
            tweet_popularity = {}
            tweet_text_dic = {}
            for input_file in [input_rumor, input_non_rumor]:
                for line in input_file:
                    line.replace('\n', '')
                    line_splt = line.split('\t')
                    tweet_txt = line_splt[1]
                    tweet_link = line_splt[1]
                    tweet_id += 1
                    publisher_name += 1
                    tweet_popularity[tweet_id] = int(line_splt[2])
                    tweet_text_dic[tweet_id] = tweet_txt

            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor = {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id) < 100060:
                    tweet_lable_dict[tweet_id] = 'rumor'
                else:
                    tweet_lable_dict[tweet_id] = 'non-rumor'

        if dataset == 'snopes_nonpol':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable

        if dataset == 'snopes' or dataset=='snopes_ssi':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable

        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_ssi']:  # ['snopes_nonpol', 'snopes','politifact','mia']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset=='snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes_ssi'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    # tweet_avg_group[t_id] = np.mean(dem_val_list) - np.mean(rep_val_list)
                    # tweet_med_group[t_id] = np.median(dem_val_list) - np.median(rep_val_list)
                    # tweet_var_group[t_id] = np.var(dem_val_list) - np.var(rep_val_list)
                    # tweet_kldiv_group[t_id] = np.mean(dem_val_list)+np.mean(rep_val_list) + np.mean(neut_val_list)
                    # tweet_kldiv_group[t_id] = np.var(dem_val_list) * np.var(rep_val_list) / np.var(neut_val_list)

                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))

                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_l.append(tweet_skew[t_id])

                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)

                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
                    # tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    # tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)

            # news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

            if dataset == 'snopes' or dataset == 'snopes_nonpol' or dataset=='snopes_ssi':
                col_l = ['darkred', 'orange', 'gray', 'lime', 'green']
                # col = 'purple'
                col = 'k'
                news_cat_list_n = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_t_f = [['FALSE', 'MOSTLY FALSE'], ['MOSTLY TRUE', 'TRUE']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']
            if dataset == 'politifact':
                col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']
                news_cat_list_n = ['PANTS ON FIRE', 'FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'MOSTLY TRUE', 'TRUE']
                # col = 'c'
                col = 'k'
                news_cat_list_t_f = [['pants-fire', 'false', 'mostly-false'], ['mostly-true', 'true']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            if dataset == 'mia':
                col_l = ['red', 'green']
                news_cat_list_n = ['RUMORS', 'NON RUMORS']
                # col = 'brown'
                col = 'k'
                col_t_f = ['red', 'green']
                news_cat_list_t_f = [['rumors'], ['non-rumors']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            count = 0
            # Y = [0]*len(thr_list)




            tweet_abs_perc_rnd_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            # tweet_perc_rnd_sort = sorted(tweet_dev_avg_rnd, key=tweet_dev_avg_rnd.get, reverse=True)
            tweet_abs_perc_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            # tweet_perc_sort = sorted(tweet_dev_avg, key=tweet_dev_avg.get, reverse=True)
            tweet_disp_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            gt_l = []
            pt_l = []

            fig_cdf = True
            # fig_cdf = False
            if fig_cdf == True:

                #####################################################33
                out_dict = tweet_dev_avg
                # out_dict = tweet_avg
                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                pt_l = []
                pt_l_dict = collections.defaultdict(list)
                for t_id in tweet_l_sort:
                    pt_l.append(out_dict[t_id])
                count = 0

                # num_bins = len(pt_l)
                # counts, bin_edges = np.histogram(pt_l, bins=num_bins, normed=True)
                # cdf = np.cumsum(counts)
                # scale = 1.0 / cdf[-1]
                # ncdf = scale * cdf
                # mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5,linestyle='--', label='All news stories')
                mplpl.rcParams['figure.figsize'] = 5.8, 3.7
                mplpl.rcParams['figure.figsize'] = 7, 5.3
                mplpl.rc('xtick', labelsize='large')
                mplpl.rc('ytick', labelsize='large')
                mplpl.rc('legend', fontsize='small')
                for cat_m in news_cat_list:
                    count += 1
                    pt_l_dict[cat_m] = []
                    for t_id in tweet_l_sort:
                        if tweet_lable_dict[t_id] == cat_m:
                            pt_l_dict[cat_m].append(out_dict[t_id])

                    df_tt = pd.DataFrame(np.array(pt_l_dict[cat_m]), columns=[cat_m])
                    df_tt[cat_m].plot(kind='kde', lw=6, color=col_l[count - 1], label=cat_m)
                    print(cat_m + ' : ' + str(np.min(list(df_tt[cat_m]))))
                    print(cat_m + ' : ' + str(np.max(list(df_tt[cat_m]))))
                    # print(cat_m +  ' : '+ str(len(df_tt[df_tt[cat_m]>=0.5])))
                    # print(cat_m +  ' : '+ str(len(df_tt[df_tt[cat_m]<=-0.5])))
                    # num_bins = len(pt_l_dict[cat_m])
                    # counts, bin_edges = np.histogram(pt_l_dict[cat_m], bins=num_bins, normed=True)
                    # cdf = np.cumsum(counts)
                    # scale = 1.0 / cdf[-1]
                    # ncdf = scale * cdf
                    # mplpl.plot(bin_edges[1:], ncdf, c=col_l[count-1], lw=5, label=cat_m)
                # exit()
                mplpl.ylabel('PDF', fontsize=22, fontweight='bold')
                # mplpl.ylabel('CDF', fontsize=22, fontweight = 'bold')
                # mplpl.xlabel('Perceived Truth Level(SSI)', fontsize=22, fontweight = 'bold')
                mplpl.xlabel('Mean Perception Bias', fontsize=22, fontweight='bold')
                mplpl.grid()
                mplpl.title(data_name, fontsize='x-large')
                labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
                y = [0.0, 0.5, 1, 1.5, 2]
                mplpl.yticks(y, labels)
                legend_properties = {'weight': 'bold'}

                # plt.legend(prop=legend_properties)
                mplpl.legend(loc="upper left", prop=legend_properties, fontsize='small', ncol=1)  # , fontweight = 'bold')
                mplpl.xlim([-2, 2])
                mplpl.ylim([0, 2.5])
                # mplpl.xlim([-1, 1])
                # mplpl.ylim([0, 1])
                mplpl.subplots_adjust(bottom=0.24)
                mplpl.subplots_adjust(left=0.18)

                pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_mpb_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_pt_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_pt_cdf'
                mplpl.savefig(pp + '.pdf', format='pdf')
                mplpl.savefig(pp + '.png', format='png')
                mplpl.figure()

                exit()
    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_PDF_CDF_APB_fig":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'

        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []




        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_ssi','snopes', 'politifact', 'mia']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            w_cyn_dict = collections.defaultdict()
            w_gull_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    # tweet_avg_group[t_id] = np.mean(dem_val_list) - np.mean(rep_val_list)
                    # tweet_med_group[t_id] = np.median(dem_val_list) - np.median(rep_val_list)
                    # tweet_var_group[t_id] = np.var(dem_val_list) - np.var(rep_val_list)
                    # tweet_kldiv_group[t_id] = np.mean(dem_val_list)+np.mean(rep_val_list) + np.mean(neut_val_list)
                    # tweet_kldiv_group[t_id] = np.var(dem_val_list) * np.var(rep_val_list) / np.var(neut_val_list)

                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    # w_abs_err_list = list(df_tmp['abs_err'])
                    w_sus_list = list(df_tmp['susc'])
                    # w_norm_err_list = list(df_tmp['norm_err'])
                    # w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    # w_cyn_list = list(df_tmp['cyn'])
                    # w_gull_list = list(df_tmp['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])

                    df_cyn = df_tmp[df_tmp['cyn'] > 0]
                    df_gull = df_tmp[df_tmp['gull'] > 0]

                    w_cyn_list = list(df_cyn['cyn'])
                    w_gull_list = list(df_gull['gull'])

                    w_cyn_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_dict[t_id] = np.mean(w_gull_list)
                    w_apb_dict[t_id] = np.mean(w_sus_list)

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))

                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_l.append(tweet_skew[t_id])

                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)

                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
                    # tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    # tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)

            # news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

            if dataset == 'snopes' or dataset == 'snopes_ssi':
                col_l = ['darkred', 'orange', 'gray', 'lime', 'green']
                # col = 'purple'
                col = 'k'
                news_cat_list_n = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_t_f = [['FALSE', 'MOSTLY FALSE'], ['MOSTLY TRUE', 'TRUE']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']
            if dataset == 'politifact':
                col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']
                news_cat_list_n = ['PANTS ON FIRE', 'FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'MOSTLY TRUE', 'TRUE']
                # col = 'c'
                col = 'k'
                news_cat_list_t_f = [['pants-fire', 'false', 'mostly-false'], ['mostly-true', 'true']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            if dataset == 'mia':
                col_l = ['red', 'green']
                news_cat_list_n = ['RUMORS', 'NON RUMORS']
                # col = 'brown'
                col = 'k'
                col_t_f = ['red', 'green']
                news_cat_list_t_f = [['rumors'], ['non-rumors']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            count = 0
            # Y = [0]*len(thr_list)




            tweet_abs_perc_rnd_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            # tweet_perc_rnd_sort = sorted(tweet_dev_avg_rnd, key=tweet_dev_avg_rnd.get, reverse=True)
            tweet_abs_perc_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            # tweet_perc_sort = sorted(tweet_dev_avg, key=tweet_dev_avg.get, reverse=True)
            tweet_disp_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            gt_l = []
            pt_l = []

            fig_cdf = True
            # fig_cdf = False
            if fig_cdf == True:

                #####################################################33

                out_dict = w_apb_dict
                # out_dict = w_gull_dict
                # out_dict = w_cyn_dict

                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                pt_l = []
                pt_l_dict = collections.defaultdict(list)
                for t_id in tweet_l_sort:
                    pt_l.append(out_dict[t_id])
                count = 0

                # num_bins = len(pt_l)
                # counts, bin_edges = np.histogram(pt_l, bins=num_bins, normed=True)
                # cdf = np.cumsum(counts)
                # scale = 1.0 / cdf[-1]
                # ncdf = scale * cdf
                # mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5,linestyle='--', label='All news stories')
                # mplpl.rcParams['figure.figsize'] = 5.8, 3.7
                mplpl.rcParams['figure.figsize'] = 5.6, 3.4
                mplpl.rc('xtick', labelsize='x-large')
                mplpl.rc('ytick', labelsize='x-large')
                mplpl.rc('legend', fontsize='medium')
                for cat_m in news_cat_list:
                    count += 1
                    pt_l_dict[cat_m] = []
                    for t_id in tweet_l_sort:
                        if tweet_lable_dict[t_id] == cat_m:
                            if out_dict[t_id] > 0 or out_dict[t_id] <= 0:
                                pt_l_dict[cat_m].append(out_dict[t_id])

                    # df_tt = pd.DataFrame(np.array(pt_l_dict[cat_m]), columns=[cat_m])
                    # df_tt[cat_m].plot(kind='kde', lw=6, color=col_l[count-1], label=cat_m)

                    num_bins = len(pt_l_dict[cat_m])
                    counts, bin_edges = np.histogram(pt_l_dict[cat_m], bins=num_bins, normed=True)
                    cdf = np.cumsum(counts)
                    scale = 1.0 / cdf[-1]
                    ncdf = scale * cdf
                    mplpl.plot(bin_edges[1:], ncdf, c=col_l[count - 1], lw=5, label=cat_m)

                # mplpl.ylabel('PDF of Perceived Truth Level', fontsize=20, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n Absolute Perception Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n False Positive Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n False Negative Bias', fontsize=13, fontweight = 'bold')

                # mplpl.ylabel('PDF of \n Absolute Perception Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('PDF of \n False Positive Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('PDF of \n False Negative Bias', fontsize=13, fontweight='bold')

                mplpl.xlabel('Total Perception Bias', fontsize=24, fontweight = 'bold')
                # mplpl.xlabel('False Positive Bias', fontsize=24, fontweight = 'bold')
                # mplpl.xlabel('False Negative Bias', fontsize=24, fontweight='bold')
                mplpl.ylabel('CDF', fontsize=24, fontweight='bold')
                mplpl.grid()
                mplpl.title(data_name, fontsize='x-large')
                labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
                y = [0.0, 0.5, 1, 1.5, 2]
                # mplpl.yticks(y, labels)
                legend_properties = {'weight': 'bold'}

                # plt.legend(prop=legend_properties)
                # mplpl.legend(loc="upper left",prop=legend_properties,fontsize='small', ncol=1)#, fontweight = 'bold')
                mplpl.legend(loc="lower right", prop=legend_properties, fontsize='medium', ncol=1)  # , fontweight = 'bold')
                mplpl.xlim([0, 2])
                # mplpl.ylim([0, 2])
                mplpl.ylim([0, 1])
                mplpl.subplots_adjust(bottom=0.24)
                mplpl.subplots_adjust(left=0.18)

                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_pt_pdf'
                pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_cdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_cdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_FNB_cdf'

                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_FNB_pdf'

                mplpl.savefig(pp + '.pdf', format='pdf')
                mplpl.savefig(pp + '.png', format='png')
                mplpl.figure()

                exit()
    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_PDF_CDF_disp_fig":



        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []


        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l= []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False



        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in  ['snopes_ssi','snopes_nonpol','snopes','politifact','mia']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = [ 'rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                # outF = open(remotedir + 'table_out.txt', 'w')


            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = [ 'FALSE','MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false',  'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = [ 'FALSE','MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false',  'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = [ 'pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true','true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false','half-true', 'mostly-true',  'true']
                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                # outF = open(remotedir + 'table_out.txt', 'w')





            if dataset=='snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1,2,3]
                data_name = 'Snopes'
            if dataset=='snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1,2,3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset=='politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1,2,3]
                data_name = 'PolitiFact'
            elif dataset=='mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()

            tweet_kldiv_group= collections.defaultdict()

            w_cyn_dict= collections.defaultdict()
            w_gull_dict= collections.defaultdict()
            w_apb_dict= collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_'+data_n+'_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()


                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []


                    dem_df = df_tmp[df_tmp['leaning']==1]
                    rep_df = df_tmp[df_tmp['leaning']==-1]
                    neut_df = df_tmp[df_tmp['leaning']==0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    # tweet_avg_group[t_id] = np.mean(dem_val_list) - np.mean(rep_val_list)
                    # tweet_med_group[t_id] = np.median(dem_val_list) - np.median(rep_val_list)
                    # tweet_var_group[t_id] = np.var(dem_val_list) - np.var(rep_val_list)
                    # tweet_kldiv_group[t_id] = np.mean(dem_val_list)+np.mean(rep_val_list) + np.mean(neut_val_list)
                    # tweet_kldiv_group[t_id] = np.var(dem_val_list) * np.var(rep_val_list) / np.var(neut_val_list)

                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list,rep_val_list)[1], 4)



                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    # w_abs_err_list = list(df_tmp['abs_err'])
                    w_sus_list = list(df_tmp['susc'])
                    # w_norm_err_list = list(df_tmp['norm_err'])
                    # w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    # w_cyn_list = list(df_tmp['cyn'])
                    # w_gull_list = list(df_tmp['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])


                    df_cyn = df_tmp[df_tmp['cyn']>0]
                    df_gull = df_tmp[df_tmp['gull']>0]

                    w_cyn_list = list(df_cyn['cyn'])
                    w_gull_list = list(df_gull['gull'])

                    w_cyn_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_dict[t_id] = np.mean(w_gull_list)
                    w_apb_dict[t_id] = np.mean(w_sus_list)


                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])

                    for vot in vot_list_tmp:
                        if vot < 0 :
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))



                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_l.append(tweet_skew[t_id])



                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)


                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2/float(3), -1/float(3), 0, 1/float(3), 2/float(3),1]:
                        sum_rnd_perc+= val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
                    # tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    # tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)

            # news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

            if dataset=='snopes' or dataset=='snopes_nonpol'or dataset=='snopes_ssi':
                col_l = ['darkred', 'orange', 'gray', 'lime', 'green']
                # col = 'purple'
                col = 'k'
                news_cat_list_n = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_t_f = [['FALSE', 'MOSTLY FALSE'],['MOSTLY TRUE', 'TRUE']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']
            if dataset=='politifact':
                col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']
                news_cat_list_n = ['PANTS ON FIRE', 'FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'MOSTLY TRUE', 'TRUE']
                # col = 'c'
                col = 'k'
                news_cat_list_t_f = [['pants-fire', 'false', 'mostly-false'],['mostly-true', 'true']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            if dataset=='mia':
                col_l = ['red', 'green']
                news_cat_list_n = ['RUMORS', 'NON RUMORS']
                # col = 'brown'
                col = 'k'
                col_t_f = ['red', 'green']
                news_cat_list_t_f = [['rumors'], ['non-rumors']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            count = 0
            # Y = [0]*len(thr_list)




            tweet_abs_perc_rnd_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            # tweet_perc_rnd_sort = sorted(tweet_dev_avg_rnd, key=tweet_dev_avg_rnd.get, reverse=True)
            tweet_abs_perc_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            # tweet_perc_sort = sorted(tweet_dev_avg, key=tweet_dev_avg.get, reverse=True)
            tweet_disp_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            gt_l = []
            pt_l = []

            fig_cdf = True
            # fig_cdf = False
            if fig_cdf==True:

        #####################################################33

                # out_dict = w_apb_dict
                # out_dict = w_gull_dict
                # out_dict = w_cyn_dict
                out_dict = tweet_var

                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                pt_l = []
                pt_l_dict = collections.defaultdict(list)
                for t_id in tweet_l_sort:
                    pt_l.append(out_dict[t_id])
                count=0

                # num_bins = len(pt_l)
                # counts, bin_edges = np.histogram(pt_l, bins=num_bins, normed=True)
                # cdf = np.cumsum(counts)
                # scale = 1.0 / cdf[-1]
                # ncdf = scale * cdf
                # mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5,linestyle='--', label='All news stories')
                # mplpl.rcParams['figure.figsize'] = 5.8, 3.7
                mplpl.rcParams['figure.figsize'] = 4.5, 2.5
                mplpl.rc('xtick', labelsize='large')
                mplpl.rc('ytick', labelsize='large')
                mplpl.rc('legend', fontsize='small')
                for cat_m in news_cat_list:
                    count += 1
                    pt_l_dict[cat_m] = []
                    for t_id in tweet_l_sort:
                        if tweet_lable_dict[t_id]==cat_m:
                            if out_dict[t_id]>0 or out_dict[t_id]<=0:
                                pt_l_dict[cat_m].append(out_dict[t_id])



                    # df_tt = pd.DataFrame(np.array(pt_l_dict[cat_m]), columns=[cat_m])
                    # df_tt[cat_m].plot(kind='kde', lw=6, color=col_l[count-1], label=cat_m)

                    num_bins = len(pt_l_dict[cat_m])
                    counts, bin_edges = np.histogram(pt_l_dict[cat_m], bins=num_bins, normed=True)
                    cdf = np.cumsum(counts)
                    scale = 1.0 / cdf[-1]
                    ncdf = scale * cdf
                    mplpl.plot(bin_edges[1:], ncdf, c=col_l[count-1], lw=5, label=cat_m)

                # mplpl.ylabel('PDF of Perceived Truth Level', fontsize=20, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n Absolute Perception Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n False Positive Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n False Negative Bias', fontsize=13, fontweight = 'bold')

                # mplpl.ylabel('PDF of \n Absolute Perception Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('PDF of \n False Positive Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('PDF of \n False Negative Bias', fontsize=13, fontweight='bold')

                # mplpl.xlabel('Total Perception Bias', fontsize=24, fontweight = 'bold')
                # mplpl.xlabel('False Positive Bias', fontsize=24, fontweight = 'bold')
                mplpl.xlabel('Disputability', fontsize=20, fontweight = 'bold')
                mplpl.ylabel('CDF', fontsize=20, fontweight = 'bold')
                mplpl.grid()
                mplpl.title(data_name, fontsize='x-large')
                labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
                y = [0.0, 0.5, 1, 1.5, 2]
                # mplpl.yticks(y, labels)
                legend_properties = {'weight': 'bold'}


                # plt.legend(prop=legend_properties)
                # mplpl.legend(loc="upper left",prop=legend_properties,fontsize='small', ncol=1)#, fontweight = 'bold')
                mplpl.legend(loc="lower right",prop=legend_properties,fontsize='small', ncol=1)#, fontweight = 'bold')
                mplpl.xlim([0, 1])
                # mplpl.ylim([0, 2])
                mplpl.ylim([0, 1])
                mplpl.subplots_adjust(bottom=0.24)
                mplpl.subplots_adjust(left=0.18)

                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_pt_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_cdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_cdf'
                pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_disput_cdf'

                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_FNB_pdf'

                mplpl.savefig(pp + '.pdf', format='pdf')
                mplpl.savefig(pp + '.png', format='png')
                mplpl.figure()


                exit()





    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_true-false_accuracy":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []



        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        # exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []



        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'

        inp10 = remotedir + 'amt_answers_' + 'sp_incentive_10' + '_claims_exp' + str(1) + '_final_weighted.csv'
        df10 = pd.read_csv(inp10, sep="\t")

        claims_10_list = set(df10['tweet_id'])
        for dataset in ['snopes_incentive','snopes_incentive_10','snopes_2','snopes_ssi', 'snopes_nonpol', 'snopes', 'mia', 'mia', 'politifact']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset=='snopes_ssi'or dataset=='snopes_incentive'or dataset=='snopes_incentive_10' or dataset=='snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'
            elif dataset == 'snopes_incentive_10':
                data_n = 'sp_incentive_10'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive_10'
            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            news_cat_list_tf = [4, 2, 3, 1]
            t_f_dict_len = collections.defaultdict(int)
            t_f_dict = {}

            if dataset == 'snopes' or dataset == 'snopes_nonpol' or dataset=='snopes_ssi'or dataset=='snopes_incentive' or dataset=='snopes_incentive_10'or dataset=='snopes_2':
                news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_v = [-1, -.5, 0, 0.5, 1]

            if dataset == 'politifact':
                news_cat_list_t_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_v = [-1, -1, -0.5, 0, 0.5, 1]

            if dataset == 'mia':
                news_cat_list_t_f = ['rumor', 'non-rumor']
                news_cat_list_v = [-1, 1]

            w_fnb_dict = collections.defaultdict()
            w_fpb_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()
            gt_acc = collections.defaultdict()
            for cat in news_cat_list_v:
                gt_acc[cat] = [0] * (len(news_cat_list_t_f))
            weight_list = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
            # weight_list = [-2.5, -0.84, 0.25, 1, -1.127, 1.1, 1.05]
            # weight_list = [-2.36, -0.73, 0.53, 0.87, -0.87, 0.93, 1.53]
            pt_list = []
            gt_list = []
            pp = 0
            pf = 0
            tpp = 0
            tpf = 0
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                tmp_list = df_m[df_m['ra'] == 1].index.tolist()
                df_m['rel_v'][tmp_list] = [weight_list[0]] * len(tmp_list)
                tmp_list = df_m[df_m['ra'] == 2].index.tolist()
                df_m['rel_v'][tmp_list] = [weight_list[1]] * len(tmp_list)
                tmp_list = df_m[df_m['ra'] == 3].index.tolist()
                df_m['rel_v'][tmp_list] = [weight_list[2]] * len(tmp_list)
                tmp_list = df_m[df_m['ra'] == 4].index.tolist()
                df_m['rel_v'][tmp_list] = [weight_list[3]] * len(tmp_list)
                tmp_list = df_m[df_m['ra'] == 5].index.tolist()
                df_m['rel_v'][tmp_list] = [weight_list[4]] * len(tmp_list)
                tmp_list = df_m[df_m['ra'] == 6].index.tolist()
                df_m['rel_v'][tmp_list] = [weight_list[5]] * len(tmp_list)
                tmp_list = df_m[df_m['ra'] == 7].index.tolist()
                df_m['rel_v'][tmp_list] = [weight_list[6]] * len(tmp_list)

                # for t_id in grouped.groups.keys():
                for t_id in claims_10_list:

                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    pt = np.mean(df_tmp['rel_v_b'])
                    # pt = np.mean(df_tmp['acc'])
                    gt = list(df_tmp['rel_gt_v'])[0]
                    if gt>0:
                        if pt>0:
                            pp+=1
                        tpp+=1
                    if gt<0:
                        if pt<0:
                            pf+=1
                        tpf+=1
                    min_dist = 10
                    for i_cat in range(len(news_cat_list_t_f)):
                        curr_dist = np.abs(pt - news_cat_list_v[i_cat])
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            cat = i_cat

                    gt_acc[gt][cat] += 1
                    gt_list.append(gt)
                    pt_list.append(news_cat_list_v[cat])
            ##################################################
            print(pp/float(tpp))
            # print(tpp)
            print(pf/float(tpf))
            # print(tpf)
            print(np.corrcoef(pt_list, gt_list))
            print(scipy.stats.spearmanr(pt_list, gt_list))
            width = 0.05
            pr = -10
            title_l = news_cat_list
            outp = {}
            outp_var = {}
            # news_cat_list = ['pants-fire', 'false', 'mostly_false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            if dataset == 'snopes' or dataset == 'snopes_nonpol' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_incentive_10':
                # col_l = ['b', 'g', 'c', 'y', 'r']
                col_l = ['red', 'orange', 'gray', 'lime', 'green']

                news_cat_list_n = ['FALSE', 'MOSTLY\nFALSE', 'MIXTURE', 'MOSTLY\nTRUE', 'TRUE']
            if dataset == 'politifact':
                # col_l = ['grey','b', 'g', 'c', 'y', 'r']
                col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']

                news_cat_list_n = ['PANTS ON\nFIRE', 'FALSE', 'MOSTLY\nFALSE', 'HALF\nTRUE', 'MOSTLY\nTRUE', 'TRUE']

            if dataset == 'mia':
                # col_l = ['b', 'r']
                col_l = ['red', 'green']
                news_cat_list_n = ['RUMORS', 'NON RUMORS']
            count = 0
            Y = [0] * len(news_cat_list_n)
            # Y1 = [0] * len(thr_list)
            mplpl.rcParams['figure.figsize'] = 6.8, 5
            mplpl.rc('xtick', labelsize='large')
            mplpl.rc('ytick', labelsize='large')
            mplpl.rc('legend', fontsize='small')
            cat_num_st = 0
            for cat_v_ind in range(len(news_cat_list_v)):
                cat_v = news_cat_list_v[cat_v_ind]
                for i in range(len(gt_acc[cat_v])):
                    cat_num_st += gt_acc[cat_v][i]
                break
            acc_dict = {}
            for cat_v_ind in range(len(news_cat_list_v)):
                cat_v = news_cat_list_v[cat_v_ind]
                acc_dict[cat_v] = gt_acc[cat_v][cat_v_ind] / float(cat_num_st)

                acc = np.mean(acc_dict.values())
            for cat_v in news_cat_list_v:
                count += 1
                count += 1
                outp[cat_v] = []
                outp[cat_v] = gt_acc[cat_v]
                # for i in range(len(gt_acc[cat_v])):
                #     outp[i].append(gt_acc[cat_v][i])
                if dataset == 'snopes' or dataset == 'snopes_nonpol'or dataset == 'snopes_ssi':
                    mplpl.bar([0.09, 0.18, 0.28, 0.38, 0.48], outp[cat_v], width, bottom=np.array(Y),
                              color=col_l[count - 1],
                              label=news_cat_list_n[count - 1])
                elif dataset == 'politifact':
                    mplpl.bar([0.09, 0.18, 0.28, 0.38, 0.48], outp[cat_v], width, bottom=np.array(Y),
                              color=col_l[count - 1],
                              label=news_cat_list_n[count - 1])
                Y = np.array(Y) + np.array(outp[cat_v])

            mplpl.xlim([0.08, 0.58])
            # mplpl.ylim([0, 1.38])
            mplpl.ylabel('Composition of labeled news stories', fontsize=14, fontweight='bold')
            # mplpl.xlabel('Top k news stories reported by negative PTL', fontsize=13.8,fontweight = 'bold')
            mplpl.xlabel('Users\' perception', fontsize=14,
                         fontweight='bold')
            # mplpl.xlabel('Top k news stories ranked by NAPB', fontsize=18)

            mplpl.legend(loc="upper right", ncol=2, fontsize='small')

            mplpl.subplots_adjust(bottom=0.2)

            mplpl.subplots_adjust(left=0.18)
            mplpl.grid()
            mplpl.title(data_name, fontsize='x-large')
            labels = news_cat_list_n
            x = [0.1, 0.2, 0.3, 0.4, 0.5]
            mplpl.xticks(x, labels)
            # pp = remotedir + '/fig/fig_exp1/news_based/initial/' + data_n + '_vote_composition_gt'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_vote_composition_gt'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_napb_composition_gt'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_gt_accuracy_ordinalreg_weight'
            pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_gt_accuracy'
            mplpl.savefig(pp + '.pdf', format='pdf')
            mplpl.savefig(pp + '.png', format='png')

            exit()

    if args.t == "AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_mpb_cdf_toghether":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []


        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []



        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        # for dataset in ['snopes','snopes_ssi', 'snopes_nonpol', 'politifact', 'mia']:
        for dataset in ['snopes', 'snopes_ssi', 'snopes_incentive']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                tweet_id = 100010
                publisher_name = 110
                tweet_popularity = {}
                tweet_text_dic = {}
                for input_file in [input_rumor, input_non_rumor]:
                    for line in input_file:
                        line.replace('\n', '')
                        line_splt = line.split('\t')
                        tweet_txt = line_splt[1]
                        tweet_link = line_splt[1]
                        tweet_id += 1
                        publisher_name += 1
                        tweet_popularity[tweet_id] = int(line_splt[2])
                        tweet_text_dic[tweet_id] = tweet_txt

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # if int(tweet_id) in [100012, 100016, 100053, 100038, 100048]:
                        #     tweet_lable_dict[tweet_id] = 'undecided'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive':
                claims_list = []
                col = 'r'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
                news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                col = 'r'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
                news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'politifact':
                col = 'g'

                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
                news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1,2,3]
                ind_l = [1]
                col = 'purple'

                data_name = 'Snopes'
            if dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1,2,3]
                ind_l = [1]
                col = 'g'

                data_name = 'Snopes_incentive'

            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1,2,3]
                ind_l = [1]
                col = 'k'

                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                col = 'green'

                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                col = 'c'
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                col = 'orange'
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()
                df[ind].loc[:, 'abs_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_abs_err'] = df[ind]['tweet_id'] * 0.0

                groupby_ftr = 'tweet_id'
                grouped = df[ind].groupby(groupby_ftr, sort=False)
                grouped_sum = df[ind].groupby(groupby_ftr, sort=False).sum()

                for ind_t in df[ind].index.tolist():
                    t_id = df[ind]['tweet_id'][ind_t]
                    err = df[ind]['err'][ind_t]
                    abs_err = np.abs(err)
                    df[ind]['abs_err'][ind_t] = abs_err
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += (val - df[ind]['rel_gt_v'][ind_t])
                        sum_rnd_abs_perc += np.abs(val - df[ind]['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    norm_err = err / float(random_perc)
                    norm_abs_err = abs_err / float(random_abs_perc)
                    df[ind]['norm_err'][ind_t] = norm_err
                    df[ind]['norm_abs_err'][ind_t] = norm_abs_err

                    # df[ind] = df[ind].copy()

            w_pt_avg_l = []
            w_err_avg_l = []
            w_abs_err_avg_l = []
            w_norm_err_avg_l = []
            w_norm_abs_err_avg_l = []
            w_acc_avg_l = []

            w_pt_std_l = []
            w_err_std_l = []
            w_abs_err_std_l = []
            w_norm_err_std_l = []
            w_norm_abs_err_std_l = []
            w_acc_std_l = []

            w_pt_avg_dict = collections.defaultdict()
            w_err_avg_dict = collections.defaultdict()
            w_abs_err_avg_dict = collections.defaultdict()
            w_norm_err_avg_dict = collections.defaultdict()
            w_norm_abs_err_avg_dict = collections.defaultdict()
            w_acc_avg_dict = collections.defaultdict()

            w_pt_std_dict = collections.defaultdict()
            w_err_std_dict = collections.defaultdict()
            w_abs_err_std_dict = collections.defaultdict()
            w_norm_err_std_dict = collections.defaultdict()
            w_norm_abs_err_std_dict = collections.defaultdict()
            w_acc_std_dict = collections.defaultdict()

            all_w_pt_list = []
            all_w_err_list = []
            all_w_abs_err_list = []
            all_w_norm_err_list = []
            all_w_norm_abs_err_list = []
            all_w_acc_list = []

            all_w_cyn_list = []
            all_w_gull_list = []
            w_cyn_avg_l = []
            w_gull_avg_l = []
            w_cyn_std_l = []
            w_gull_std_l = []
            w_cyn_avg_dict = collections.defaultdict()
            w_gull_avg_dict = collections.defaultdict()
            w_cyn_std_dict = collections.defaultdict()
            w_gull_std_dict = collections.defaultdict()
            for ind in ind_l:

                df_m = df[ind].copy()
                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]

                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    w_abs_err_list = list(df_tmp['abs_err'])
                    w_norm_err_list = list(df_tmp['norm_err'])
                    w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    w_cyn_list = list(df_tmp['cyn'])
                    w_gull_list = list(df_tmp['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])
                    w_acc_list = []
                    # w_ind_acc_list
                    acc_c = 0
                    nacc_c = 0

                    all_w_pt_list += list(df_tmp['rel_v'])
                    all_w_err_list += list(df_tmp['err'])
                    all_w_abs_err_list += list(df_tmp['abs_err'])
                    all_w_norm_err_list += list(df_tmp['norm_err'])
                    all_w_norm_abs_err_list += list(df_tmp['norm_abs_err'])
                    all_w_cyn_list += list(df_tmp['cyn'])
                    all_w_gull_list += list(df_tmp['gull'])
                    all_w_acc_list += list(w_acc_list)

                    w_pt_avg_l.append(np.mean(w_pt_list))
                    w_err_avg_l.append(np.mean(w_err_list))
                    w_abs_err_avg_l.append(np.mean(w_abs_err_list))
                    w_norm_err_avg_l.append(np.mean(w_norm_err_list))
                    w_norm_abs_err_avg_l.append(np.mean(w_norm_abs_err_list))
                    w_cyn_avg_l.append(np.mean(w_cyn_list))
                    w_gull_avg_l.append(np.mean(w_gull_list))
                    # w_acc_avg_l.append(w_ind_acc_list)

                    w_pt_std_l.append(np.std(w_pt_list))
                    w_err_std_l.append(np.std(w_err_list))
                    w_abs_err_std_l.append(np.std(w_abs_err_list))
                    w_norm_err_std_l.append(np.std(w_norm_err_list))
                    w_norm_abs_err_std_l.append(np.std(w_norm_abs_err_list))
                    w_cyn_std_l.append(np.std(w_cyn_list))
                    w_gull_std_l.append(np.std(w_gull_list))
                    # w_acc_std_l.append(np.std(w_ind_acc_list))


                    w_pt_avg_dict[t_id] = np.mean(w_pt_list)
                    w_err_avg_dict[t_id] = np.mean(w_err_list)
                    w_abs_err_avg_dict[t_id] = np.mean(w_abs_err_list)
                    w_norm_err_avg_dict[t_id] = np.mean(w_norm_err_list)
                    w_norm_abs_err_avg_dict[t_id] = np.mean(w_norm_abs_err_list)
                    w_cyn_avg_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_avg_dict[t_id] = np.mean(w_gull_list)
                    # w_acc_avg_dict[t_id] = w_ind_acc_list

                    w_pt_std_dict[t_id] = np.std(w_pt_list)
                    w_err_std_dict[t_id] = np.std(w_err_list)
                    w_abs_err_std_dict[t_id] = np.std(w_abs_err_list)
                    w_norm_err_std_dict[t_id] = np.std(w_norm_err_list)
                    w_norm_abs_err_std_dict[t_id] = np.std(w_norm_abs_err_list)
                    w_cyn_std_dict[t_id] = np.std(w_cyn_list)
                    w_gull_std_dict[t_id] = np.std(w_gull_list)
                    w_acc_std_dict[t_id] = np.std(w_acc_list)
                    # ind_
            ##################################################


            # fig_f = True
            fig_f = False
            # fig_f_1 = True
            fig_f_1 = False
            fig_f_together = True
            if fig_f == True:

                df_tmp = pd.DataFrame({'val': all_w_acc_list})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                mplpl.hist(list(df_tmp['val']), weights=weights, color='g')
                # mplpl.hist(list(df_tmp['val']), normed=1, color='g')


                mplpl.xlim([-1.5, 1.5])
                mplpl.ylim([0, 1.5])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Readers accuracy', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(all_w_acc_list), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_acc_density'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_acc_density| alt text| width = 500px}}')

                # df_tmp = pd.DataFrame(w_pt_avg_l, col=['val'])
                df_tmp = pd.DataFrame({'val': w_acc_avg_l})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                # mplpl.hist(list(df_tmp['val']), weights=weights, color=col)
                mplpl.hist(list(df_tmp['val']), normed=1, color='g')

                # mplpl.plot(gt_set, pt_mean,  color='k')
                mplpl.xlim([-1.5, 1.5])
                mplpl.ylim([0, 5])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Individual readers accuracy', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(w_acc_avg_l), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_avg_acc_pt'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_avg_acc_pt| alt text| width = 500px}}')

                df_tmp = pd.DataFrame({'val': all_w_pt_list})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                mplpl.hist(list(df_tmp['val']), weights=weights, color='c')
                # mplpl.hist(list(df_tmp['val']), normed=1, color='g')


                mplpl.xlim([-1.5, 1.5])
                mplpl.ylim([0, 1.5])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Readers percevied truth value', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(all_w_pt_list), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_pt_density'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_pt_density| alt text| width = 500px}}')

                # df_tmp = pd.DataFrame(w_pt_avg_l, col=['val'])
                df_tmp = pd.DataFrame({'val': w_pt_avg_l})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                # mplpl.hist(list(df_tmp['val']), weights=weights, color=col)
                mplpl.hist(list(df_tmp['val']), normed=1, color='c')

                # mplpl.plot(gt_set, pt_mean,  color='k')
                mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([0, 3.5])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Individual readers percevied truth value', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(w_pt_avg_l), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_avg_pt_density'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_avg_pt_density| alt text| width = 500px}}')

                df_tmp = pd.DataFrame({'val': w_err_avg_l})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                mplpl.hist(list(df_tmp['val']), normed=1, color='y')

                mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([0, 3.5])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Individual readers perception bias value', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(w_err_avg_l), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_avg_pb_density'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_avg_pb_density| alt text| width = 500px}}')

                df_tmp = pd.DataFrame({'val': w_abs_err_avg_l})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                mplpl.hist(list(df_tmp['val']), normed=1, color='y')

                mplpl.xlim([0, 1.2])
                mplpl.ylim([0, 5])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Individual readers absolute perception bias value', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(w_abs_err_avg_l), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_avg_apb_density'
                mplpl.savefig(pp, format='png')
                mplpl.figure()
                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_avg_apb_density| alt text| width = 500px}}||')

                df_tmp = pd.DataFrame({'val': w_gull_avg_l})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                mplpl.hist(list(df_tmp['val']), normed=1, color='m')

                mplpl.xlim([0, 1.2])
                mplpl.ylim([0, 10])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Individual readers gullibility value', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(w_gull_avg_l), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_avg_gull_density'
                mplpl.savefig(pp, format='png')
                mplpl.figure()
                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_avg_gull_density| alt text| width = 500px}}||')

                df_tmp = pd.DataFrame({'val': w_cyn_avg_l})
                weights = []
                weights.append(np.ones_like(list(df_tmp['val'])) / float(len(df_tmp)))
                col = 'r'
                try:
                    df_tmp['val'].plot(kind='kde', lw=4, color=col)
                except:
                    print('hmm')

                mplpl.hist(list(df_tmp['val']), normed=1, color='k')

                mplpl.xlim([0, 1.2])
                mplpl.ylim([0, 10])
                mplpl.ylabel('Density', fontsize=18)
                mplpl.xlabel('Individual readers cynicallity value', fontsize=18)
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(w_cyn_avg_l), 4)))
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_avg_cyn_density'
                mplpl.savefig(pp, format='png')
                mplpl.figure()
                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_avg_cyn_density| alt text| width = 500px}}||\n\n')

                ########################

                tweet_l_sort = sorted(w_pt_avg_dict, key=w_pt_avg_dict.get, reverse=False)
                pt_l = []
                for t_id in tweet_l_sort:
                    pt_l.append(w_pt_avg_dict[t_id])

                mplpl.scatter(range(len(pt_l)), pt_l, s=40, color='c', marker='o', label='All users')
                # mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([-1, 1])
                mplpl.ylabel('Perception truth value (PTL)', fontsize=18)
                mplpl.xlabel('Ranked readers according PTL', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(pt_l), 4)))
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_pt_pt'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_pt_pt| alt text| width = 500px}}')

                tweet_l_sort = sorted(w_acc_avg_dict, key=w_acc_avg_dict.get, reverse=False)
                acc_l = []
                for t_id in tweet_l_sort:
                    acc_l.append(w_acc_avg_dict[t_id])

                mplpl.scatter(range(len(acc_l)), acc_l, s=40, color='g', marker='o', label='All users')
                # mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([-1, 1])
                mplpl.ylabel('Accuracy', fontsize=18)
                mplpl.xlabel('Ranked readers according accuracy', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(acc_l), 4)))
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_acc_acc'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_acc_acc| alt text| width = 500px}}')

                tweet_l_sort = sorted(w_err_avg_dict, key=w_err_avg_dict.get, reverse=False)
                err_l = []
                for t_id in tweet_l_sort:
                    err_l.append(w_err_avg_dict[t_id])

                mplpl.scatter(range(len(err_l)), err_l, s=40, color='y', marker='o', label='All users')
                # mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([-2, 2])
                mplpl.ylabel('Perception bias (PB)', fontsize=18)
                mplpl.xlabel('Ranked readers according PB', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(err_l), 4)))
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_err_err'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_err_err| alt text| width = 500px}}')

                tweet_l_sort = sorted(w_abs_err_avg_dict, key=w_abs_err_avg_dict.get, reverse=False)
                abs_err_l = []
                for t_id in tweet_l_sort:
                    abs_err_l.append(w_abs_err_avg_dict[t_id])

                mplpl.scatter(range(len(abs_err_l)), abs_err_l, s=40, color='y', marker='o', label='All users')
                # mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([0, 2])
                mplpl.ylabel('Absolute perception bias (APB)', fontsize=18)
                mplpl.xlabel('Ranked news stories according APB', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(abs_err_l), 4)))
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_abs-err_abs-err'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_abs-err_abs-err| alt text| width = 500px}}')

                tweet_l_sort = sorted(w_gull_avg_dict, key=w_gull_avg_dict.get, reverse=False)
                gull_l = []
                for t_id in tweet_l_sort:
                    gull_l.append(w_gull_avg_dict[t_id])

                mplpl.scatter(range(len(gull_l)), gull_l, s=40, color='m', marker='o', label='All users')
                # mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([0, 1])
                mplpl.ylabel('Gullibility', fontsize=18)
                mplpl.xlabel('Ranked readers according gullibility', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(gull_l), 4)))
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_gull_gull'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_gull_gull| alt text| width = 500px}}')

                tweet_l_sort = sorted(w_cyn_avg_dict, key=w_cyn_avg_dict.get, reverse=False)
                cyn_l = []
                for t_id in tweet_l_sort:
                    cyn_l.append(w_cyn_avg_dict[t_id])

                mplpl.scatter(range(len(cyn_l)), cyn_l, s=40, color='k', marker='o', label='All users')
                # mplpl.xlim([-1.2, 1.2])
                mplpl.ylim([0, 1])
                mplpl.ylabel('Cynicality', fontsize=18)
                mplpl.xlabel('Ranked readers according cynicality', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean(cyn_l), 4)))
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_cyn_cyn'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_cyn_cyn| alt text| width = 500px}}||\n\n')

                # outF.write('|| Table ||\n\n')

                # mplpl.show()
                # exit()
                #####################################################33


                num_bins = len(pt_l)
                counts, bin_edges = np.histogram(pt_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='c', lw=5, label='')
                mplpl.ylabel('CDF', fontsize=18)
                mplpl.xlabel('Perception truth value (PTL)', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name)
                #         mplpl.legend(loc="upper right")
                mplpl.xlim([-1, 1])
                mplpl.ylim([0, 1])
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_user_pt_cdf'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_user_pt_cdf| alt text| width = 500px}}')

                num_bins = len(acc_l)
                counts, bin_edges = np.histogram(acc_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='g', lw=5, label='')
                mplpl.ylabel('CDF', fontsize=18)
                mplpl.xlabel('Accuracy', fontsize=18)
                mplpl.title(data_name)
                #         mplpl.legend(loc="upper right")
                mplpl.xlim([0, 1])
                mplpl.ylim([0, 1])
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_user_acc_cdf'
                mplpl.savefig(pp, format='png')
                mplpl.figure()
                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_user_acc_cdf| alt text| width = 500px}}')

                num_bins = len(err_l)
                counts, bin_edges = np.histogram(err_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='y', lw=5, label='')
                mplpl.ylabel('CDF', fontsize=18)
                mplpl.xlabel('Perception bias (PB)', fontsize=18)
                mplpl.title(data_name)
                #         mplpl.legend(loc="upper right")
                mplpl.xlim([-1, 1])
                mplpl.ylim([0, 1])
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_user_err_cdf'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_user_err_cdf| alt text| width = 500px}}')

                num_bins = len(abs_err_l)
                counts, bin_edges = np.histogram(abs_err_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='y', lw=5, label='All users')
                mplpl.ylabel('CDF', fontsize=18)
                mplpl.xlabel('Absolute perception bias (APB)', fontsize=18)
                mplpl.title(data_name)
                #         mplpl.legend(loc="upper right")
                mplpl.xlim([0, 1])
                mplpl.ylim([0, 1])
                mplpl.grid()
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_user_abs-err_cdf'
                mplpl.savefig(pp, format='png')
                mplpl.figure()
                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_user_abs-err_cdf| alt text| width = 500px}}')

                # outF.write('|| Table ||\n\n')

                num_bins = len(gull_l)
                counts, bin_edges = np.histogram(gull_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='m', lw=5, label='')
                mplpl.ylabel('CDF', fontsize=18)
                mplpl.xlabel('Gullibility', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name)
                #         mplpl.legend(loc="upper right")
                mplpl.xlim([0, 1])
                mplpl.ylim([0, 1])
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_user_gull_cdf'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_user_gull_cdf| alt text| width = 500px}}')

                num_bins = len(cyn_l)
                counts, bin_edges = np.histogram(cyn_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='k', lw=5, label='')
                mplpl.ylabel('CDF', fontsize=18)
                mplpl.xlabel('Cynicality', fontsize=18)
                mplpl.grid()
                mplpl.title(data_name)
                #         mplpl.legend(loc="upper right")
                mplpl.xlim([0, 1])
                mplpl.ylim([0, 1])
                pp = remotedir + '/fig/fig_exp1/user_based/initial/' + data_n + '_user_cyn_cdf'
                mplpl.savefig(pp, format='png')
                mplpl.figure()

                outF.write(
                    '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/user_based/initial/' + data_n + '_user_cyn_cdf| alt text| width = 500px}}||\n')


            elif fig_f_together == True:

                ####ptl_cdf
                mplpl.rcParams['figure.figsize'] = 5.4, 3.2
                mplpl.rc('xtick', labelsize='x-large')
                mplpl.rc('ytick', labelsize='x-large')
                mplpl.rc('legend', fontsize='small')
                w_err_avg_dict
                # tweet_l_sort = sorted(w_norm_abs_err_avg_dict, key=w_norm_abs_err_avg_dict.get, reverse=False)
                tweet_l_sort = sorted(w_err_avg_dict, key=w_err_avg_dict.get, reverse=False)
                acc_l = []
                for t_id in tweet_l_sort:
                    acc_l.append(w_err_avg_dict[t_id])

                num_bins = len(acc_l)
                counts, bin_edges = np.histogram(acc_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5, label=data_name)

        legend_properties = {'weight': 'bold'}

        #
        mplpl.ylabel('CDF', fontsize=20, fontweight='bold')
        mplpl.xlabel('Mean Perception Bias', fontsize=20, fontweight='bold')
        mplpl.legend(loc="upper left", prop=legend_properties, fontsize='small', ncol=1)
        # mplpl.title(data_name)
        # mplpl.legend(loc="upper left",fontsize = 'large')
        mplpl.xlim([-1.5, 1.5])
        mplpl.ylim([0, 1])
        mplpl.grid()
        mplpl.subplots_adjust(bottom=0.24)
        mplpl.subplots_adjust(left=0.18)
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/NAPB_cdf_alldataset'
        pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/MPB_cdf_alldataset_new_1'
        mplpl.savefig(pp + '.pdf', format='pdf')
        mplpl.savefig(pp + '.png', format='png')
    if args.t == "AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_apb_cdf_toghether":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []


        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'

        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []


        # balance_f = 'balanced'


        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        # for dataset in ['snopes','snopes_ssi', 'snopes_incentive', 'snopes_nonpol', 'politifact', 'mia']:
        for dataset in ['snopes', 'snopes_ssi', 'snopes_incentive']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                tweet_id = 100010
                publisher_name = 110
                tweet_popularity = {}
                tweet_text_dic = {}
                for input_file in [input_rumor, input_non_rumor]:
                    for line in input_file:
                        line.replace('\n', '')
                        line_splt = line.split('\t')
                        tweet_txt = line_splt[1]
                        tweet_link = line_splt[1]
                        tweet_id += 1
                        publisher_name += 1
                        tweet_popularity[tweet_id] = int(line_splt[2])
                        tweet_text_dic[tweet_id] = tweet_txt

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # if int(tweet_id) in [100012, 100016, 100053, 100038, 100048]:
                        #     tweet_lable_dict[tweet_id] = 'undecided'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                col = 'r'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
                news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes'or dataset == 'snopes_ssi'or dataset == 'snopes_incentive':
                claims_list = []
                col = 'r'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
                news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'politifact':
                col = 'g'

                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
                news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                col = 'purple'

                data_name = 'Snopes'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                col = 'g'

                data_name = 'Snopes_incentive'
            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                col = 'k'

                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1,2,3]
                ind_l = [1]
                col = 'green'

                data_name = 'Snopes\nnonpolitical'
            elif dataset == 'politifact':
                col = 'c'
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                col = 'orange'
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()
                df[ind].loc[:, 'abs_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_abs_err'] = df[ind]['tweet_id'] * 0.0

                groupby_ftr = 'tweet_id'
                grouped = df[ind].groupby(groupby_ftr, sort=False)
                grouped_sum = df[ind].groupby(groupby_ftr, sort=False).sum()

                for ind_t in df[ind].index.tolist():
                    t_id = df[ind]['tweet_id'][ind_t]
                    err = df[ind]['err'][ind_t]
                    abs_err = np.abs(err)
                    df[ind]['abs_err'][ind_t] = abs_err
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += (val - df[ind]['rel_gt_v'][ind_t])
                        sum_rnd_abs_perc += np.abs(val - df[ind]['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    norm_err = err / float(random_perc)
                    norm_abs_err = abs_err / float(random_abs_perc)
                    df[ind]['norm_err'][ind_t] = norm_err
                    df[ind]['norm_abs_err'][ind_t] = norm_abs_err

                    # df[ind] = df[ind].copy()

            w_pt_avg_l = []
            w_err_avg_l = []
            w_abs_err_avg_l = []
            w_norm_err_avg_l = []
            w_norm_abs_err_avg_l = []
            w_acc_avg_l = []

            w_pt_std_l = []
            w_err_std_l = []
            w_abs_err_std_l = []
            w_norm_err_std_l = []
            w_norm_abs_err_std_l = []
            w_acc_std_l = []

            w_pt_avg_dict = collections.defaultdict()
            w_err_avg_dict = collections.defaultdict()
            w_abs_err_avg_dict = collections.defaultdict()
            w_norm_err_avg_dict = collections.defaultdict()
            w_norm_abs_err_avg_dict = collections.defaultdict()
            w_acc_avg_dict = collections.defaultdict()

            w_pt_std_dict = collections.defaultdict()
            w_err_std_dict = collections.defaultdict()
            w_abs_err_std_dict = collections.defaultdict()
            w_norm_err_std_dict = collections.defaultdict()
            w_norm_abs_err_std_dict = collections.defaultdict()
            w_acc_std_dict = collections.defaultdict()

            all_w_pt_list = []
            all_w_err_list = []
            all_w_abs_err_list = []
            all_w_norm_err_list = []
            all_w_norm_abs_err_list = []
            all_w_acc_list = []

            all_w_cyn_list = []
            all_w_gull_list = []
            w_cyn_avg_l = []
            w_gull_avg_l = []
            w_cyn_std_l = []
            w_gull_std_l = []
            w_cyn_avg_dict = collections.defaultdict()
            w_gull_avg_dict = collections.defaultdict()
            w_cyn_std_dict = collections.defaultdict()
            w_gull_std_dict = collections.defaultdict()
            for ind in ind_l:

                df_m = df[ind].copy()
                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]

                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    # w_abs_err_list = list(df_tmp['abs_err'])
                    w_abs_err_list = list(df_tmp['susc'])
                    w_norm_err_list = list(df_tmp['norm_err'])
                    w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    df_cyn = df_tmp[df_tmp['cyn'] > 0]
                    df_gull = df_tmp[df_tmp['gull'] > 0]

                    w_cyn_list = list(df_cyn['cyn'])
                    w_gull_list = list(df_gull['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])
                    w_acc_list = []
                    # w_ind_acc_list
                    acc_c = 0
                    nacc_c = 0

                    # w_acc_avg_l.append(w_ind_acc_list)

                    w_pt_std_l.append(np.std(w_pt_list))
                    w_err_std_l.append(np.std(w_err_list))
                    w_abs_err_std_l.append(np.std(w_abs_err_list))
                    w_norm_err_std_l.append(np.std(w_norm_err_list))
                    w_norm_abs_err_std_l.append(np.std(w_norm_abs_err_list))
                    w_cyn_std_l.append(np.std(w_cyn_list))
                    w_gull_std_l.append(np.std(w_gull_list))
                    # w_acc_std_l.append(np.std(w_ind_acc_list))


                    w_pt_avg_dict[t_id] = np.mean(w_pt_list)
                    w_err_avg_dict[t_id] = np.mean(w_err_list)
                    w_abs_err_avg_dict[t_id] = np.mean(w_abs_err_list)
                    w_norm_err_avg_dict[t_id] = np.mean(w_norm_err_list)
                    w_norm_abs_err_avg_dict[t_id] = np.mean(w_norm_abs_err_list)
                    w_cyn_avg_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_avg_dict[t_id] = np.mean(w_gull_list)
                    # w_acc_avg_dict[t_id] = w_ind_acc_list




            # fig_f = True
            fig_f = False
            # fig_f_1 = True
            fig_f_1 = False
            fig_f_together = True


            # fig_f_together == True:
            out_dict = w_abs_err_avg_dict
            # out_dict = w_gull_avg_dict
            # out_dict = w_cyn_avg_dict
            ####ptl_cdf
            mplpl.rcParams['figure.figsize'] = 4.5, 2.5
            mplpl.rc('xtick', labelsize='large')
            mplpl.rc('ytick', labelsize='large')
            mplpl.rc('legend', fontsize='small')
            w_err_avg_dict
            # tweet_l_sort = sorted(w_norm_abs_err_avg_dict, key=w_norm_abs_err_avg_dict.get, reverse=False)
            tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
            # tweet_l_sort = [x for x in tweet_l_sort if x >= 0 or x < 0]
            acc_l = []
            for t_id in tweet_l_sort:
                if out_dict[t_id] >= 0 or out_dict[t_id] < 0:
                    acc_l.append(out_dict[t_id])

            num_bins = len(acc_l)
            counts, bin_edges = np.histogram(acc_l, bins=num_bins, normed=True)
            cdf = np.cumsum(counts)
            scale = 1.0 / cdf[-1]
            ncdf = scale * cdf
            mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5, label=data_name)

        legend_properties = {'weight': 'bold'}

        #
        mplpl.ylabel('CDF', fontsize=20, fontweight='bold')
        mplpl.xlabel('Total Perception Bias', fontsize=20, fontweight = 'bold')
        # mplpl.xlabel('False Positive Bias', fontsize=20, fontweight = 'bold')
        # mplpl.xlabel('False Negative Bias', fontsize=20, fontweight='bold')
        mplpl.legend(loc="lower right", prop=legend_properties, fontsize='small', ncol=1)
        # mplpl.title(data_name)
        # mplpl.legend(loc="upper left",fontsize = 'large')
        mplpl.xlim([0, 2])
        mplpl.ylim([0, 1])
        mplpl.grid()
        mplpl.subplots_adjust(bottom=0.24)
        mplpl.subplots_adjust(left=0.18)
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/NAPB_cdf_alldataset'
        pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/APB_cdf_alldataset_new_1'
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/FPB_cdf_alldataset_new_1'
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/FNB_cdf_alldataset_new_1'
        mplpl.savefig(pp + '.pdf', format='pdf')
        mplpl.savefig(pp + '.png', format='png')

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_true-false(gt-pt)_news_ktop_nptl_scatter_fig1":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []

        # dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'
        dataset = 'snopes_nonpol'
        # dataset = 'snopes_ssi'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []

        if dataset=='mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'


            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                     + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets','r')



            exp1_list = sample_tweets_exp1
            tweet_id = 100010
            publisher_name = 110
            tweet_popularity = {}
            tweet_text_dic = {}
            for input_file in [input_rumor, input_non_rumor]:
                for line in input_file:
                    line.replace('\n', '')
                    line_splt = line.split('\t')
                    tweet_txt = line_splt[1]
                    tweet_link = line_splt[1]
                    tweet_id += 1
                    publisher_name += 1
                    tweet_popularity[tweet_id] = int(line_splt[2])
                    tweet_text_dic[tweet_id] = tweet_txt

            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor= {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id)<100060:
                    tweet_lable_dict[tweet_id]='rumor'
                else:
                    tweet_lable_dict[tweet_id]='non-rumor'


        if dataset == 'snopes_nonpol':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            tweet_topic_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims_1.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                topic = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_topic_dict[tweet_id] = topic

        if dataset == 'snopes' or dataset == 'snopes_ssi':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable


        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name


        # dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'
        dataset = 'snopes_nonpol'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []


        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []


        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_nonpol','snopes_ssi', 'snopes_nonpol', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                tweet_topic_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims_1.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    topic = line_splt[5]
                    tweet_topic_dict[tweet_id] = topic
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            news_cat_list_tf = [4, 2, 3, 1]
            t_f_dict_len = collections.defaultdict(int)
            t_f_dict = {}
            if dataset == 'snopes' or dataset == 'snopes_nonpol'or dataset == 'snopes_ssi':
                news_cat_list_t_f = [['FALSE', 'MOSTLY FALSE'], ['MOSTLY TRUE', 'TRUE']]
            if dataset == 'politifact':
                news_cat_list_t_f = [['pants-fire', 'false', 'mostly-false'], ['mostly-true', 'true']]

            if dataset == 'mia':
                news_cat_list_t_f = [['rumor'], ['non-rumor']]

            w_fnb_dict = collections.defaultdict()
            w_fpb_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()
            tweet_avg_dem = collections.defaultdict()
            tweet_avg_rep = collections.defaultdict()
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                    # inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    # if t_id == 1367:
                    #     continue
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    if t_id == 1116:
                        print(df_m['text'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []



                    w_sus_list = list(df_tmp['susc'])
                    # w_norm_err_list = list(df_tmp['norm_err'])
                    # w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    # w_cyn_list = list(df_tmp['cyn'])
                    # w_gull_list = list(df_tmp['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])

                    df_cyn = df_tmp[df_tmp['cyn'] > 0]
                    df_gull = df_tmp[df_tmp['gull'] > 0]

                    w_cyn_list_t = list(df_cyn['cyn'])
                    w_gull_list_t = list(df_gull['gull'])

                    w_cyn_list = []
                    w_gull_list = []

                    for tt in w_cyn_list_t:
                        if tt > 0 or tt <= 0:
                            w_cyn_list.append(tt)

                    for tt in w_gull_list_t:
                        if tt > 0 or tt <= 0:
                            w_gull_list.append(tt)
                    w_fnb_dict[t_id] = np.mean(w_cyn_list)
                    w_fpb_dict[t_id] = np.mean(w_gull_list)
                    w_apb_dict[t_id] = np.mean(w_sus_list)

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))
                    # ['FALSE-FALSE', 'FALSE-TRUE', 'TRUE-FALSE', 'TRUE-TRUE']

                    if tweet_lable_dict[t_id] in news_cat_list_t_f[0] and np.mean(df_tmp['rel_v']) < 0:
                        t_f_dict[t_id] = 4
                        t_f_dict_len[4] += 1
                    elif tweet_lable_dict[t_id] in news_cat_list_t_f[0] and np.mean(df_tmp['rel_v']) > 0:
                        t_f_dict[t_id] = 2
                        t_f_dict_len[2] += 1
                    elif tweet_lable_dict[t_id] in news_cat_list_t_f[1] and np.mean(df_tmp['rel_v']) > 0:
                        t_f_dict[t_id] = 1
                        t_f_dict_len[1] += 1
                    elif tweet_lable_dict[t_id] in news_cat_list_t_f[1] and np.mean(df_tmp['rel_v']) < 0:
                        t_f_dict[t_id] = 3
                        t_f_dict_len[3] += 1
                    else:
                        t_f_dict[t_id] = -10
                        t_f_dict_len[-10] += 1

                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    val_list = list(df_tmp['rel_v'])

                    tweet_avg_dem[t_id] = np.mean(dem_val_list)
                    tweet_avg_rep[t_id] = np.mean(rep_val_list)

                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])



                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)

                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
                    # tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    # tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)

            ##################################################
            gt_fpb = []
            pt_fpb = []
            gt_fnb = []
            pt_fnb = []
            gt_pt = []
            pt_pt = []
            len_cat_dict = {}
            # if dataset=='snopes' or dataset=='politifact':
            for cat in news_cat_list_tf:
                len_cat_dict[cat] = t_f_dict_len[cat]
            # elif dataset=='mia':
            #     for cat in news_cat_list_tf:
            #         if cat=='rumor':
            #             len_cat_dict[cat]=t_f_dict_len[cat]
            #         else:
            #             len_cat_dict[cat] = t_f_dict_len[cat]
            tweet_vote_sort = sorted(tweet_avg, key=tweet_avg.get, reverse=False)
            tweet_pick = tweet_vote_sort[:2]
            for t_id in tweet_pick:
                print(t_id)
                gt_pt.append(tweet_gt_var[t_id])
                pt_pt.append(tweet_avg[t_id])
            # tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)

            tweet_apb_sort = sorted(w_apb_dict, key=w_apb_dict.get, reverse=True)
            tweet_fpb_sort = sorted(w_fpb_dict, key=w_fpb_dict.get, reverse=True)
            tweet_fnb_sort = sorted(w_fnb_dict, key=w_fnb_dict.get, reverse=True)
            tweet_ideological_mpb_sort = sorted(tweet_avg_group, key=tweet_avg_group.get, reverse=True)
            # tweet_vote_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            # tweet_vote_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)



            top_pic_des_FPB = tweet_fpb_sort[:1]
            for t_id in top_pic_des_FPB:
                print(t_id)
                print(tweet_txt_dict[t_id])
                gt_fpb.append(tweet_gt_var[t_id])
                pt_fpb.append(tweet_avg[t_id])

            top_pic_des_FNB = tweet_fnb_sort[:1]
            for t_id in top_pic_des_FNB:
                print(t_id)
                print(tweet_txt_dict[t_id])
                gt_fnb.append(tweet_gt_var[t_id])
                pt_fnb.append(tweet_avg[t_id])
            #
            # cc = 0
            #
            # tpb_all = collections.defaultdict(int)
            # for t_id in tweet_apb_sort:
            #     cc += 1
            #     tpb_all[tweet_topic_dict[t_id]] += 1
            #
            # top_tpb = collections.defaultdict(float)
            # for t_id in tweet_apb_sort[:25]:
            #     cc += 1
            #     top_tpb[tweet_topic_dict[t_id]] += 1
            # # + '_' + str(mid_tpb[key])
            # print('----top ----')
            # for key in top_tpb:
            #     print(str(int(100 * (top_tpb[key] / float(tpb_all[key])))) + '\t' + key + '_' + str(
            #         int(100 * (top_tpb[key] / float(tpb_all[key])))) + '%')
            #
            # bot_tpb = collections.defaultdict(float)
            # for t_id in tweet_apb_sort[75:]:
            #     cc += 1
            #     bot_tpb[tweet_topic_dict[t_id]] += 1
            #
            # print('----bottomn ----')
            # for key in bot_tpb:
            #     print(str(int(100 * (bot_tpb[key] / float(tpb_all[key])))) + '\t' + key + '_' + str(
            #         int(100 * (bot_tpb[key] / float(tpb_all[key])))) + '%')
            #
            # mid_tpb = collections.defaultdict(float)
            # for t_id in tweet_apb_sort[25:75]:
            #     cc += 1
            #     mid_tpb[tweet_topic_dict[t_id]] += 1
            #
            # print('----mid ----')
            # for key in mid_tpb:
            #     print(str(int(100 * (mid_tpb[key] / float(tpb_all[key])))) + '\t' + key + '_' + str(
            #         int(100 * (mid_tpb[key] / float(tpb_all[key])))) + '%')
            #
            # # exit()
            # cc = 0
            # for t_id in tweet_ideological_mpb_sort:
            #     cc += 1
            #     print('|| ' + str(cc) + ' || ' + str(t_id) + ' || ' + tweet_txt_dict[t_id] + ' || ' + tweet_lable_dict[t_id]
            #           + ' || ' + tweet_topic_dict[t_id] + ' || ' + str(tweet_avg_group[t_id]) + ' || ' + str(
            #         tweet_avg_dem[t_id]) +
            #           ' || ' + str(tweet_avg_rep[t_id]) + ' || ' + str(w_apb_dict[t_id]) + ' || ' + str(
            #         tweet_avg[t_id]) + '||')
            # exit()

            thr = 10
            thr_list = []
            categ_dict = collections.defaultdict(int)
            categ_dict_n = collections.defaultdict(int)
            len_t = len(tweet_vote_sort)
            k_list = [int(0.1 * len_t), int(0.2 * len_t), int(0.3 * len_t), int(0.4 * len_t), int(0.5 * len_t),
                      int(1 * len_t)]
            count = 0
            for k in k_list:
                thr_list.append(k)
                perc_rnd_l = []
                abs_perc_rnd_l = []
                disputability_l = []
                above_avg = 0
                less_avg = 0
                above_avg_rnd = 0
                less_avg_rnd = 0
                above_avg = 0
                less_avg = 0
                categ_dict[k] = collections.defaultdict(float)
                categ_dict_n[k] = collections.defaultdict(list)
                for j in range(k):
                    for cat_n in news_cat_list_tf:
                        if cat_n == -10:
                            continue
                        if t_f_dict[tweet_vote_sort[j]] == cat_n:
                            categ_dict[k][cat_n] += 1 / float(len_cat_dict[cat_n])

                            categ_dict_n[k][cat_n].append(tweet_vote_sort[j])
            # if dataset=='mia':
            total_data = collections.defaultdict(int)
            for j in categ_dict:
                sum = np.sum(categ_dict[j].values())
                # for cat_n in categ_dict[j]:
                for cat_n in [4, 2, 3, 1]:
                    categ_dict[j][cat_n] = categ_dict[j][cat_n] / sum

                    total_data[j] += len(categ_dict_n[k][cat_n])

            width = 0.03
            pr = -10
            title_l = news_cat_list_tf
            outp = {}
            # news_cat_list = ['pants-fire', 'false', 'mostly_false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            if dataset == 'snopes':
                # col_l = ['b', 'g', 'c', 'y', 'r']
                col_l = ['red', 'magenta', 'lime', 'green']
                marker_l = ['s', '*', 'o', 'd']
                news_cat_list_n = ['GT:FALSE, PT:FALSE', 'GT:FALSE, PT:TRUE', 'GT:TRUE, PT:FALSE', 'GT:TRUE, PT:TRUE']
            if dataset == 'politifact':
                # col_l = ['grey','b', 'g', 'c', 'y', 'r']
                col_l = ['red', 'magenta', 'lime', 'green']

                news_cat_list_n = ['FALSE-FALSE', 'FALSE-TRUE', 'TRUE-FALSE', 'TRUE-TRUE']

            if dataset == 'mia':
                # col_l = ['b', 'r']
                col_l = ['red', 'magenta', 'lime', 'green']
                news_cat_list_n = ['FALSE-FALSE', 'FALSE-TRUE', 'TRUE-FALSE', 'TRUE-TRUE']
            count = 0
            Y = [0] * len(thr_list)
            mplpl.rcParams['figure.figsize'] = 4.8, 4
            mplpl.rc('xtick', labelsize='large')
            mplpl.rc('ytick', labelsize='large')
            mplpl.rc('legend', fontsize='small')
            # for cat_m in news_cat_list_tf:
            #     count+=1
            #     outp[cat_m] = []
            #     for i in thr_list:
            #         outp[cat_m].append(categ_dict[i][cat_m])
            #     mplpl.bar([0.1, 0.2, 0.3, 0.4,0.5,0.6], outp[cat_m], width, bottom= np.array(Y), color=col_l[count-1], label=news_cat_list_n[count-1])
            #     Y = np.array(Y) + np.array(outp[cat_m])
            gt_l_t = []
            pt_l_t = []
            for i in [int(1 * len_t)]:
                tweet_id_list = tweet_vote_sort
                for t_id in tweet_id_list:
                    gt_l_t.append(tweet_gt_var[t_id])
                    pt_l_t.append(tweet_avg[t_id])

            #
            # for cat_m in news_cat_list_tf:
            #     count+=1
            #     outp[cat_m] = []
            #     gt_l_t = []
            #     pt_l_t = []
            #     for i in [int(1*len_t)]:
            #         tweet_id_list = categ_dict_n[i][cat_m]
            #         for t_id in tweet_id_list:
            #             gt_l_t.append(tweet_gt_var[t_id])
            #             pt_l_t.append(tweet_avg[t_id])
            #
            #     m_label = str(np.round(len(tweet_id_list) / float(total_data[int(1 * len_t)]),2))

            # mplpl.scatter(gt_l_t, pt_l_t,c=col_l[count-1],marker=marker_l[count-1], s=60, label = m_label)
            # mplpl.scatter(pt_l_t,gt_l_t, c=col_l[count - 1], marker=marker_l[count - 1], s=60, label=m_label)


            mplpl.scatter(gt_fnb, pt_fnb, c='r', marker='s', s=200)  # , label=m_label)
            mplpl.scatter(gt_fpb, pt_fpb, c='r', marker='d', s=200)  # , label=m_label)
            mplpl.scatter(gt_pt, pt_pt, c='orange', marker='^', s=200)  # , label=m_label)
            mplpl.scatter(gt_l_t, pt_l_t, c='g', marker='o', s=60)  # , label=m_label)

            mplpl.plot([-1.1, 1.1], [-1.1, 1.1], c='k', linewidth=4)
            # mplpl.plot([0,0],[-1.1, 1.1],c='k',linewidth=4)

            mplpl.xlim([-1.04, 1.04])
            mplpl.ylim([-1.04, 1.04])

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16,
                    }

            font_1 = {'family': 'serif',
                      'color': 'darkblue',
                      'weight': 'normal',
                      'size': 16,
                      }

            font_t = {'family': 'serif',
                      'color': 'darkred',
                      'weight': 'bold',
                      'size': 12,
                      }

            font_t_1 = {'family': 'serif',
                        'color': 'darkblue',
                        'weight': 'normal',
                        'size': 12,
                        }
            # mplpl.text(-0.7, 0.85, ' Mostly deserve to pick', fontdict=font_t)
            # # mplpl.text(-0.3, -0.9, '# news has negative perception truth value', fontdict=font_t_1)
            # mplpl.annotate('', xy=(-1, 0.38), xytext=(0, 0.82),
            #             arrowprops=dict(facecolor='r', shrink=0.1))
            #
            # mplpl.annotate('', xy=(1, -0.25), xytext=(0, 0.82),
            #             arrowprops=dict(facecolor='r', shrink=0.1))

            mplpl.subplots_adjust(bottom=0.32)

            mplpl.subplots_adjust(left=0.2)
            mplpl.title(data_name)
            labels = ['-1\nFalse', '-.05\nMostly\nFalse', '0\nMixture', '0.5\nMostly\n True', '1\nTrue']
            x = [-1, -.5, 0, 0.5, 1]
            mplpl.xticks(x, labels)

            # mplpl.ylabel('Composition of news stories \n in different quadrants', fontsize=14,fontweight = 'bold')
            mplpl.xlabel('Ground Truth Level', fontsize=18, fontweight='bold')
            mplpl.ylabel('Perceived Truth Level', fontsize=18, fontweight='bold')
            # mplpl.xlabel('Top k news stories ranked by NAPB', fontsize=18)

            # mplpl.legend(loc="center", ncol=2, fontsize='small')



            mplpl.grid()
            pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_gt_pt_compos_true-false_scatter_1'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_gt_pt_compos_true-false_scatter_1_weighted'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_napb_compos_true-false_scatter'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_vote_compos_true-false_scatter'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_disp_compos_true-false_scatter'
            mplpl.savefig(pp + '.pdf', format='pdf')
            mplpl.savefig(pp + '.png', format='png')

            mplpl.figure()

            exit()

    if args.t == "AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_disp_cdf_toghether":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []




        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes','snopes_ssi', 'snopes_nonpol', 'politifact', 'mia']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                tweet_id = 100010
                publisher_name = 110
                tweet_popularity = {}
                tweet_text_dic = {}
                for input_file in [input_rumor, input_non_rumor]:
                    for line in input_file:
                        line.replace('\n', '')
                        line_splt = line.split('\t')
                        tweet_txt = line_splt[1]
                        tweet_link = line_splt[1]
                        tweet_id += 1
                        publisher_name += 1
                        tweet_popularity[tweet_id] = int(line_splt[2])
                        tweet_text_dic[tweet_id] = tweet_txt

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # if int(tweet_id) in [100012, 100016, 100053, 100038, 100048]:
                        #     tweet_lable_dict[tweet_id] = 'undecided'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                col = 'r'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
                news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                col = 'r'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
                news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'politifact':
                col = 'g'

                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
                news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                col = 'purple'

                data_name = 'Snopes'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                col = 'k'

                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                col = 'green'

                data_name = 'Snopes\nnonpolitic'
            elif dataset == 'politifact':
                col = 'c'
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                col = 'orange'
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()
                df[ind].loc[:, 'abs_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_abs_err'] = df[ind]['tweet_id'] * 0.0

                groupby_ftr = 'tweet_id'
                grouped = df[ind].groupby(groupby_ftr, sort=False)
                grouped_sum = df[ind].groupby(groupby_ftr, sort=False).sum()

                for ind_t in df[ind].index.tolist():
                    t_id = df[ind]['tweet_id'][ind_t]
                    err = df[ind]['err'][ind_t]
                    abs_err = np.abs(err)
                    df[ind]['abs_err'][ind_t] = abs_err
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += (val - df[ind]['rel_gt_v'][ind_t])
                        sum_rnd_abs_perc += np.abs(val - df[ind]['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    norm_err = err / float(random_perc)
                    norm_abs_err = abs_err / float(random_abs_perc)
                    df[ind]['norm_err'][ind_t] = norm_err
                    df[ind]['norm_abs_err'][ind_t] = norm_abs_err

                    # df[ind] = df[ind].copy()

            w_pt_avg_l = []
            w_err_avg_l = []
            w_abs_err_avg_l = []
            w_norm_err_avg_l = []
            w_norm_abs_err_avg_l = []
            w_acc_avg_l = []

            w_pt_std_l = []
            w_err_std_l = []
            w_abs_err_std_l = []
            w_norm_err_std_l = []
            w_norm_abs_err_std_l = []
            w_acc_std_l = []

            w_pt_avg_dict = collections.defaultdict()
            w_err_avg_dict = collections.defaultdict()
            w_abs_err_avg_dict = collections.defaultdict()
            w_norm_err_avg_dict = collections.defaultdict()
            w_norm_abs_err_avg_dict = collections.defaultdict()
            w_acc_avg_dict = collections.defaultdict()

            w_pt_std_dict = collections.defaultdict()
            w_err_std_dict = collections.defaultdict()
            w_abs_err_std_dict = collections.defaultdict()
            w_norm_err_std_dict = collections.defaultdict()
            w_norm_abs_err_std_dict = collections.defaultdict()
            w_acc_std_dict = collections.defaultdict()

            all_w_pt_list = []
            all_w_err_list = []
            all_w_abs_err_list = []
            all_w_norm_err_list = []
            all_w_norm_abs_err_list = []
            all_w_acc_list = []

            all_w_cyn_list = []
            all_w_gull_list = []
            w_cyn_avg_l = []
            w_gull_avg_l = []
            w_cyn_std_l = []
            w_gull_std_l = []
            w_cyn_avg_dict = collections.defaultdict()
            w_gull_avg_dict = collections.defaultdict()
            w_cyn_std_dict = collections.defaultdict()
            w_gull_std_dict = collections.defaultdict()
            w_pt_var_dict = collections.defaultdict()
            for ind in ind_l:

                df_m = df[ind].copy()
                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]

                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    # w_abs_err_list = list(df_tmp['abs_err'])
                    w_abs_err_list = list(df_tmp['susc'])
                    w_norm_err_list = list(df_tmp['norm_err'])
                    w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    df_cyn = df_tmp[df_tmp['cyn'] > 0]
                    df_gull = df_tmp[df_tmp['gull'] > 0]

                    w_cyn_list = list(df_cyn['cyn'])
                    w_gull_list = list(df_gull['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])
                    w_acc_list = []
                    # w_ind_acc_list
                    acc_c = 0
                    nacc_c = 0

                    # w_acc_avg_l.append(w_ind_acc_list)

                    w_pt_std_l.append(np.std(w_pt_list))
                    w_err_std_l.append(np.std(w_err_list))
                    w_abs_err_std_l.append(np.std(w_abs_err_list))
                    w_norm_err_std_l.append(np.std(w_norm_err_list))
                    w_norm_abs_err_std_l.append(np.std(w_norm_abs_err_list))
                    w_cyn_std_l.append(np.std(w_cyn_list))
                    w_gull_std_l.append(np.std(w_gull_list))
                    # w_acc_std_l.append(np.std(w_ind_acc_list))


                    w_pt_avg_dict[t_id] = np.mean(w_pt_list)
                    w_err_avg_dict[t_id] = np.mean(w_err_list)
                    w_abs_err_avg_dict[t_id] = np.mean(w_abs_err_list)
                    w_norm_err_avg_dict[t_id] = np.mean(w_norm_err_list)
                    w_norm_abs_err_avg_dict[t_id] = np.mean(w_norm_abs_err_list)
                    w_cyn_avg_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_avg_dict[t_id] = np.mean(w_gull_list)
                    # w_acc_avg_dict[t_id] = w_ind_acc_list


                    w_pt_var_dict[t_id] = np.var(w_pt_list)
                    w_pt_var_dict[t_id] = np.std(w_pt_list)

                    # ind_
            ##################################################
            #


            # fig_f = True
            fig_f = False
            # fig_f_1 = True
            fig_f_1 = False
            fig_f_together = True



            if fig_f_together == True:

                out_dict = w_pt_var_dict
                ####ptl_cdf
                mplpl.rcParams['figure.figsize'] = 4.5, 2.5
                mplpl.rc('xtick', labelsize='large')
                mplpl.rc('ytick', labelsize='large')
                mplpl.rc('legend', fontsize='small')
                w_err_avg_dict
                # tweet_l_sort = sorted(w_norm_abs_err_avg_dict, key=w_norm_abs_err_avg_dict.get, reverse=False)
                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                # tweet_l_sort = [x for x in tweet_l_sort if x >= 0 or x < 0]
                acc_l = []
                for t_id in tweet_l_sort:
                    if out_dict[t_id] >= 0 or out_dict[t_id] < 0:
                        acc_l.append(out_dict[t_id])

                num_bins = len(acc_l)
                counts, bin_edges = np.histogram(acc_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5, label=data_name)

        legend_properties = {'weight': 'bold'}

        #

        mplpl.ylabel('CDF', fontsize=20, fontweight='bold')
        mplpl.xlabel('Disputability', fontsize=20, fontweight='bold')
        mplpl.legend(loc="lower right", prop=legend_properties, fontsize='small', ncol=1)
        # mplpl.title(data_name)
        # mplpl.legend(loc="upper left",fontsize = 'large')
        mplpl.xlim([0, 1])
        mplpl.ylim([0, 1])
        mplpl.grid()
        mplpl.subplots_adjust(bottom=0.24)
        mplpl.subplots_adjust(left=0.18)
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/NAPB_cdf_alldataset'
        pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/Disputability_cdf_alldataset_new'
        mplpl.savefig(pp + '.pdf', format='pdf')
        mplpl.savefig(pp + '.png', format='png')

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_DISP_TPB_scatter_fig":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []


        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []



        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_2']:  # ['snopes_ssi','snopes_incentive','snopes_2','snopes_ssi','politifact','mia',snopes_nonpol]:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'
            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            w_cyn_dict = collections.defaultdict()
            w_gull_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()
            tweet_avg_susc_group = collections.defaultdict()
            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []

            same_side = 0
            other_side = 0

            rep_b = 0
            dem_b = 0
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]

                    df_tmp = df_tmp[df_tmp['ra']!=4]
                    df_tmp = df_tmp[df_tmp['ra']!=3]
                    df_tmp = df_tmp[df_tmp['ra']!=5]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]

                    dem_err = np.mean(dem_df['err'])
                    rep_err = np.mean(rep_df['err'])

                    if dem_err * rep_err >= 0:
                        same_side += 1

                        if np.abs(dem_err) > np.abs(rep_err):
                            rep_b += 1
                        else:
                            dem_b += 1
                    else:
                        other_side += 1

                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    dem_susc_list = list(dem_df['susc'])
                    rep_susc_list = list(rep_df['susc'])
                    neut_susc_list = list(neut_df['susc'])

                    tweet_avg_susc_group[t_id] = np.abs(np.mean(dem_susc_list) - np.mean(rep_susc_list))

                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    # w_abs_err_list = list(df_tmp['abs_err'])
                    w_sus_list = list(df_tmp['susc'])
                    # w_norm_err_list = list(df_tmp['norm_err'])
                    # w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    # w_cyn_list = list(df_tmp['cyn'])
                    # w_gull_list = list(df_tmp['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])

                    df_cyn = df_tmp[df_tmp['cyn'] > 0]
                    df_gull = df_tmp[df_tmp['gull'] > 0]

                    w_cyn_list = list(df_cyn['cyn'])
                    w_gull_list = list(df_gull['gull'])

                    w_cyn_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_dict[t_id] = np.mean(w_gull_list)
                    w_apb_dict[t_id] = np.mean(w_sus_list)

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))

                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_l.append(tweet_skew[t_id])

                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)

                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
                    # tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    # tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)

            # news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']
            print('same side ' + str(same_side))
            print('other side : ' + str(other_side))

            print('dem better : ' + str(dem_b))
            print('rep better : ' + str(rep_b))
            # exit()
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_2':
                col_l = ['darkred', 'orange', 'gray', 'lime', 'green']
                # col = 'purple'
                col = 'k'
                news_cat_list_n = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_t_f = [['FALSE', 'MOSTLY FALSE'], ['MOSTLY TRUE', 'TRUE']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']
            if dataset == 'politifact':
                col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']
                news_cat_list_n = ['PANTS ON FIRE', 'FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'MOSTLY TRUE', 'TRUE']
                # col = 'c'
                col = 'k'
                news_cat_list_t_f = [['pants-fire', 'false', 'mostly-false'], ['mostly-true', 'true']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            if dataset == 'mia':
                col_l = ['red', 'green']
                news_cat_list_n = ['RUMORS', 'NON RUMORS']
                # col = 'brown'
                col = 'k'
                col_t_f = ['red', 'green']
                news_cat_list_t_f = [['rumors'], ['non-rumors']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            count = 0
            # Y = [0]*len(thr_list)




            tweet_abs_perc_rnd_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            # tweet_perc_rnd_sort = sorted(tweet_dev_avg_rnd, key=tweet_dev_avg_rnd.get, reverse=True)
            tweet_abs_perc_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            # tweet_perc_sort = sorted(tweet_dev_avg, key=tweet_dev_avg.get, reverse=True)
            tweet_disp_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            gt_l = []
            pt_l = []

            fig_cdf = True
            # fig_cdf = False
            if fig_cdf == True:

                #####################################################33

                out_dict = w_apb_dict
                # out_dict = w_gull_dict
                # out_dict = w_cyn_dict
                # out_dict = tweet_avg_group
                # out_dict = tweet_avg
                # out_dict = w_apb_dict
                # out_dict = tweet_avg_susc_group
                tweet_l_sort = sorted(tweet_var, key=tweet_var.get, reverse=False)
                # tweet_l_sort = sorted(tweet_avg, key=tweet_avg.get, reverse=False)
                pt_l = []
                t_var_l = []
                pt_l_dict = collections.defaultdict(list)
                for t_id in tweet_l_sort:
                    pt_l.append(out_dict[t_id])
                    t_var_l.append(tweet_var[t_id])
                count = 0

                # num_bins = len(pt_l)
                # counts, bin_edges = np.histogram(pt_l, bins=num_bins, normed=True)
                # cdf = np.cumsum(counts)
                # scale = 1.0 / cdf[-1]
                # ncdf = scale * cdf
                # mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5,linestyle='--', label='All news stories')
                # mplpl.rcParams['figure.figsize'] = 5.8, 3.7
                # mplpl.rcParams['figure.figsize'] = 5.6, 3.6
                mplpl.rcParams['figure.figsize'] = 4.5, 2.5
                mplpl.rc('xtick', labelsize='large')
                mplpl.rc('ytick', labelsize='large')
                mplpl.rc('legend', fontsize='medium')

                # for t_id in tweet_l_sort:
                #     if w_apb_dict[t_id]>0 or w_apb_dict[t_id]<=0:
                #         pt_l.append(w_apb_dict[t_id])



                # df_tt = pd.DataFrame(np.array(pt_l_dict[cat_m]), columns=[cat_m])
                # df_tt[cat_m].plot(kind='kde', lw=6, color=col_l[count-1], label=cat_m)

                # num_bins = len(pt_l_dict[cat_m])
                # counts, bin_edges = np.histogram(pt_l_dict[cat_m], bins=num_bins, normed=True)
                # cdf = np.cumsum(counts)
                # scale = 1.0 / cdf[-1]
                # ncdf = scale * cdf
                # mplpl.plot(bin_edges[1:], ncdf, c=col_l[count-1], lw=5, label=cat_m)


                mplpl.scatter(range(len(pt_l)), pt_l, c='purple')
                z = np.polyfit(range(len(pt_l)), pt_l, 1)
                p = np.poly1d(z)
                mplpl.plot(range(len(pt_l)), p(range(len(pt_l))), 'r-', linewidth=4.0)

                print(np.corrcoef(pt_l, t_var_l))
                print(scipy.stats.spearmanr(pt_l, t_var_l))
                mplpl.xlabel('News stories ranked by disputability', fontsize=13, fontweight='bold')
                # mplpl.xlabel('News stories ranked by PTL', fontsize=13, fontweight = 'bold')
                mplpl.ylabel('Total Perception Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('Ideological Perception Bias', fontsize=12, fontweight = 'bold')
                # mplpl.ylabel(r'$|MPB_{Dem} - MPB_{Rep}|$', fontsize=15, fontweight='bold')
                # mplpl.ylabel('|TPB of Dems - TPB of Reps|', fontsize=13, fontweight = 'bold')
                mplpl.grid()
                mplpl.title(data_name, fontsize='x-large')
                labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
                y = [0.0, 0.5, 1, 1.5, 2]
                # mplpl.yticks(y, labels)
                legend_properties = {'weight': 'bold'}

                # plt.legend(prop=legend_properties)
                # mplpl.legend(loc="upper left",prop=legend_properties,fontsize='small', ncol=1)#, fontweight = 'bold')
                mplpl.legend(loc="lower right", prop=legend_properties, fontsize='medium', ncol=1)  # , fontweight = 'bold')
                # mplpl.xlim([0, 150])
                mplpl.xlim([0, 52])
                mplpl.ylim([0, 2])
                # mplpl.ylim([0, 1])
                mplpl.subplots_adjust(bottom=0.24)
                mplpl.subplots_adjust(left=0.18)

                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_pt_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_cdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_cdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_disput_TPB_scatter'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_PTL_TPB_scatter'
                pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_disput_TPB_scatter_1'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_disput_ITPB_scatter'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_disput_IPB_scatter'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_FNB_pdf'

                mplpl.savefig(pp + '.pdf', format='pdf')
                mplpl.savefig(pp + '.png', format='png')
                mplpl.figure()

                exit()

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_time_analysis":

        # dataset = 'snopes'
        dataset = 'snopes_ssi'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        if dataset == 'mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                  + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

            exp1_list = sample_tweets_exp1

            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor = {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id) < 100060:
                    tweet_lable_dict[tweet_id] = 'rumor'
                else:
                    tweet_lable_dict[tweet_id] = 'non-rumor'

        if dataset == 'snopes' or dataset == 'snopes_ssi':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                # m_day = int(dt_splt[2])
                # m_month = int(dt_splt[1])
                # m_year = int(dt_splt[0])
                # m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                # tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        # exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0
        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        fig_f = True
        # fig_f = False
        if dataset == 'snopes':
            data_n = 'sp'
            ind_l = [1, 2, 3]
        elif dataset == 'snopes_ssi':
            data_n = 'sp_ssi'
            ind_l = [1, 2, 3]
        elif dataset == 'politifact':
            data_n = 'pf'
            ind_l = [1, 2, 3]
        elif dataset == 'mia':
            data_n = 'mia'
            ind_l = [1]
        t_time_all = []
        for ind in ind_l:
            if balance_f == 'balanced':
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
            else:
                # inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_time.csv'
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
            inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
            df[ind] = pd.read_csv(inp1, sep="\t")
            # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

            df_m = df[ind].copy()

            groupby_ftr = 'tweet_id'
            grouped = df_m.groupby(groupby_ftr, sort=False)
            grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

            # df_tmp = df_m[df_m['tweet_id'] == t_id]
            for t_id in grouped.groups.keys():
                df_tmp = df_m[df_m['tweet_id'] == t_id]
                ind_t = df_tmp.index.tolist()[0]
                weights = []
                weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                val_list = list(df_tmp['rel_v'])
                tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                tweet_avg[t_id] = np.mean(val_list)
                tweet_med[t_id] = np.median(val_list)
                tweet_var[t_id] = np.var(val_list)
                tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                tweet_avg_l.append(np.mean(val_list))
                tweet_med_l.append(np.median(val_list))
                tweet_var_l.append(np.var(val_list))
                tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])
                accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))

                # all_acc.append(accuracy)

                t_time_avg = np.mean(list(df_tmp['delta_time']))
                t_time_all.append(t_time_avg)
                all_acc += list(df_tmp['delta_time'])
        print(np.mean(t_time_all))
        print(np.mean(all_acc))

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_time_fig":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        if dataset == 'mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                  + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

            exp1_list = sample_tweets_exp1
            tweet_id = 100010
            publisher_name = 110
            tweet_popularity = {}
            tweet_text_dic = {}
            for input_file in [input_rumor, input_non_rumor]:
                for line in input_file:
                    line.replace('\n', '')
                    line_splt = line.split('\t')
                    tweet_txt = line_splt[1]
                    tweet_link = line_splt[1]
                    tweet_id += 1
                    publisher_name += 1
                    tweet_popularity[tweet_id] = int(line_splt[2])
                    tweet_text_dic[tweet_id] = tweet_txt

            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor = {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id) < 100060:
                    tweet_lable_dict[tweet_id] = 'rumor'
                else:
                    tweet_lable_dict[tweet_id] = 'non-rumor'

        if dataset == 'snopes':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable

        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        ##########################prepare balanced data (same number of rep, dem, neut #############



        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_incentive','snopes_nonpol', 'snopes', 'politifact', 'mia']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset=='snopes_incentive':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes_incentive'
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpolitical'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            w_cyn_dict = collections.defaultdict()
            w_gull_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            cat_time_dict = collections.defaultdict(dict)
            time_dict = collections.defaultdict()
            cat_time_list = collections.defaultdict(list)
            cat_time_dict_var = collections.defaultdict(dict)
            time_dict_var = collections.defaultdict()
            cat_time_list_var = collections.defaultdict(list)

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    # tweet_avg_group[t_id] = np.mean(dem_val_list) - np.mean(rep_val_list)
                    # tweet_med_group[t_id] = np.median(dem_val_list) - np.median(rep_val_list)
                    # tweet_var_group[t_id] = np.var(dem_val_list) - np.var(rep_val_list)
                    # tweet_kldiv_group[t_id] = np.mean(dem_val_list)+np.mean(rep_val_list) + np.mean(neut_val_list)
                    # tweet_kldiv_group[t_id] = np.var(dem_val_list) * np.var(rep_val_list) / np.var(neut_val_list)

                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    cat_time_dict[tweet_lable_dict[t_id]][t_id] = np.mean(df_tmp['delta_time'])
                    cat_time_list[tweet_lable_dict[t_id]].append(np.mean(df_tmp['delta_time']))
                    time_dict[t_id] = np.mean(df_tmp['delta_time'])

                    cat_time_dict_var[tweet_lable_dict[t_id]][t_id] = np.std(df_tmp['delta_time'])
                    cat_time_list_var[tweet_lable_dict[t_id]].append(np.std(df_tmp['delta_time']))
                    time_dict_var[t_id] = np.std(df_tmp['delta_time'])

                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    # w_abs_err_list = list(df_tmp['abs_err'])
                    w_sus_list = list(df_tmp['susc'])
                    # w_norm_err_list = list(df_tmp['norm_err'])
                    # w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    # w_cyn_list = list(df_tmp['cyn'])
                    # w_gull_list = list(df_tmp['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])

                    df_cyn = df_tmp[df_tmp['cyn'] > 0]
                    df_gull = df_tmp[df_tmp['gull'] > 0]

                    w_cyn_list = list(df_cyn['cyn'])
                    w_gull_list = list(df_gull['gull'])

                    w_cyn_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_dict[t_id] = np.mean(w_gull_list)
                    w_apb_dict[t_id] = np.mean(w_sus_list)

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))

                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_l.append(tweet_skew[t_id])

                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)

                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
                    # tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    # tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)

            # news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

            for cat in cat_time_dict:
                print(
                cat + ' : mean = ' + str(np.mean(cat_time_list[cat])) + ' ,median = ' + str(np.median(cat_time_list[cat]))
                + ' ,std= ' + str(np.std(cat_time_list[cat])))

            tweet_abs_perc_rnd_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            # tweet_perc_rnd_sort = sorted(tweet_dev_avg_rnd, key=tweet_dev_avg_rnd.get, reverse=True)
            tweet_abs_perc_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            # tweet_perc_sort = sorted(tweet_dev_avg, key=tweet_dev_avg.get, reverse=True)
            tweet_disp_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            gt_l = []
            pt_l = []

            out_dict = w_apb_dict
            # out_dict = w_gull_dict
            # out_dict = w_cyn_dict
            # out_dict = tweet_var
            # out_dict = time_dict

            tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
            t_l = []
            t_var_l = []
            gt = []
            pt_l_dict = collections.defaultdict(list)
            for t_id in tweet_l_sort:
                t_var_l.append(time_dict_var[t_id])
                t_l.append(time_dict[t_id])
                gt.append(out_dict[t_id])
                count = 0

            print(np.corrcoef(t_var_l, gt)[0][1])
            print(np.corrcoef(t_l, gt)[0][1])
            # print(sklearn.metrics.normalized_mutual_info_score(t_l,gt))
            # print(sklearn.metrics.normalized_mutual_info_score(t_var_l,gt))
            # print(sklearn.metrics.normalized_mutual_info_score(t_l,t_var_l))
            # print(sklearn.feature_selection.mutual_info_classif(t_l,t_var_l, discrete_features='auto', n_neighbors=3, copy=True, random_state=None))
            # exit()
            if dataset == 'snopes' or dataset == 'snopes_nonpol':
                col_l = ['darkred', 'orange', 'gray', 'lime', 'green']
                # col = 'purple'
                col = 'k'
                news_cat_list_n = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_t_f = [['FALSE', 'MOSTLY FALSE'], ['MOSTLY TRUE', 'TRUE']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']
            if dataset == 'politifact':
                col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']
                news_cat_list_n = ['PANTS ON FIRE', 'FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'MOSTLY TRUE', 'TRUE']
                # col = 'c'
                col = 'k'
                news_cat_list_t_f = [['pants-fire', 'false', 'mostly-false'], ['mostly-true', 'true']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            if dataset == 'mia':
                col_l = ['red', 'green']
                news_cat_list_n = ['RUMORS', 'NON RUMORS']
                # col = 'brown'
                col = 'k'
                col_t_f = ['red', 'green']
                news_cat_list_t_f = [['rumors'], ['non-rumors']]
                cat_list_t_f = ['FALSE', 'TRUE']
                col_t_f = ['red', 'green']

            count = 0
            # Y = [0]*len(thr_list)




            tweet_abs_perc_rnd_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            # tweet_perc_rnd_sort = sorted(tweet_dev_avg_rnd, key=tweet_dev_avg_rnd.get, reverse=True)
            tweet_abs_perc_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            # tweet_perc_sort = sorted(tweet_dev_avg, key=tweet_dev_avg.get, reverse=True)
            tweet_disp_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            gt_l = []
            pt_l = []

            fig_cdf = True
            # fig_cdf = False
            if fig_cdf == True:

                #####################################################33

                # out_dict = w_apb_dict
                # out_dict = w_gull_dict
                # out_dict = w_cyn_dict
                # out_dict = tweet_var
                out_dict = time_dict

                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                pt_l = []
                pt_l_dict = collections.defaultdict(list)
                for t_id in tweet_l_sort:
                    pt_l.append(out_dict[t_id])
                count = 0

                # num_bins = len(pt_l)
                # counts, bin_edges = np.histogram(pt_l, bins=num_bins, normed=True)
                # cdf = np.cumsum(counts)
                # scale = 1.0 / cdf[-1]
                # ncdf = scale * cdf
                # mplpl.plot(bin_edges[1:], ncdf, c=col, lw=5,linestyle='--', label='All news stories')
                # mplpl.rcParams['figure.figsize'] = 5.8, 3.7
                mplpl.rcParams['figure.figsize'] = 4.5, 2.5
                mplpl.rc('xtick', labelsize='large')
                mplpl.rc('ytick', labelsize='large')
                mplpl.rc('legend', fontsize='small')
                for cat_m in news_cat_list:
                    count += 1
                    pt_l_dict[cat_m] = []
                    for t_id in tweet_l_sort:
                        if tweet_lable_dict[t_id] == cat_m:
                            # if out_dict[t_id]>0 or out_dict[t_id]<=0:
                            pt_l_dict[cat_m].append(out_dict[t_id])

                    # df_tt = pd.DataFrame(np.array(pt_l_dict[cat_m]), columns=[cat_m])
                    # df_tt[cat_m].plot(kind='kde', lw=6, color=col_l[count-1], label=cat_m)

                    num_bins = len(pt_l_dict[cat_m])
                    counts, bin_edges = np.histogram(pt_l_dict[cat_m], bins=num_bins, normed=True)
                    cdf = np.cumsum(counts)
                    scale = 1.0 / cdf[-1]
                    ncdf = scale * cdf
                    mplpl.plot(bin_edges[1:], ncdf, c=col_l[count - 1], lw=5, label=cat_m)

                # mplpl.ylabel('PDF of Perceived Truth Level', fontsize=20, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n Absolute Perception Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n False Positive Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('CDF of \n False Negative Bias', fontsize=13, fontweight = 'bold')

                # mplpl.ylabel('PDF of \n Absolute Perception Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('PDF of \n False Positive Bias', fontsize=13, fontweight = 'bold')
                # mplpl.ylabel('PDF of \n False Negative Bias', fontsize=13, fontweight='bold')

                # mplpl.xlabel('Total Perception Bias', fontsize=24, fontweight = 'bold')
                # mplpl.xlabel('False Positive Bias', fontsize=24, fontweight = 'bold')
                mplpl.xlabel('Time', fontsize=20, fontweight='bold')
                mplpl.ylabel('CDF', fontsize=20, fontweight='bold')
                mplpl.grid()
                mplpl.title(data_name, fontsize='x-large')
                labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
                y = [0.0, 0.5, 1, 1.5, 2]
                # mplpl.yticks(y, labels)
                legend_properties = {'weight': 'bold'}

                # plt.legend(prop=legend_properties)
                # mplpl.legend(loc="upper left",prop=legend_properties,fontsize='small', ncol=1)#, fontweight = 'bold')
                mplpl.legend(loc="lower right", prop=legend_properties, fontsize='small', ncol=1)  # , fontweight = 'bold')
                # mplpl.xlim([0, 1])
                # mplpl.ylim([0, 2])
                # mplpl.ylim([0, 1])
                mplpl.subplots_adjust(bottom=0.24)
                mplpl.subplots_adjust(left=0.18)

                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_pt_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_cdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_cdf'
                pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_time_cdf'

                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_APB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/'+dataset+'_FPB_pdf'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + dataset + '_FNB_pdf'

                mplpl.savefig(pp + '.pdf', format='pdf')
                mplpl.savefig(pp + '.png', format='png')
                mplpl.figure()

                exit()

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_true-false(gt-pt)_news_ktop_nptl_scatter":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # dataset = 'snopes'
        # dataset = 'mia'
        dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        ##########################prepare balanced data (same number of rep, dem, neut #############


        # balance_f = 'balanced'


        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_ssi','snopes_nonpol', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            news_cat_list_tf = [4, 2, 3, 1]
            t_f_dict_len = collections.defaultdict(int)
            t_f_dict = {}
            if dataset == 'snopes' or dataset == 'snopes_nonpol'or dataset == 'snopes_ssi':
                news_cat_list_t_f = [['FALSE', 'MOSTLY FALSE'], ['MOSTLY TRUE', 'TRUE']]
            if dataset == 'politifact':
                news_cat_list_t_f = [['pants-fire', 'false', 'mostly-false'], ['mostly-true', 'true']]

            if dataset == 'mia':
                news_cat_list_t_f = [['rumor'], ['non-rumor']]

            w_fnb_dict = collections.defaultdict()
            w_fpb_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp' + str(ind) + '_final_weighted.csv'
                    # inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_'+data_n+'_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()


                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []


                    dem_df = df_tmp[df_tmp['leaning']==1]
                    rep_df = df_tmp[df_tmp['leaning']==-1]
                    neut_df = df_tmp[df_tmp['leaning']==0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    # tweet_avg_group[t_id] = np.mean(dem_val_list) - np.mean(rep_val_list)
                    # tweet_med_group[t_id] = np.median(dem_val_list) - np.median(rep_val_list)
                    # tweet_var_group[t_id] = np.var(dem_val_list) - np.var(rep_val_list)
                    # tweet_kldiv_group[t_id] = np.mean(dem_val_list)+np.mean(rep_val_list) + np.mean(neut_val_list)
                    # tweet_kldiv_group[t_id] = np.var(dem_val_list) * np.var(rep_val_list) / np.var(neut_val_list)

                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list,rep_val_list)[1], 4)


                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])

                    for vot in vot_list_tmp:
                        if vot < 0 :
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))



                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_l.append(tweet_skew[t_id])



                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)


                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2/float(3), -1/float(3), 0, 1/float(3), 2/float(3),1]:
                        sum_rnd_perc+= val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)



                    val_list = list(df_tmp['susc'])
                    w_apb_dict[t_id] = np.mean(val_list)


            ##################################################
            gt_fpb = []
            pt_fpb = []
            gt_fnb = []
            pt_fnb = []
            gt_pt = []
            pt_pt = []
            len_cat_dict = {}



            # Y = [0]*len(thr_list)
            mplpl.rcParams['figure.figsize'] = 4.8, 4
            mplpl.rc('xtick', labelsize='large')
            mplpl.rc('ytick', labelsize='large')
            mplpl.rc('legend', fontsize='small')


            tweet_vote_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)

            gt_l_t = []
            pt_l_t = []
            apb_l_t = []
            for i in [int(0.1*len(tweet_vote_sort))]:
                tweet_id_list = tweet_vote_sort[:int(0.1*len(tweet_vote_sort))]
                for t_id in tweet_id_list:
                    gt_l_t.append(tweet_gt_var[t_id])
                    pt_l_t.append(tweet_avg[t_id])
                    apb_l_t.append(w_apb_dict[t_id])


            tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)

            gt_l_t_vot = []
            pt_l_t_vot = []
            apb_l_t_vot = []
            for i in [int(0.1 * len(tweet_vote_sort))]:
                tweet_id_list = tweet_vote_sort[:int(0.1 * len(tweet_vote_sort))]
                for t_id in tweet_id_list:
                    gt_l_t_vot.append(tweet_gt_var[t_id])
                    pt_l_t_vot.append(tweet_avg[t_id])
                    apb_l_t_vot.append(w_apb_dict[t_id])

            #
            # for cat_m in news_cat_list_tf:
            #     count+=1
            #     outp[cat_m] = []
            #     gt_l_t = []
            #     pt_l_t = []
            #     for i in [int(1*len_t)]:
            #         tweet_id_list = categ_dict_n[i][cat_m]
            #         for t_id in tweet_id_list:
            #             gt_l_t.append(tweet_gt_var[t_id])
            #             pt_l_t.append(tweet_avg[t_id])
            #
            #     m_label = str(np.round(len(tweet_id_list) / float(total_data[int(1 * len_t)]),2))

            # mplpl.scatter(gt_l_t, pt_l_t,c=col_l[count-1],marker=marker_l[count-1], s=60, label = m_label)
            # mplpl.scatter(pt_l_t,gt_l_t, c=col_l[count - 1], marker=marker_l[count - 1], s=60, label=m_label)


            # mplpl.scatter(gt_fnb,pt_fnb, c='r', marker='s', s=200)#, label=m_label)
            # mplpl.scatter(gt_fpb,pt_fpb, c='r', marker='d', s=200)#, label=m_label)
            # mplpl.scatter(gt_pt,pt_pt, c='orange', marker='^', s=200)#, label=m_label)
            # mplpl.scatter(gt_l_t,pt_l_t, c='g', marker='o', s=60)#, label=m_label)
            mplpl.scatter(gt_l_t,apb_l_t, c='g', marker='o', s=60, label='Picked by Disputability')
            mplpl.scatter(gt_l_t_vot,apb_l_t_vot, c='r', marker='+', s=120, label='Picked by Negative PTL')


            # mplpl.plot([-1.1, 1.1], [-1.1,1.1],c='k',linewidth=4)
            # mplpl.plot([0,0],[-1.1, 1.1],c='k',linewidth=4)

            mplpl.xlim([-1.04, 1.04])
            mplpl.ylim([0, 1.5])

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16,
                    }

            font_1 = {'family': 'serif',
                    'color': 'darkblue',
                    'weight': 'normal',
                    'size': 16,
                    }


            font_t = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'bold',
                    'size': 12,
                    }

            font_t_1 = {'family': 'serif',
                    'color': 'darkblue',
                    'weight': 'normal',
                    'size': 12,
                    }
            # mplpl.text(-0.7, 0.85, ' Mostly deserve to pick', fontdict=font_t)
            # # mplpl.text(-0.3, -0.9, '# news has negative perception truth value', fontdict=font_t_1)
            # mplpl.annotate('', xy=(-1, 0.38), xytext=(0, 0.82),
            #             arrowprops=dict(facecolor='r', shrink=0.1))
            #
            # mplpl.annotate('', xy=(1, -0.25), xytext=(0, 0.82),
            #             arrowprops=dict(facecolor='r', shrink=0.1))

            mplpl.subplots_adjust(bottom=0.32)

            mplpl.subplots_adjust(left=0.2)
            mplpl.title(data_name)
            labels = ['-1\nFalse', '-.05\nMostly\nFalse', '0\nMixture', '0.5\nMostly\n True', '1\nTrue']
            x = [ -1, -.5, 0, 0.5, 1]
            mplpl.xticks(x, labels)

            # mplpl.ylabel('Composition of news stories \n in different quadrants', fontsize=16,fontweight = 'bold')
            mplpl.xlabel('Ground Truth Level', fontsize=16,fontweight = 'bold')
            # mplpl.ylabel('Perceived Truth Level', fontsize=16,fontweight = 'bold')
            mplpl.ylabel('Total Perception Bias', fontsize=16,fontweight = 'bold')
            # mplpl.xlabel('Top k news stories ranked by NAPB', fontsize=18)

            mplpl.legend(loc="upper center", ncol=1, fontsize='medium')



            mplpl.grid()
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_10_top_disp_gt_pt_compos_true-false_scatter'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_20_top_ideolg_disp_gt_pt_compos_true-false_scatter'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_20_top_ideolg_disp_gt_apb_compos_true-false_scatter'
            pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_10_top_disp_gt_apb_compos_true-false_scatter'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_10_top_disp_gt_apb_compos_true-false_scatter_weighted'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_napb_compos_true-false_scatter'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_vote_compos_true-false_scatter'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_disp_compos_true-false_scatter'
            mplpl.savefig(pp + '.pdf', format='pdf')
            mplpl.savefig(pp+ '.png', format='png')

            mplpl.figure()

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_composition_labeld_news_ktop_nptl_together":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # dataset = 'snopes'
        # dataset = 'mia'
        dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []



        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []



        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_ssi','snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    val_list = list(df_tmp['rel_v'])
                    # tweet_avg_group[t_id] = np.mean(dem_val_list) - np.mean(rep_val_list)
                    # tweet_med_group[t_id] = np.median(dem_val_list) - np.median(rep_val_list)
                    # tweet_var_group[t_id] = np.var(dem_val_list) - np.var(rep_val_list)
                    # tweet_kldiv_group[t_id] = np.mean(dem_val_list)+np.mean(rep_val_list) + np.mean(neut_val_list)
                    # tweet_kldiv_group[t_id] = np.var(dem_val_list) * np.var(rep_val_list) / np.var(neut_val_list)


                    # n_dem, bins, patches = Plab.hist(dem_val_list,
                    #                                      bins=frange(0, 1, 0.05), normed=1)
                    # n_rep, bins, patches = Plab.hist(rep_val_list,
                    #                                      bins=frange(0, 1, 0.05), normed=1)
                    # n_neut, bins, patches = Plab.hist(neut_val_list,
                    #                                   bins=frange(0, 1, 0.05), normed=1)
                    #
                    # dem_rep_chi = chi_sqr(n_dem, n_rep)
                    # dem_neut_chi = chi_sqr(n_dem, n_neut)
                    # neut_rep_chi = chi_sqr(n_neut, n_rep)



                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # if ~(dem_rep_chi > 0 or dem_rep_chi <0 or dem_rep_chi ==0):
                    #     tweet_chi_group[t_id] = 0
                    # else:
                    #     tweet_chi_group[t_id] = dem_rep_chi
                    #
                    # if ~(dem_neut_chi > 0 or dem_neut_chi <0 or dem_neut_chi ==0):
                    #     tweet_chi_group_1[t_id] = 0
                    # else:
                    #     tweet_chi_group_1[t_id] = dem_neut_chi
                    #
                    # if ~(dem_neut_chi > 0 or dem_neut_chi <0 or dem_neut_chi ==0):
                    #     tweet_chi_group_2[t_id] = 0
                    # else:
                    #     tweet_chi_group_2[t_id] = dem_rep_chi


                    # tweet_chi_group[t_id] = np.var([tweet_chi_group[t_id], tweet_chi_group_1[t_id], tweet_chi_group_2[t_id]])

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))

                    # accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
                    # all_acc.append(accuracy)


                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])



                    # val_list = list(df_tmp['susc'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)

                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l.append(np.var(abs_var_err))

                    # tweet_popularity_dict[t_id] = tweet_popularity[t_id]
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2 / float(3), -1 / float(3), 0, 1 / float(3), 2 / float(3), 1]:
                        sum_rnd_perc += val - df_tmp['rel_gt_v'][ind_t]
                        sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)

                    tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
                    # tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
                    # tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)

            ##################################################
            # news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE','TRUE']
            # news_cat_list_f = [ 'false','mostly_false', 'mixture', 'mostly_true', 'true']
            len_cat_dict = {}
            if dataset == 'snopes' or dataset == 'snopes_ssi' or dataset == 'politifact':
                for cat in news_cat_list:
                    len_cat_dict[cat] = 30
            elif dataset == 'snopes_nonpol':
                for cat in news_cat_list:
                    len_cat_dict[cat] = 20
            elif dataset == 'mia':
                for cat in news_cat_list:
                    if cat == 'rumor':
                        len_cat_dict[cat] = 30
                    else:
                        len_cat_dict[cat] = 30
            tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)

            # tweet_vote_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            tweet_var_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            thr = 20
            thr_list = []
            len_t = len(tweet_vote_sort)
            categ_dict = collections.defaultdict(int)
            k_list = [int(0.1 * len_t), int(0.2 * len_t), int(0.3 * len_t), int(1 * len_t)]
            count = 0
            for k in k_list:
                # k = (i+1)*thr
                # k = k_list[count]
                # count+=1
                thr_list.append(k)
                perc_rnd_l = []
                abs_perc_rnd_l = []
                disputability_l = []
                above_avg = 0
                less_avg = 0
                above_avg_rnd = 0
                less_avg_rnd = 0
                above_avg = 0
                less_avg = 0
                categ_dict[k] = collections.defaultdict(float)
                for j in range(k):
                    for cat_n in news_cat_list:
                        if tweet_lable_dict[tweet_vote_sort[j]] == cat_n:
                            categ_dict[k][cat_n] += (1 / float(len_cat_dict[cat_n]))

            # if dataset=='mia':
            for j in categ_dict:
                sum = np.sum(categ_dict[j].values())
                for cat_n in categ_dict[j]:
                    categ_dict[j][cat_n] = categ_dict[j][cat_n] / sum

            thr = 20
            thr_list = []
            len_var = len(tweet_var_sort)
            categ_dict_var = collections.defaultdict(int)
            k_list = [int(0.1 * len_var), int(0.2 * len_var), int(0.3 * len_t), int(1 * len_var)]
            count = 0
            for k in k_list:
                # k = (i+1)*thr
                # k = k_list[count]
                # count+=1
                thr_list.append(k)
                perc_rnd_l = []
                abs_perc_rnd_l = []
                disputability_l = []
                above_avg = 0
                less_avg = 0
                above_avg_rnd = 0
                less_avg_rnd = 0
                above_avg = 0
                less_avg = 0
                categ_dict_var[k] = collections.defaultdict(float)
                for j in range(k):
                    for cat_n in news_cat_list:
                        if tweet_lable_dict[tweet_var_sort[j]] == cat_n:
                            categ_dict_var[k][cat_n] += (1 / float(len_cat_dict[cat_n]))

            # if dataset=='mia':
            for j in categ_dict_var:
                sum = np.sum(categ_dict_var[j].values())
                for cat_n in categ_dict_var[j]:
                    categ_dict_var[j][cat_n] = categ_dict_var[j][cat_n] / sum

            width = 0.03
            pr = -10
            title_l = news_cat_list
            outp = {}
            outp_var = {}
            # news_cat_list = ['pants-fire', 'false', 'mostly_false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            if dataset == 'snopes' or dataset == 'snopes_nonpol'or dataset == 'snopes_ssi':
                # col_l = ['b', 'g', 'c', 'y', 'r']
                col_l = ['red', 'orange', 'gray', 'lime', 'green']

                news_cat_list_n = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
            if dataset == 'politifact':
                # col_l = ['grey','b', 'g', 'c', 'y', 'r']
                col_l = ['darkred', 'red', 'orange', 'gray', 'lime', 'green']

                news_cat_list_n = ['PANTS ON FIRE', 'FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'MOSTLY TRUE', 'TRUE']

            if dataset == 'mia':
                # col_l = ['b', 'r']
                col_l = ['red', 'green']
                news_cat_list_n = ['RUMORS', 'NON RUMORS']
            count = 0
            Y = [0] * len(thr_list)
            Y1 = [0] * len(thr_list)
            mplpl.rcParams['figure.figsize'] = 6.8, 5
            mplpl.rc('xtick', labelsize='large')
            mplpl.rc('ytick', labelsize='large')
            mplpl.rc('legend', fontsize='small')

            for cat_m in news_cat_list:
                count += 1
                outp[cat_m] = []
                outp_var[cat_m] = []
                for i in thr_list:
                    outp[cat_m].append(categ_dict[i][cat_m])
                    outp_var[cat_m].append(categ_dict_var[i][cat_m])
                # mplpl.bar([xx/float(len(tweet_vote_sort)) for xx in thr_list], outp[cat_m], width, bottom= np.array(Y), color=col_l[count-1], label=news_cat_list_n[count-1])
                mplpl.bar([0.1, 0.2, 0.3, 0.4], outp[cat_m], width, bottom=np.array(Y), color=col_l[count - 1],
                          label=news_cat_list_n[count - 1])
                Y = np.array(Y) + np.array(outp[cat_m])

                mplpl.bar([0.1 + 0.03, 0.2 + 0.03, 0.3 + 0.03, 0.4 + 0.03], outp_var[cat_m], width, bottom=np.array(Y1),
                          color=col_l[count - 1])
                Y1 = np.array(Y1) + np.array(outp_var[cat_m])

            mplpl.xlim([0.08, 0.5])
            mplpl.ylim([0, 1.38])
            mplpl.ylabel('Composition of labeled news stories', fontsize=14, fontweight='bold')
            # mplpl.xlabel('Top k news stories reported by negative PTL', fontsize=13.8,fontweight = 'bold')
            mplpl.xlabel('K fraction stories ranked by \n negative PTL(1st bar) and Disputability(2nd bar)', fontsize=14,
                         fontweight='bold')
            # mplpl.xlabel('Top k news stories ranked by NAPB', fontsize=18)

            mplpl.legend(loc="upper center", ncol=3, fontsize='small')

            mplpl.subplots_adjust(bottom=0.2)

            mplpl.subplots_adjust(left=0.18)
            mplpl.grid()
            mplpl.title(data_name, fontsize='x-large')
            labels = ['0.1', '0.2', '0.3', '1.0']
            x = [0.12, 0.22, 0.33, 0.44]
            mplpl.xticks(x, labels)
            # pp = remotedir + '/fig/fig_exp1/news_based/initial/' + data_n + '_vote_composition_gt'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_vote_composition_gt'
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_napb_composition_gt'
            pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + data_n + '_vote_disp_composition_gt'
            mplpl.savefig(pp + '.pdf', format='pdf')
            mplpl.savefig(pp + '.png', format='png')

            mplpl.figure()

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_comparing_ssi_amt":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []


        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_ssi']:#, 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    # remotedir='snopes/'
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l = []
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_new.csv'
                    # inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()



                df_modify_1 = df_m[df_m['tweet_id']==2001]
                df_modify_2 = df_m[df_m['tweet_id']==2002]
                worker_id_1 = df_modify_1[df_modify_1['ra']>4]['worker_id']
                worker_id_2 = df_modify_2[df_modify_2['ra']<4]['worker_id']
                df_modify = df_m[df_m['worker_id'].isin(set(worker_id_1).intersection(set(worker_id_2)))]
                # df_modify = df_m[df_m['tweet_id'].isin([2001, 2002])]
                #
                # df_m = df_modify.copy()
                df_m = df_m[df_m['tweet_id']!=2001]
                df_m = df_m[df_m['tweet_id']!=2002]
                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])
                    # df_tmp = neut_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l += list(df_tmp['ra'])


                    tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)


                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    # tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l.append(np.mean(val_list))
                    tweet_med_l.append(np.median(val_list))
                    tweet_var_l.append(np.var(val_list))
                    # tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg[t_id] = np.mean(vot_list)
                    tweet_vote_med[t_id] = np.median(vot_list)
                    tweet_vote_var[t_id] = np.var(vot_list)

                    tweet_vote_avg_l.append(np.mean(vot_list))
                    tweet_vote_med_l.append(np.median(vot_list))
                    tweet_vote_var_l.append(np.var(vot_list))




                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg[t_id] = np.mean(val_list)
                    tweet_dev_med[t_id] = np.median(val_list)
                    tweet_dev_var[t_id] = np.var(val_list)

                    tweet_dev_avg_l.append(np.mean(val_list))
                    tweet_dev_med_l.append(np.median(val_list))
                    tweet_dev_var_l.append(np.var(val_list))

                    tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var[t_id] = np.var(abs_var_err)

                    # tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
                    # tweet_abs_dev_med_l.append(np.median(abs_var_err))
                    # tweet_abs_dev_var_l.append(np.var(abs_var_err))

        for dataset in ['snopes_2']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            if dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_2'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_2 = {}
            tweet_dev_med_2 = {}
            tweet_dev_var_2 = {}
            tweet_avg_2 = {}
            tweet_med_2 = {}
            tweet_var_2 = {}
            tweet_gt_var_2 = {}

            tweet_dev_avg_l_2 = []
            tweet_dev_med_l_2 = []
            tweet_dev_var_l_2 = []
            tweet_avg_l_2 = []
            tweet_med_l_2 = []
            tweet_var_l_2 = []
            tweet_gt_var_l_2 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_2 = {}
            tweet_abs_dev_med_2 = {}
            tweet_abs_dev_var_2 = {}

            tweet_abs_dev_avg_l_2 = []
            tweet_abs_dev_med_l_2 = []
            tweet_abs_dev_var_l_2 = []

            tweet_abs_dev_avg_rnd_2 = {}
            tweet_dev_avg_rnd_2 = {}

            tweet_skew_2 = {}
            tweet_skew_l_2 = []

            tweet_vote_avg_med_var_2 = collections.defaultdict(list)
            tweet_vote_avg_2 = collections.defaultdict()
            tweet_vote_med_2 = collections.defaultdict()
            tweet_vote_var_2 = collections.defaultdict()

            tweet_avg_group_2 = collections.defaultdict()
            tweet_med_group_2 = collections.defaultdict()
            tweet_var_group_2 = collections.defaultdict()
            tweet_var_diff_group_2 = collections.defaultdict()

            tweet_kldiv_group_2 = collections.defaultdict()

            tweet_vote_avg_l_2 = []
            tweet_vote_med_l_2 = []
            tweet_vote_var_l_2 = []
            tweet_chi_group_2 = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            ind_l=[1]
            ans_l_2 = []
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()


                demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment',
                                         'income','political_view', 'race', 'marital_status']
                for demographic_feat in demographic_feat_list:
                    print('--------------------' + demographic_feat + '--------------------')
                    for lean_f in set(df_m[demographic_feat]):
                        df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                        print(lean_f + ' : '+str(len(set(df_lean_f['worker_id']))))





                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l_2 += list(df_tmp['ra'])
                    tweet_avg_group_2[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_2[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_2[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_2[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_2[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_2[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_2[t_id] = np.mean(val_list)
                    tweet_med_2[t_id] = np.median(val_list)
                    tweet_var_2[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_2.append(np.mean(val_list))
                    tweet_med_l_2.append(np.median(val_list))
                    tweet_var_l_2.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_2[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg_2[t_id] = np.mean(vot_list)
                    tweet_vote_med_2[t_id] = np.median(vot_list)
                    tweet_vote_var_2[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_2.append(np.mean(vot_list))
                    tweet_vote_med_l_2.append(np.median(vot_list))
                    tweet_vote_var_l_2.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg_2[t_id] = np.mean(val_list)
                    tweet_dev_med_2[t_id] = np.median(val_list)
                    tweet_dev_var_2[t_id] = np.var(val_list)

                    tweet_dev_avg_l_2.append(np.mean(val_list))
                    tweet_dev_med_l_2.append(np.median(val_list))
                    tweet_dev_var_l_2.append(np.var(val_list))

                    tweet_abs_dev_avg_2[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_2[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_2[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_2.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_2.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_2.append(np.var(abs_var_err))

        for dataset in ['snopes_incentive']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_3 = {}
            tweet_dev_med_3 = {}
            tweet_dev_var_3 = {}
            tweet_avg_3 = {}
            tweet_med_3 = {}
            tweet_var_3 = {}
            tweet_gt_var_3 = {}

            tweet_dev_avg_l_3 = []
            tweet_dev_med_l_3 = []
            tweet_dev_var_l_3 = []
            tweet_avg_l_3 = []
            tweet_med_l_3 = []
            tweet_var_l_3 = []
            tweet_gt_var_l_3 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_3 = {}
            tweet_abs_dev_med_3 = {}
            tweet_abs_dev_var_3 = {}

            tweet_abs_dev_avg_l_3 = []
            tweet_abs_dev_med_l_3 = []
            tweet_abs_dev_var_l_3 = []

            tweet_abs_dev_avg_rnd_3= {}
            tweet_dev_avg_rnd_3 = {}

            tweet_skew_3 = {}
            tweet_skew_l_3 = []

            tweet_vote_avg_med_var_3 = collections.defaultdict(list)
            tweet_vote_avg_3 = collections.defaultdict()
            tweet_vote_med_3 = collections.defaultdict()
            tweet_vote_var_3 = collections.defaultdict()

            tweet_avg_group_3 = collections.defaultdict()
            tweet_med_group_3 = collections.defaultdict()
            tweet_var_group_3 = collections.defaultdict()
            tweet_var_diff_group_3 = collections.defaultdict()

            tweet_kldiv_group_3 = collections.defaultdict()

            tweet_vote_avg_l_3 = []
            tweet_vote_med_l_3 = []
            tweet_vote_var_l_3 = []
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_3 = []
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
                #                          'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l_3 += list(df_tmp['ra'])
                    tweet_avg_group_3[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_3[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_3[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_3[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_3[t_id] = np.round(
                        scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_3[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_3[t_id] = np.mean(val_list)
                    tweet_med_3[t_id] = np.median(val_list)
                    tweet_var_3[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_3.append(np.mean(val_list))
                    tweet_med_l_3.append(np.median(val_list))
                    tweet_var_l_3.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_3[t_id] = [np.mean(vot_list), np.median(vot_list),
                                                      np.var(vot_list)]
                    tweet_vote_avg_3[t_id] = np.mean(vot_list)
                    tweet_vote_med_3[t_id] = np.median(vot_list)
                    tweet_vote_var_3[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_3.append(np.mean(vot_list))
                    tweet_vote_med_l_3.append(np.median(vot_list))
                    tweet_vote_var_l_3.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list),
                                                     np.var(val_list)]
                    tweet_dev_avg_3[t_id] = np.mean(val_list)
                    tweet_dev_med_3[t_id] = np.median(val_list)
                    tweet_dev_var_3[t_id] = np.var(val_list)

                    tweet_dev_avg_l_3.append(np.mean(val_list))
                    tweet_dev_med_l_3.append(np.median(val_list))
                    tweet_dev_var_l_3.append(np.var(val_list))

                    tweet_abs_dev_avg_3[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_3[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_3[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_3.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_3.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_3.append(np.var(abs_var_err))

            ##################################################

            len_cat_dict = {}

            # tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)
            tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)

            # tweet_vote_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            tweet_var_sort = sorted(tweet_var_2, key=tweet_var_2.get, reverse=True)
            # tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)
            # tweet_var_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)

            dispt_diff = collections.defaultdict()
            TPB_diff = collections.defaultdict()
            tweet_avg_ll = []
            tweet_avg_ll_2 = []
            tweet_var_ll = []
            tweet_var_ll_2 = []
            tweet_avg_ll_3 = []
            tweet_var_ll_3 = []
            TPB_l=[]
            TPB_l_2=[]
            TPB_l_3=[]
            for t_id in tweet_var_sort:
                tweet_avg_ll_2.append(tweet_avg_2[t_id])
                # tweet_avg_ll.append(tweet_avg[t_id])

                tweet_var_ll_2.append(tweet_var_2[t_id])
                # tweet_var_ll.append(tweet_var[t_id])

                tweet_avg_ll_3.append(tweet_avg_3[t_id])

                tweet_var_ll_3.append(tweet_var_3[t_id])
                # TPB_l.append(tweet_abs_dev_avg[t_id])
                TPB_l_2.append(tweet_abs_dev_avg_2[t_id])
                TPB_l_3.append(tweet_abs_dev_avg_3[t_id])


                # dispt_diff[t_id] = np.abs(tweet_var_2[t_id]-tweet_var[t_id])

                # TPB_diff[t_id] = np.abs(tweet_abs_dev_avg_2[t_id] - tweet_abs_dev_avg[t_id])
            # mplpl.scatter(range(len(tweet_avg_ll_2)),tweet_avg_ll_2,color='r', label='PTL(AMT)')
            # mplpl.scatter(range(len(tweet_avg_ll)),tweet_avg_ll,color='b', label='PTL(SSI)')
            # # mplpl.xlim([-.02, 1.02])
            # # mplpl.ylim([0, 1.02])
            # mplpl.xlabel('Snopes news stories sorted based on PTL(AMT)', fontsize=18)
            # mplpl.ylabel('PTL', fontsize=18)
            # # #
            # mplpl.legend(loc="upper right")
            # # #
            # mplpl.figure()
            #
            # mplpl.scatter(range(len(tweet_var_ll_2)),tweet_var_ll_2,color='r', label='PTL(AMT)')
            # mplpl.scatter(range(len(tweet_var_ll)),tweet_var_ll,color='b', label='PTL(SSI)')
            # # mplpl.xlim([-.02, 1.02])
            # # mplpl.ylim([0, 1.02])
            # mplpl.xlabel('Snopes news stories sorted based on Disputability(AMT)', fontsize=18)
            # mplpl.ylabel('Disputability', fontsize=18)
            # # #
            # mplpl.legend(loc="upper right")


            # mplpl.figure()
            # mplpl.hist([ans_l,ans_l_2, ans_l_3],normed=True, color=['r','b', 'g'], label=['SSI','AMT', 'AMT_incentive'])
            mplpl.hist([ans_l_2, ans_l_3],normed=True, color=['c', 'g'], label=['AMT', 'AMT_incentive'])
            # mplpl.figure()
            # mplpl.hist(ans_l_2,color='b')
            # # mplpl.grid()
            # # mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean((tweet_abs_dev_avg_rnd.values())),4)))
            # # pp = remotedir + '/fig/fig_exp1/news_based/initial/' + data_n + '_disput_nabs_perception'
            # # mplpl.savefig(pp, format='png')
            # # mplpl.figure()
            mplpl.legend(loc="upper left")
            labels = ['Confirm \nit to\nbe a false', 'Very likely\nto be\na false', 'Possibly\nfalse',
                      'Can\'t tell', 'Possibly\ntrue', 'Very likely\nto be\ntrue', 'Confirm\nit to be\ntrue']
            x = range(1, 8)
            mplpl.xticks(x, labels)#, rotation='90')
            mplpl.subplots_adjust(bottom=0.2)
            mplpl.xlabel('Workers judgements', fontsize=18)

            # mplpl.show()
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + 'SSI_AMT_judgment'
            pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + 'AMT_AMTincentive_judgment_new_1'
            mplpl.savefig(pp + '.pdf', format='pdf')
            mplpl.savefig(pp + '.png', format='png')

            # tweet_var_sort = sorted(dispt_diff, key=dispt_diff.get, reverse=True)
            # tweet_var_sort = sorted(TPB_diff, key=TPB_diff.get, reverse=True)
            tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)

            # for t_id in tweet_var_sort:
            #     print('||' + str(t_id) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id]+'||' + str(tweet_avg_2[t_id]) +
            #           '||' + str(tweet_avg[t_id]) + '||' + str(tweet_var_2[t_id]) + '||'
            #           +str(tweet_var[t_id]) + '||'+str(tweet_abs_dev_avg_2[t_id]) + '||' + str(tweet_abs_dev_avg[t_id])+'||')


    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_effect_10claims":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []


        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        claims_10_list = []

        for dataset in ['snopes_incentive_10']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_2'or dataset == 'snopes_incentive_10':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
            if dataset == 'snopes_incentive_10':
                data_n = 'sp_incentive_10'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_incentive_10'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_2 = {}
            tweet_dev_med_2 = {}
            tweet_dev_var_2 = {}
            tweet_avg_2 = {}
            tweet_med_2 = {}
            tweet_var_2 = {}
            tweet_gt_var_2 = {}

            tweet_dev_avg_l_2 = []
            tweet_dev_med_l_2 = []
            tweet_dev_var_l_2 = []
            tweet_avg_l_2 = []
            tweet_med_l_2 = []
            tweet_var_l_2 = []
            tweet_gt_var_l_2 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_2 = {}
            tweet_abs_dev_med_2 = {}
            tweet_abs_dev_var_2 = {}

            tweet_abs_dev_avg_l_2 = []
            tweet_abs_dev_med_l_2 = []
            tweet_abs_dev_var_l_2 = []

            tweet_abs_dev_avg_rnd_2 = {}
            tweet_dev_avg_rnd_2 = {}

            tweet_skew_2 = {}
            tweet_skew_l_2 = []

            tweet_vote_avg_med_var_2 = collections.defaultdict(list)
            tweet_vote_avg_2 = collections.defaultdict()
            tweet_vote_med_2 = collections.defaultdict()
            tweet_vote_var_2 = collections.defaultdict()

            tweet_avg_group_2 = collections.defaultdict()
            tweet_med_group_2 = collections.defaultdict()
            tweet_var_group_2 = collections.defaultdict()
            tweet_var_diff_group_2 = collections.defaultdict()

            tweet_kldiv_group_2 = collections.defaultdict()

            tweet_vote_avg_l_2 = []
            tweet_vote_med_l_2 = []
            tweet_vote_var_l_2 = []
            tweet_chi_group_2 = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            ind_l=[1]
            ans_l_2 = []
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()


                demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment',
                                         'income','political_view', 'race', 'marital_status']
                for demographic_feat in demographic_feat_list:
                    print('--------------------' + demographic_feat + '--------------------')
                    for lean_f in set(df_m[demographic_feat]):
                        df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                        print(lean_f + ' : '+str(len(set(df_lean_f['worker_id']))))

                claims_10_list = grouped.groups.keys()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l_2 += list(df_tmp['ra'])
                    tweet_avg_group_2[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_2[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_2[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_2[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_2[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_2[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_2[t_id] = np.mean(val_list)
                    tweet_med_2[t_id] = np.median(val_list)
                    tweet_var_2[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_2.append(np.mean(val_list))
                    tweet_med_l_2.append(np.median(val_list))
                    tweet_var_l_2.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_2[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg_2[t_id] = np.mean(vot_list)
                    tweet_vote_med_2[t_id] = np.median(vot_list)
                    tweet_vote_var_2[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_2.append(np.mean(vot_list))
                    tweet_vote_med_l_2.append(np.median(vot_list))
                    tweet_vote_var_l_2.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg_2[t_id] = np.mean(val_list)
                    tweet_dev_med_2[t_id] = np.median(val_list)
                    tweet_dev_var_2[t_id] = np.var(val_list)

                    tweet_dev_avg_l_2.append(np.mean(val_list))
                    tweet_dev_med_l_2.append(np.median(val_list))
                    tweet_dev_var_l_2.append(np.var(val_list))

                    tweet_abs_dev_avg_2[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_2[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_2[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_2.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_2.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_2.append(np.var(abs_var_err))

        for dataset in ['snopes_incentive']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_3 = {}
            tweet_dev_med_3 = {}
            tweet_dev_var_3 = {}
            tweet_avg_3 = {}
            tweet_med_3 = {}
            tweet_var_3 = {}
            tweet_gt_var_3 = {}

            tweet_dev_avg_l_3 = []
            tweet_dev_med_l_3 = []
            tweet_dev_var_l_3 = []
            tweet_avg_l_3 = []
            tweet_med_l_3 = []
            tweet_var_l_3 = []
            tweet_gt_var_l_3 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_3 = {}
            tweet_abs_dev_med_3 = {}
            tweet_abs_dev_var_3 = {}

            tweet_abs_dev_avg_l_3 = []
            tweet_abs_dev_med_l_3 = []
            tweet_abs_dev_var_l_3 = []

            tweet_abs_dev_avg_rnd_3= {}
            tweet_dev_avg_rnd_3 = {}

            tweet_skew_3 = {}
            tweet_skew_l_3 = []

            tweet_vote_avg_med_var_3 = collections.defaultdict(list)
            tweet_vote_avg_3 = collections.defaultdict()
            tweet_vote_med_3 = collections.defaultdict()
            tweet_vote_var_3 = collections.defaultdict()

            tweet_avg_group_3 = collections.defaultdict()
            tweet_med_group_3 = collections.defaultdict()
            tweet_var_group_3 = collections.defaultdict()
            tweet_var_diff_group_3 = collections.defaultdict()

            tweet_kldiv_group_3 = collections.defaultdict()

            tweet_vote_avg_l_3 = []
            tweet_vote_med_l_3 = []
            tweet_vote_var_l_3 = []
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_3 = []
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
                #                          'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))

                # for t_id in grouped.groups.keys():
                for t_id in claims_10_list:
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l_3 += list(df_tmp['ra'])
                    tweet_avg_group_3[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_3[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_3[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_3[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_3[t_id] = np.round(
                        scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_3[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    tweet_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_3[t_id] = np.mean(val_list)
                    tweet_med_3[t_id] = np.median(val_list)
                    tweet_var_3[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_3.append(np.mean(val_list))
                    tweet_med_l_3.append(np.median(val_list))
                    tweet_var_l_3.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_3[t_id] = [np.mean(vot_list), np.median(vot_list),
                                                      np.var(vot_list)]
                    tweet_vote_avg_3[t_id] = np.mean(vot_list)
                    tweet_vote_med_3[t_id] = np.median(vot_list)
                    tweet_vote_var_3[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_3.append(np.mean(vot_list))
                    tweet_vote_med_l_3.append(np.median(vot_list))
                    tweet_vote_var_l_3.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list),
                                                     np.var(val_list)]
                    tweet_dev_avg_3[t_id] = np.mean(val_list)
                    tweet_dev_med_3[t_id] = np.median(val_list)
                    tweet_dev_var_3[t_id] = np.var(val_list)

                    tweet_dev_avg_l_3.append(np.mean(val_list))
                    tweet_dev_med_l_3.append(np.median(val_list))
                    tweet_dev_var_l_3.append(np.var(val_list))

                    tweet_abs_dev_avg_3[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_3[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_3[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_3.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_3.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_3.append(np.var(abs_var_err))

            ##################################################

            len_cat_dict = {}

            # tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)
            # tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)

            # tweet_vote_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            tweet_var_sort = sorted(tweet_var_2, key=tweet_var_2.get, reverse=True)
            # tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)
            # tweet_var_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)

            dispt_diff = collections.defaultdict()
            TPB_diff = collections.defaultdict()
            tweet_avg_ll = []
            tweet_avg_ll_2 = []
            tweet_var_ll = []
            tweet_var_ll_2 = []
            tweet_avg_ll_3 = []
            tweet_var_ll_3 = []
            TPB_l=[]
            TPB_l_2=[]
            TPB_l_3=[]
            for t_id in tweet_var_sort:
                tweet_avg_ll_2.append(tweet_avg_2[t_id])
                # tweet_avg_ll.append(tweet_avg[t_id])

                tweet_var_ll_2.append(tweet_var_2[t_id])
                # tweet_var_ll.append(tweet_var[t_id])

                tweet_avg_ll_3.append(tweet_avg_3[t_id])

                tweet_var_ll_3.append(tweet_var_3[t_id])
                # TPB_l.append(tweet_abs_dev_avg[t_id])
                TPB_l_2.append(tweet_abs_dev_avg_2[t_id])
                TPB_l_3.append(tweet_abs_dev_avg_3[t_id])


                # dispt_diff[t_id] = np.abs(tweet_var_2[t_id]-tweet_var[t_id])

                # TPB_diff[t_id] = np.abs(tweet_abs_dev_avg_2[t_id] - tweet_abs_dev_avg[t_id])
            # mplpl.scatter(range(len(tweet_avg_ll_2)),tweet_avg_ll_2,color='r', label='PTL(AMT)')
            # mplpl.scatter(range(len(tweet_avg_ll)),tweet_avg_ll,color='b', label='PTL(SSI)')
            # # mplpl.xlim([-.02, 1.02])
            # # mplpl.ylim([0, 1.02])
            # mplpl.xlabel('Snopes news stories sorted based on PTL(AMT)', fontsize=18)
            # mplpl.ylabel('PTL', fontsize=18)
            # # #
            # mplpl.legend(loc="upper right")
            # # #
            # mplpl.figure()
            #
            # mplpl.scatter(range(len(tweet_var_ll_2)),tweet_var_ll_2,color='r', label='PTL(AMT)')
            # mplpl.scatter(range(len(tweet_var_ll)),tweet_var_ll,color='b', label='PTL(SSI)')
            # # mplpl.xlim([-.02, 1.02])
            # # mplpl.ylim([0, 1.02])
            # mplpl.xlabel('Snopes news stories sorted based on Disputability(AMT)', fontsize=18)
            # mplpl.ylabel('Disputability', fontsize=18)
            # # #
            # mplpl.legend(loc="upper right")


            # mplpl.figure()
            # mplpl.hist([ans_l,ans_l_2, ans_l_3],normed=True, color=['r','b', 'g'], label=['SSI','AMT', 'AMT_incentive'])
            mplpl.hist([ans_l_2, ans_l_3],normed=True, color=['c', 'g'], label=['AMT_incentive_10claims', 'AMT_incentive_50claims'])
            # mplpl.figure()
            # mplpl.hist(ans_l_2,color='b')
            # # mplpl.grid()
            # # mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean((tweet_abs_dev_avg_rnd.values())),4)))
            # # pp = remotedir + '/fig/fig_exp1/news_based/initial/' + data_n + '_disput_nabs_perception'
            # # mplpl.savefig(pp, format='png')
            # # mplpl.figure()
            mplpl.legend(loc="upper left")
            labels = ['Confirm \nit to\nbe a false', 'Very likely\nto be\na false', 'Possibly\nfalse',
                      'Can\'t tell', 'Possibly\ntrue', 'Very likely\nto be\ntrue', 'Confirm\nit to be\ntrue']
            x = range(1, 8)
            mplpl.xticks(x, labels)#, rotation='90')
            mplpl.subplots_adjust(bottom=0.2)
            mplpl.xlabel('Workers judgements', fontsize=18)

            # mplpl.show()
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + 'SSI_AMT_judgment'
            pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + 'AMTincentive_10_50_judgment_new_1'
            mplpl.savefig(pp + '.pdf', format='pdf')
            mplpl.savefig(pp + '.png', format='png')

            # tweet_var_sort = sorted(dispt_diff, key=dispt_diff.get, reverse=True)
            # tweet_var_sort = sorted(TPB_diff, key=TPB_diff.get, reverse=True)
            tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)

            # for t_id in tweet_var_sort:
            #     print('||' + str(t_id) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id]+'||' + str(tweet_avg_2[t_id]) +
            #           '||' + str(tweet_avg[t_id]) + '||' + str(tweet_var_2[t_id]) + '||'
            #           +str(tweet_var[t_id]) + '||'+str(tweet_abs_dev_avg_2[t_id]) + '||' + str(tweet_abs_dev_avg[t_id])+'||')


    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_creating_features_prediction_old":

        one_d_array = [1, 2, 3]
        list_tmp = []
        list_tmp.append([one_d_array])
        list_tmp.append([one_d_array])

        pd.DataFrame([
            [one_d_array],
            [one_d_array]])

        # array([[0., 1., 1., 0., 0., 1., 0., 0., 0.]])

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        if dataset == 'mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                  + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

            exp1_list = sample_tweets_exp1
            tweet_id = 100010
            publisher_name = 110
            tweet_popularity = {}
            tweet_text_dic = {}
            for input_file in [input_rumor, input_non_rumor]:
                for line in input_file:
                    line.replace('\n', '')
                    line_splt = line.split('\t')
                    tweet_txt = line_splt[1]
                    tweet_link = line_splt[1]
                    tweet_id += 1
                    publisher_name += 1
                    tweet_popularity[tweet_id] = int(line_splt[2])
                    tweet_text_dic[tweet_id] = tweet_txt

            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor = {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id) < 100060:
                    tweet_lable_dict[tweet_id] = 'rumor'
                else:
                    tweet_lable_dict[tweet_id] = 'non-rumor'

        if dataset == 'snopes' or dataset == 'snopes_ssi':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            # remotedir = ''
            inp_all = glob.glob(remotedir + 'all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable

        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        ##########################prepare balanced data (same number of rep, dem, neut #############

        #
        # if dataset=='snopes':
        #     data_n = 'sp'
        #     ind_l = [1,2,3]
        # elif dataset=='politifact':
        #     data_n = 'pf'
        #     ind_l = [1,2,3]
        # elif dataset=='mia':
        #     data_n = 'mia'
        #     ind_l = [1]
        #
        # for ind in ind_l:
        #     if dataset == 'mia':
        #         inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp_final.csv'
        #         inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp.csv'
        #         df[ind] = pd.read_csv(inp1, sep="\t")
        #         df_w[ind] = pd.read_csv(inp1_w, sep="\t")
        #
        #         df_m = df[ind].copy()
        #     else:
        #         inp1 = remotedir  +'amt_answers_'+data_n+'_claims_exp'+str(ind)+'_final.csv'
        #         inp1_w = remotedir  +'worker_amt_answers_'+data_n+'_claims_exp'+str(ind)+'.csv'
        #         df[ind] = pd.read_csv(inp1, sep="\t")
        #         df_w[ind] = pd.read_csv(inp1_w, sep="\t")
        #
        #         df_m = df[ind].copy()
        #
        #
        #
        #     rep_num = len(df_m[df_m['leaning']==-1])/float(60)
        #     dem_num = len(df_m[df_m['leaning'] == 1])/float(60)
        #     neut_num = len(df_m[df_m['leaning'] == 0])/float(60)
        #
        #     min_num = np.min([int(rep_num), int(dem_num), int(neut_num)])
        #
        #     dem_workers = list(set(df_m[df_m['leaning'] == 1]['worker_id']))
        #     rep_workers = list(set(df_m[df_m['leaning'] == -1]['worker_id']))
        #     neut_workers = list(set(df_m[df_m['leaning'] == 0]['worker_id']))
        #
        #     random.shuffle(dem_workers)
        #     random.shuffle(rep_workers)
        #     random.shuffle(neut_workers)
        #
        #     dem_workers = dem_workers[:min_num]
        #     rep_workers = rep_workers[:min_num]
        #     neut_workers = neut_workers[:min_num]
        #
        #     all_workers = []
        #     all_workers += dem_workers
        #     all_workers += rep_workers
        #     all_workers += neut_workers
        #
        #     df[ind] = df_m[df_m['worker_id'].isin(all_workers)]
        #
        #     df[ind].to_csv(remotedir + 'amt_answers_'+data_n+'_claims_exp'+str(ind)+'_final_balanced.csv',
        #                 columns=df[ind].columns, sep="\t", index=False)
        #
        # exit()

        # balance_f = 'balanced'


        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes','snopes_ssi', 'mia',  'snopes_nonpol', 'snopes', 'politifact']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir = ''
                inp_all = glob.glob(remotedir + 'all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
                num_claims = 150
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
                num_claims = 150

            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
                num_claims = 100
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
                num_claims = 180

            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'
                num_claims = 60

            columns_l = []
            columns_l.append('label')
            columns_l.append('tweet_id')
            # for i in range(0,100):
            for i in range(1, 8):
                columns_l.append(str('pt_count_' + str(i)))
            # for i in range(0, 100):
            #         columns_l.append(str('worker_pt_' + str(i)))

            # for i in range(0, 100):
            #         columns_l.append(str('worker_leaning_' + str(i)))

            df_feat = pd.DataFrame(index=range(num_claims), columns=columns_l)
            df_feat = df_feat.fillna(0)
            df_feat.loc[:, 'tpb'] = df_feat['tweet_id'] * 0.0
            df_feat.loc[:, 'text'] = df_feat['tweet_id'] * 0.0
            enc = preprocessing.OneHotEncoder(n_values=[3, 7])
            # lb = preprocessing.LabelBinarizer()
            # lb.fit(['dem', 'rep', 'neut', '1','2','3','4','5','6','7'])
            enc.fit([1, 1])
            test = np.array(enc.transform([[0, 0]]).toarray())
            # test = np.array(lb.transform(['dem', '1']))
            test_list = []
            # for j in range(0,100):
            #     test_list = []
            #     for i in range(len(df_feat)):
            #         test_list.append([list(test[0])])
            #     df_tmp_f = pd.DataFrame(test_list, columns=[str('worker_leaning_' + str(j))])
            #     df_feat = pd.concat([df_feat, df_tmp_f], axis=1, join='inner')
            #

            # df_feat.loc[:, str('worker_leaning_' + str(i))] = df_feat['tweet_id'] * [0]
            # columns_l.append(str('worker_leaning_' + str(i)))

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                # inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp1_final.csv'
                df_in = pd.read_csv(inp1, sep="\t")

                demographic_feat = 'leaning'
                for lean_f in set(df_in[demographic_feat]):
                    for judge in set(df_in['ra']):
                        print (str(lean_f) + ' : ' + str(judge))
                        if demographic_feat + '_' + str(lean_f) + ',' + 'pt_' + str(judge) not in df_feat.columns:
                            df_feat.loc[:, demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(judge)] = df_feat[
                                                                                                                  'tweet_id'] * 0.0

                # converting_demographic_num
                demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment', 'income',
                                         'political_view', 'race', 'marital_status']
                for demographic_feat in demographic_feat_list:
                    for lean_f in set(df_in[demographic_feat]):
                        for judge in set(df_in['ra']):
                            # print (str(lean_f) + ' : ' + str(judge))
                            if demographic_feat + '_' + str(lean_f) + ',' + 'pt_' + str(judge) not in df_feat.columns:
                                df_feat.loc[:, demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(judge)] = df_feat[
                                                                                                                      'tweet_id'] * 0.0

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            rel_gt_dict = {}
            news_cat_list_tf = [4, 2, 3, 1]
            t_f_dict_len = collections.defaultdict(int)
            t_f_dict = {}
            if dataset == 'snopes' or dataset == 'snopes_nonpol'or dataset == 'snopes_ssi':
                news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_v = [-1, -.5, 0, 0.5, 1]
                rel_gt_dict['FALSE'] = -2
                rel_gt_dict['MOSTLY FALSE'] = -1
                rel_gt_dict['MIXTURE'] = 0
                rel_gt_dict['MOSTLY TRUE'] = 1
                rel_gt_dict['TRUE'] = 2
            if dataset == 'politifact':
                news_cat_list_t_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_v = [-1, -1, -0.5, 0, 0.5, 1]
                rel_gt_l = [-2, -2, -1, 0, 1, 2]
                rel_gt_dict['pants-fire'] = -3
                rel_gt_dict['false'] = -2
                rel_gt_dict['mostly-false'] = -1
                rel_gt_dict['half-true'] = 0
                rel_gt_dict['mostly-true'] = 1
                rel_gt_dict['true'] = 2

            if dataset == 'mia':
                news_cat_list_t_f = ['rumor', 'non-rumor']
                news_cat_list_v = [-1, 1]
                rel_gt_l = [-1, 1]
                rel_gt_dict['rumor'] = -1
                rel_gt_dict['non-rumor'] = 1

            w_fnb_dict = collections.defaultdict()
            w_fpb_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()
            gt_acc = collections.defaultdict()

            for cat in news_cat_list_v:
                gt_acc[cat] = [0] * (len(news_cat_list_t_f))
            t_id_ind = 0

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                ccc = 0
                for t_id in grouped.groups.keys():
                    if t_id == 1:
                        continue

                    ccc += 1
                    print(ccc)
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    df_tmp_ind = df_tmp[df_tmp['tweet_id'] == t_id].index.tolist()[0]

                    groupby_ftr = 'ra'
                    grouped_ra = df_tmp.groupby(groupby_ftr, sort=False)
                    grouped_count = df_tmp.groupby(groupby_ftr, sort=False).count()
                    # ra_list = grouped_groups.keys()
                    # print(grouped_count.index.tolist())
                    for i in range(1, 8):
                        try:
                            df_feat['pt_count_' + str(i)][t_id_ind] = grouped_count['age'][i]
                        except:
                            df_feat['pt_count_' + str(i)][t_id_ind] = 0
                            continue
                    df_tmp_sort = df_tmp.sort(['worker_id'])

                    demographic_feat = 'leaning'
                    for lean_f in set(df_tmp[demographic_feat]):
                        df_lean_f = df_tmp[df_tmp[demographic_feat] == lean_f]
                        if len(df_lean_f) == 0:
                            continue
                        groupby_ftr = 'ra'
                        grouped_ra = df_lean_f.groupby(groupby_ftr, sort=False)
                        grouped_count = df_lean_f.groupby(groupby_ftr, sort=False).count()
                        for i in range(1, 8):
                            try:
                                df_feat[demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][t_id_ind] = \
                                grouped_count['age'][i] / float(len(df_lean_f))
                            except:
                                df_feat[demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][t_id_ind] = 0
                                continue

                    # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment',
                    #                          'income',
                    #                          'political_view', 'race', 'marital_status']
                    for demographic_feat in demographic_feat_list:
                        print(demographic_feat)
                        for lean_f in set(df_tmp[demographic_feat]):
                            df_lean_f = df_tmp[df_tmp[demographic_feat] == lean_f]
                            if len(df_lean_f) == 0:
                                continue
                            groupby_ftr = 'ra'
                            grouped_ra = df_lean_f.groupby(groupby_ftr, sort=False)
                            grouped_count = df_lean_f.groupby(groupby_ftr, sort=False).count()
                            for i in range(1, 8):
                                try:
                                    df_feat[demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][
                                        t_id_ind] = grouped_count['age'][i] / float(len(df_lean_f))
                                except:
                                    df_feat[demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][t_id_ind] = 0
                                    continue


                                    # for judge in set(df_tmp['ra']):
                                    #     # print (str(lean_f) + ' : ' + str(judge))
                                    #     df_feat.loc[:, str(lean_f) + '_' + str(judge)] = df_feat['tweet_id'] * 0.0
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]






                    df_feat['tweet_id'][t_id_ind] = t_id
                    df_feat['tpb'][t_id_ind] = np.mean(abs_var_err)
                    df_feat['text'][t_id_ind] = df_tmp['text'][df_tmp_ind]
                    df_feat['label'][t_id_ind] = rel_gt_dict[df_tmp['ra_gt'][df_tmp_ind]]
                    w_c = 0
                    # for index in df_tmp_sort.index.tolist():
                    #     ra = df_tmp_sort['ra'][index]
                    #     lean_tmp = df_tmp_sort['leaning'][index]
                    #     # if lean_tmp==-1:
                    #     #     lean='rep'
                    #     # elif lean_tmp==0:
                    #     #     lean ='netu'
                    #     # elif lean_tmp==1:
                    #     #     lean='dem'
                    #     if lean_tmp==-1:
                    #         lean=1
                    #     elif lean_tmp==0:
                    #         lean =2
                    #     elif lean_tmp==1:
                    #         lean=3
                    #
                    #     if ra==1:
                    #         rel = -3
                    #     elif ra==2:
                    #         rel=-2
                    #     elif ra==3:
                    #         rel=-1
                    #
                    #
                    #     elif ra==4:
                    #         rel=0
                    #     elif ra==5:
                    #         rel = 1
                    #     elif ra==6:
                    #         rel = 2
                    #     elif ra==7:
                    #         rel = 3
                    #     pt = rel
                    #     try:
                    #         df_feat[str('worker_pt_' + str(w_c))][t_id_ind] = pt
                    #         # df_feat[str('worker_leaning_' + str(w_c))][t_id_ind] = lean * ra
                    #         # df_feat[str('worker_leaning_' + str(w_c))][t_id_ind] = lb.transform([lean ,ra])
                    #         # df_feat[str('worker_leaning_' + str(w_c))][t_id_ind] = enc.transform([[lean-1 ,ra-1]]).toarray()[0].tolist()
                    #     except:
                    #         continue
                    #     w_c+=1
                    t_id_ind += 1
            # for i in range(len(df_feat)):


            df_feat.to_csv(remotedir + 'fake_news_features_' + data_n + '.csv', columns=df_feat.columns, sep="\t",
                           index=False)
            exit()

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_creating_features_prediction":

        one_d_array = [1, 2, 3]
        list_tmp = []
        list_tmp.append([one_d_array])
        list_tmp.append([one_d_array])

        pd.DataFrame([
            [one_d_array],
            [one_d_array]])

        # array([[0., 1., 1., 0., 0., 1., 0., 0., 0.]])

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        if dataset == 'mia':
            local_dir_saving = ''
            remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'

            final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                  + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

            sample_tweets_exp1 = json.load(final_inp_exp1)

            input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
            input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

            exp1_list = sample_tweets_exp1
            tweet_id = 100010
            publisher_name = 110
            tweet_popularity = {}
            tweet_text_dic = {}
            for input_file in [input_rumor, input_non_rumor]:
                for line in input_file:
                    line.replace('\n', '')
                    line_splt = line.split('\t')
                    tweet_txt = line_splt[1]
                    tweet_link = line_splt[1]
                    tweet_id += 1
                    publisher_name += 1
                    tweet_popularity[tweet_id] = int(line_splt[2])
                    tweet_text_dic[tweet_id] = tweet_txt

            out_list = []
            cnn_list = []
            foxnews_list = []
            ap_list = []
            tweet_txt_dict = {}
            tweet_link_dict = {}
            tweet_publisher_dict = {}
            tweet_rumor = {}
            tweet_lable_dict = {}
            tweet_non_rumor = {}
            pub_dict = collections.defaultdict(list)
            for tweet in exp1_list:

                tweet_id = tweet[0]
                publisher_name = tweet[1]
                tweet_txt = tweet[2]
                tweet_link = tweet[3]
                tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_link_dict[tweet_id] = tweet_link
                tweet_publisher_dict[tweet_id] = publisher_name
                if int(tweet_id) < 100060:
                    tweet_lable_dict[tweet_id] = 'rumor'
                else:
                    tweet_lable_dict[tweet_id] = 'non-rumor'

        if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_2':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            # remotedir = ''
            inp_all = glob.glob(remotedir + 'all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable

        if dataset == 'politifact':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
            inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
            news_cat_list = ['mostly_false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 6):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            for line in claims_list:
                line_splt = line.split('<<||>>')
                tweet_id = int(line_splt[2])
                tweet_txt = line_splt[3]
                publisher_name = line_splt[4]
                cat_lable = line_splt[5]
                dat = line_splt[6]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable
                tweet_publisher_dict[tweet_id] = publisher_name

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        ##########################prepare balanced data (same number of rep, dem, neut #############

        #
        # if dataset=='snopes':
        #     data_n = 'sp'
        #     ind_l = [1,2,3]
        # elif dataset=='politifact':
        #     data_n = 'pf'
        #     ind_l = [1,2,3]
        # elif dataset=='mia':
        #     data_n = 'mia'
        #     ind_l = [1]
        #
        # for ind in ind_l:
        #     if dataset == 'mia':
        #         inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp_final.csv'
        #         inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp.csv'
        #         df[ind] = pd.read_csv(inp1, sep="\t")
        #         df_w[ind] = pd.read_csv(inp1_w, sep="\t")
        #
        #         df_m = df[ind].copy()
        #     else:
        #         inp1 = remotedir  +'amt_answers_'+data_n+'_claims_exp'+str(ind)+'_final.csv'
        #         inp1_w = remotedir  +'worker_amt_answers_'+data_n+'_claims_exp'+str(ind)+'.csv'
        #         df[ind] = pd.read_csv(inp1, sep="\t")
        #         df_w[ind] = pd.read_csv(inp1_w, sep="\t")
        #
        #         df_m = df[ind].copy()
        #
        #
        #
        #     rep_num = len(df_m[df_m['leaning']==-1])/float(60)
        #     dem_num = len(df_m[df_m['leaning'] == 1])/float(60)
        #     neut_num = len(df_m[df_m['leaning'] == 0])/float(60)
        #
        #     min_num = np.min([int(rep_num), int(dem_num), int(neut_num)])
        #
        #     dem_workers = list(set(df_m[df_m['leaning'] == 1]['worker_id']))
        #     rep_workers = list(set(df_m[df_m['leaning'] == -1]['worker_id']))
        #     neut_workers = list(set(df_m[df_m['leaning'] == 0]['worker_id']))
        #
        #     random.shuffle(dem_workers)
        #     random.shuffle(rep_workers)
        #     random.shuffle(neut_workers)
        #
        #     dem_workers = dem_workers[:min_num]
        #     rep_workers = rep_workers[:min_num]
        #     neut_workers = neut_workers[:min_num]
        #
        #     all_workers = []
        #     all_workers += dem_workers
        #     all_workers += rep_workers
        #     all_workers += neut_workers
        #
        #     df[ind] = df_m[df_m['worker_id'].isin(all_workers)]
        #
        #     df[ind].to_csv(remotedir + 'amt_answers_'+data_n+'_claims_exp'+str(ind)+'_final_balanced.csv',
        #                 columns=df[ind].columns, sep="\t", index=False)
        #
        # exit()

        # balance_f = 'balanced'


        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in ['snopes_ssi','snopes_ssi','snopes_incentive','snopes_2','snopes_ssi', 'politifact', 'snopes_nonpol', 'snopes', 'mia', 'mia', 'politifact']:
            if dataset == 'mia':
                claims_list = []
                local_dir_saving = ''
                remotedir_1 = '/NS/twitter-8/work/Reza/reliable_news/data/'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/mia/'
                news_cat_list = ['rumor', 'non-rumor']
                final_inp_exp1 = open(remotedir_1 + local_dir_saving
                                      + 'amt_data-set/rumor_non-rumor_news_exp1.txt', 'r')

                sample_tweets_exp1 = json.load(final_inp_exp1)

                input_rumor = open(remotedir_1 + local_dir_saving + 'rumer_tweets', 'r')
                input_non_rumor = open(remotedir_1 + local_dir_saving + 'non_rumer_tweets', 'r')

                exp1_list = sample_tweets_exp1

                out_list = []
                cnn_list = []
                foxnews_list = []
                ap_list = []
                tweet_txt_dict = {}
                tweet_link_dict = {}
                tweet_publisher_dict = {}
                tweet_rumor = {}
                tweet_lable_dict = {}
                tweet_non_rumor = {}
                pub_dict = collections.defaultdict(list)
                for tweet in exp1_list:

                    tweet_id = tweet[0]
                    publisher_name = tweet[1]
                    tweet_txt = tweet[2]
                    tweet_link = tweet[3]
                    tmp_list = [tweet_id, publisher_name, tweet_txt, tweet_link]
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_link_dict[tweet_id] = tweet_link
                    tweet_publisher_dict[tweet_id] = publisher_name
                    if int(tweet_id) < 100060:
                        tweet_lable_dict[tweet_id] = 'rumor'
                    else:
                        tweet_lable_dict[tweet_id] = 'non-rumor'
                        # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir = ''
                inp_all = glob.glob(remotedir + 'all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes_nonpol':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/non_politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'politifact':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/politifact/'
                inp_all = glob.glob(remotedir + 'politic_claims/*.txt')
                news_cat_list = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 6):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    tweet_id = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    publisher_name = line_splt[4]
                    cat_lable = line_splt[5]
                    dat = line_splt[6]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    tweet_publisher_dict[tweet_id] = publisher_name
                    # outF = open(remotedir + 'table_out.txt', 'w')

            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes'
                num_claims = 50
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
                num_claims = 50

            if dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_incentive'
                num_claims = 50

            if dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_2'
                num_claims = 52


            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
                num_claims = 100
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
                num_claims = 60

            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'
                num_claims = 60

            columns_l = []
            columns_l.append('label')
            columns_l.append('tweet_id')
            # for i in range(0,100):
            for i in range(1, 8):
                columns_l.append(str('pt_count_' + str(i)))
            # for i in range(0, 100):
            #         columns_l.append(str('worker_pt_' + str(i)))

            # for i in range(0, 100):
            #         columns_l.append(str('worker_leaning_' + str(i)))
            df_feat = collections.defaultdict()
            for ind in ind_l:
                df_feat[ind] = pd.DataFrame(index=range(num_claims), columns=columns_l)
                df_feat[ind] = df_feat[ind].fillna(0)
                df_feat[ind].loc[:, 'tpb'] = df_feat[ind]['tweet_id'] * 0.0
                df_feat[ind].loc[:, 'disput'] = df_feat[ind]['tweet_id'] * 0.0
                df_feat[ind].loc[:, 'disput_4'] = df_feat[ind]['tweet_id'] * 0.0
                df_feat[ind].loc[:, 'disput_3_4_5'] = df_feat[ind]['tweet_id'] * 0.0
                df_feat[ind].loc[:, 'text'] = df_feat[ind]['tweet_id'] * 0.0
                enc = preprocessing.OneHotEncoder(n_values=[3, 7])
                enc.fit([1, 1])
                test = np.array(enc.transform([[0, 0]]).toarray())
                # test = np.array(lb.transform(['dem', '1']))
                test_list = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                # inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp1_final.csv'
                df_in = pd.read_csv(inp1, sep="\t")

                demographic_feat = 'leaning'
                for lean_f in set(df_in[demographic_feat]):
                    # for judge in set(df_in['ra']):
                    for judge in set([1,2,3,4,5,6,7]):
                        print (str(lean_f) + ' : ' + str(judge))
                        new_feat = demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(judge)
                        for ind in ind_l:
                            if new_feat not in df_feat[ind].columns:
                                df_feat[ind].loc[:, new_feat] = df_feat[ind]['tweet_id'] * 0.0

                # converting_demographic_num
                demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment', 'income',
                                         'political_view', 'race', 'marital_status']
                for demographic_feat in demographic_feat_list:
                    for lean_f in set(df_in[demographic_feat]):
                        # for judge in set(df_in['ra']):
                        for judge in set([1, 2, 3, 4, 5, 6, 7]):
                            # print (str(lean_f) + ' : ' + str(judge))
                            new_feat = demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(judge)
                            for ind in ind_l:
                                if new_feat not in df_feat[ind].columns:
                                    df_feat[ind].loc[:, new_feat] = df_feat[ind]['tweet_id'] * 0.0

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            rel_gt_dict = {}
            news_cat_list_tf = [4, 2, 3, 1]
            t_f_dict_len = collections.defaultdict(int)
            t_f_dict = {}
            if dataset == 'snopes' or dataset == 'snopes_nonpol'or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_2':
                news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_v = [-1, -.5, 0, 0.5, 1]
                rel_gt_dict['FALSE'] = -2
                rel_gt_dict['MOSTLY FALSE'] = -1
                rel_gt_dict['MIXTURE'] = 0
                rel_gt_dict['MOSTLY TRUE'] = 1
                rel_gt_dict['TRUE'] = 2
            if dataset == 'politifact':
                news_cat_list_t_f = ['pants-fire', 'false', 'mostly-false', 'half-true', 'mostly-true', 'true']
                news_cat_list_v = [-1, -1, -0.5, 0, 0.5, 1]
                rel_gt_l = [-2, -2, -1, 0, 1, 2]
                rel_gt_dict['pants-fire'] = -3
                rel_gt_dict['false'] = -2
                rel_gt_dict['mostly-false'] = -1
                rel_gt_dict['half-true'] = 0
                rel_gt_dict['mostly-true'] = 1
                rel_gt_dict['true'] = 2

            if dataset == 'mia':
                news_cat_list_t_f = ['rumor', 'non-rumor']
                news_cat_list_v = [-1, 1]
                rel_gt_l = [-1, 1]
                rel_gt_dict['rumor'] = -1
                rel_gt_dict['non-rumor'] = 1

            w_fnb_dict = collections.defaultdict()
            w_fpb_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()
            gt_acc = collections.defaultdict()

            for cat in news_cat_list_v:
                gt_acc[cat] = [0] * (len(news_cat_list_t_f))

            for ind in ind_l:
                t_id_ind = 0

                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                ccc = 0
                for t_id in grouped.groups.keys():
                    if t_id == 1:
                        continue

                    ccc += 1
                    print(ccc)
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    df_tmp_ind = df_tmp[df_tmp['tweet_id'] == t_id].index.tolist()[0]

                    groupby_ftr = 'ra'
                    grouped_ra = df_tmp.groupby(groupby_ftr, sort=False)
                    grouped_count = df_tmp.groupby(groupby_ftr, sort=False).count()
                    # ra_list = grouped_groups.keys()
                    # print(grouped_count.index.tolist())
                    for i in range(1, 8):
                        try:
                            df_feat[ind]['pt_count_' + str(i)][t_id_ind] = grouped_count['age'][i]
                        except:
                            df_feat[ind]['pt_count_' + str(i)][t_id_ind] = 0
                            continue
                    df_tmp_sort = df_tmp.sort(['worker_id'])

                    demographic_feat = 'leaning'
                    for lean_f in set(df_tmp[demographic_feat]):
                        df_lean_f = df_tmp[df_tmp[demographic_feat] == lean_f]
                        if len(df_lean_f) == 0:
                            continue
                        groupby_ftr = 'ra'
                        grouped_ra = df_lean_f.groupby(groupby_ftr, sort=False)
                        grouped_count = df_lean_f.groupby(groupby_ftr, sort=False).count()
                        for i in range(1, 8):
                            try:
                                df_feat[ind][demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][t_id_ind] = \
                                grouped_count['age'][i] / float(len(df_lean_f))
                            except:
                                df_feat[ind][demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][t_id_ind] = 0
                                continue

                    for demographic_feat in demographic_feat_list:
                        print(demographic_feat)
                        for lean_f in set(df_tmp[demographic_feat]):
                            df_lean_f = df_tmp[df_tmp[demographic_feat] == lean_f]
                            if len(df_lean_f) == 0:
                                continue
                            groupby_ftr = 'ra'
                            grouped_ra = df_lean_f.groupby(groupby_ftr, sort=False)
                            grouped_count = df_lean_f.groupby(groupby_ftr, sort=False).count()
                            for i in range(1, 8):
                                try:
                                    df_feat[ind][demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][
                                        t_id_ind] = grouped_count['age'][i] / float(len(df_lean_f))
                                except:
                                    df_feat[ind][demographic_feat + '_' + str(lean_f) + '_' + 'pt_' + str(i)][t_id_ind] = 0
                                    continue


                                    # for judge in set(df_tmp['ra']):
                                    #     # print (str(lean_f) + ' : ' + str(judge))
                                    #     df_feat.loc[:, str(lean_f) + '_' + str(judge)] = df_feat['tweet_id'] * 0.0

                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]


                    df_feat[ind]['tweet_id'][t_id_ind] = t_id
                    df_feat[ind]['tpb'][t_id_ind] = np.mean(abs_var_err)
                    df_feat[ind]['text'][t_id_ind] = df_tmp['text'][df_tmp_ind]
                    df_feat[ind]['label'][t_id_ind] = rel_gt_dict[df_tmp['ra_gt'][df_tmp_ind]]


                    df_feat[ind]['disput'][t_id_ind] = np.var(list(df_tmp['rel_v']))
                    df_tmp = df_tmp[df_tmp['ra']!=4]
                    df_feat[ind]['disput_4'][t_id_ind] = np.var(list(df_tmp['rel_v']))
                    df_tmp = df_tmp[df_tmp['ra']!=3]
                    df_tmp = df_tmp[df_tmp['ra']!=5]
                    df_feat[ind]['disput_3_4_5'][t_id_ind] = np.var(list(df_tmp['rel_v']))
                    # ind_t = df_tmp.index.tolist()[0]




                    w_c = 0
                    # for index in df_tmp_sort.index.tolist():
                    #     ra = df_tmp_sort['ra'][index]
                    #     lean_tmp = df_tmp_sort['leaning'][index]
                    #     # if lean_tmp==-1:
                    #     #     lean='rep'
                    #     # elif lean_tmp==0:
                    #     #     lean ='netu'
                    #     # elif lean_tmp==1:
                    #     #     lean='dem'
                    #     if lean_tmp==-1:
                    #         lean=1
                    #     elif lean_tmp==0:
                    #         lean =2
                    #     elif lean_tmp==1:
                    #         lean=3
                    #
                    #     if ra==1:
                    #         rel = -3
                    #     elif ra==2:
                    #         rel=-2
                    #     elif ra==3:
                    #         rel=-1
                    #
                    #
                    #     elif ra==4:
                    #         rel=0
                    #     elif ra==5:
                    #         rel = 1
                    #     elif ra==6:
                    #         rel = 2
                    #     elif ra==7:
                    #         rel = 3
                    #     pt = rel
                    #     try:
                    #         df_feat[str('worker_pt_' + str(w_c))][t_id_ind] = pt
                    #         # df_feat[str('worker_leaning_' + str(w_c))][t_id_ind] = lean * ra
                    #         # df_feat[str('worker_leaning_' + str(w_c))][t_id_ind] = lb.transform([lean ,ra])
                    #         # df_feat[str('worker_leaning_' + str(w_c))][t_id_ind] = enc.transform([[lean-1 ,ra-1]]).toarray()[0].tolist()
                    #     except:
                    #         continue
                    #     w_c+=1
                    t_id_ind += 1
                    # for i in range(len(df_feat)):

                df_feat[ind].to_csv(remotedir + 'fake_news_features_' + data_n + '_' + str(ind) + '.csv',
                                    columns=df_feat[ind].columns, sep="\t", index=False)
            exit()





    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_writing_wiki_tweets_dist_judgments":

        for dataset in ['snopes_incentive']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_3 = {}
            tweet_dev_med_3 = {}
            tweet_dev_var_3 = {}
            tweet_avg_3 = {}
            tweet_med_3 = {}
            tweet_var_3 = {}
            tweet_gt_var_3 = {}

            tweet_dev_avg_l_3 = []
            tweet_dev_med_l_3 = []
            tweet_dev_var_l_3 = []
            tweet_avg_l_3 = []
            tweet_med_l_3 = []
            tweet_var_l_3 = []
            tweet_gt_var_l_3 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_3 = {}
            tweet_abs_dev_med_3 = {}
            tweet_abs_dev_var_3 = {}

            tweet_abs_dev_avg_l_3 = []
            tweet_abs_dev_med_l_3 = []
            tweet_abs_dev_var_l_3 = []

            tweet_abs_dev_avg_rnd_3= {}
            tweet_dev_avg_rnd_3 = {}

            tweet_skew_3 = {}
            tweet_skew_l_3 = []

            tweet_vote_avg_med_var_3 = collections.defaultdict(list)
            tweet_vote_avg_3 = collections.defaultdict()
            tweet_vote_med_3 = collections.defaultdict()
            tweet_vote_var_3 = collections.defaultdict()

            tweet_avg_group_3 = collections.defaultdict()
            tweet_med_group_3 = collections.defaultdict()
            tweet_var_group_3 = collections.defaultdict()
            tweet_var_diff_group_3 = collections.defaultdict()

            tweet_kldiv_group_3 = collections.defaultdict()

            tweet_vote_avg_l_3 = []
            tweet_vote_med_l_3 = []
            tweet_vote_var_l_3 = []
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_3 = []
            for ind in ind_l:
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
                #                          'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    # dem_val_list = list(dem_df['rel_v'])
                    # rep_val_list = list(rep_df['rel_v'])
                    # neut_val_list = list(neut_df['rel_v'])

                    dem_val_list = list(dem_df['ra'])
                    rep_val_list = list(rep_df['ra'])
                    neut_val_list = list(neut_df['ra'])

                    dem_num = len(dem_df['worker_id'])
                    rep_num = len(rep_df['worker_id'])
                    neut_num = len(neut_df['worker_id'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    val_list = list(df_tmp['ra'])
                    ans_l_3 += list(df_tmp['ra'])
                    tweet_avg_group_3[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_3[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_3[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_3[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_3[t_id] = np.round(
                        scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_3[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    val_list_ra = list(df_tmp['ra'])
                    tweet_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_3[t_id] = np.mean(val_list)
                    tweet_med_3[t_id] = np.median(val_list)
                    tweet_var_3[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_3.append(np.mean(val_list))
                    tweet_med_l_3.append(np.median(val_list))
                    tweet_var_l_3.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_3[t_id] = [np.mean(vot_list), np.median(vot_list),
                                                      np.var(vot_list)]
                    tweet_vote_avg_3[t_id] = np.mean(vot_list)
                    tweet_vote_med_3[t_id] = np.median(vot_list)
                    tweet_vote_var_3[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_3.append(np.mean(vot_list))
                    tweet_vote_med_l_3.append(np.median(vot_list))
                    tweet_vote_var_l_3.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list),
                                                     np.var(val_list)]
                    tweet_dev_avg_3[t_id] = np.mean(val_list)
                    tweet_dev_med_3[t_id] = np.median(val_list)
                    tweet_dev_var_3[t_id] = np.var(val_list)

                    tweet_dev_avg_l_3.append(np.mean(val_list))
                    tweet_dev_med_l_3.append(np.median(val_list))
                    tweet_dev_var_l_3.append(np.var(val_list))

                    tweet_abs_dev_avg_3[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_3[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_3[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_3.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_3.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_3.append(np.var(abs_var_err))

        for dataset in ['snopes_2']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_2 = {}
            tweet_dev_med_2 = {}
            tweet_dev_var_2 = {}
            tweet_avg_2 = {}
            tweet_med_2 = {}
            tweet_var_2 = {}
            tweet_gt_var_2 = {}

            tweet_dev_avg_l_2 = []
            tweet_dev_med_l_2 = []
            tweet_dev_var_l_2 = []
            tweet_avg_l_2 = []
            tweet_med_l_2 = []
            tweet_var_l_2 = []
            tweet_gt_var_l_2 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_2 = {}
            tweet_abs_dev_med_2 = {}
            tweet_abs_dev_var_2 = {}

            tweet_abs_dev_avg_l_2 = []
            tweet_abs_dev_med_l_2 = []
            tweet_abs_dev_var_l_2 = []

            tweet_abs_dev_avg_rnd_2 = {}
            tweet_dev_avg_rnd_2 = {}

            tweet_skew_2 = {}
            tweet_skew_l_2 = []

            tweet_vote_avg_med_var_2 = collections.defaultdict(list)
            tweet_vote_avg_2 = collections.defaultdict()
            tweet_vote_med_2 = collections.defaultdict()
            tweet_vote_var_2 = collections.defaultdict()

            tweet_avg_group_2 = collections.defaultdict()
            tweet_med_group_2 = collections.defaultdict()
            tweet_var_group_2 = collections.defaultdict()
            tweet_var_diff_group_2 = collections.defaultdict()

            tweet_kldiv_group_2 = collections.defaultdict()

            tweet_vote_avg_l_2 = []
            tweet_vote_med_l_2 = []
            tweet_vote_var_l_2 = []
            tweet_chi_group_2 = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_2 = []
            for ind in ind_l:
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()



                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    dem_val_list = list(dem_df['ra'])
                    rep_val_list = list(rep_df['ra'])
                    neut_val_list = list(neut_df['ra'])

                    dem_num = len(dem_df['worker_id'])
                    rep_num = len(rep_df['worker_id'])
                    neut_num = len(neut_df['worker_id'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l_2 += list(df_tmp['ra'])
                    tweet_avg_group_2[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_2[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_2[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_2[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_2[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_2[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    val_list_ra = list(df_tmp['ra'])

                    tweet_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_2[t_id] = np.mean(val_list)
                    tweet_med_2[t_id] = np.median(val_list)
                    tweet_var_2[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_2.append(np.mean(val_list))
                    tweet_med_l_2.append(np.median(val_list))
                    tweet_var_l_2.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_2[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg_2[t_id] = np.mean(vot_list)
                    tweet_vote_med_2[t_id] = np.median(vot_list)
                    tweet_vote_var_2[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_2.append(np.mean(vot_list))
                    tweet_vote_med_l_2.append(np.median(vot_list))
                    tweet_vote_var_l_2.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg_2[t_id] = np.mean(val_list)
                    tweet_dev_med_2[t_id] = np.median(val_list)
                    tweet_dev_var_2[t_id] = np.var(val_list)

                    tweet_dev_avg_l_2.append(np.mean(val_list))
                    tweet_dev_med_l_2.append(np.median(val_list))
                    tweet_dev_var_l_2.append(np.var(val_list))

                    tweet_abs_dev_avg_2[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_2[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_2[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_2.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_2.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_2.append(np.var(abs_var_err))


        # tweet_var_sort = sorted(tweet_var_2, key=tweet_var_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
        tweet_var_sort = sorted(tweet_abs_dev_avg_3, key=tweet_abs_dev_avg_3.get, reverse=True)

        outF = open(remotedir + 'tweet_all_dist_judgments.txt','w')
        count = 0
        wiki_n = "{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/"

        # outF.write('|| || Text || Lable ||' +
        #            'AMT MPB' + '||' + 'AMT incentive MPB' + '||' +
        #            ' AMT TPB ' + '||' + 'AMT incentive TPB' + '||' +
        #            'Calim\' Leaning || Leaning judg by diff party ||' +
        #            'Which Party benefit || Which Party benefit judgments by diff parties ||' +
        #            'Claim judgment by all AMT workers (incentive)||Claim judgment by all AMT workers ||' +
        #            'Claim judgment by all AMT workers (incentive) diff parties (density)||Claim judgment by all AMT workers diff parties(density) ||' +
        #            'Claim judgment by all AMT workers (incentive) diff parties(histogram)||Claim judgment by all AMT workers diff parties (histogram) ||' )
        #
        #
        #
        #
        # for t_id in tweet_var_sort:
        #     count+=1
        #     outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] + '||'+
        #                str(np.round(tweet_avg_3[t_id],4)) + '||' + str(np.round(tweet_avg_2[t_id],4)) + '||'+
        #                str(np.round(tweet_abs_dev_avg_3[t_id],4))+ '||' +str(np.round(tweet_abs_dev_avg_2[t_id],4)) + '||'+
        #                wiki_n + 'tweet_leaning/claims_leaning_all_' + str(t_id)+'.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_leaning/claims_leaning_diff_party_' + str(t_id)+'.png|alt text | width = \"500px\"}} ||' +
        #                wiki_n + 'tweet_leaning_ben/claims_leaning_ben_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' + wiki_n + 'tweet_leaning_ben/claims_leaning_ben_diff_party_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +
        #                wiki_n + 'tweet_judgment_incentive/claims_sp_incentive_judgment_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +
        #                wiki_n + 'tweet_judgment_incentive/claims_sp_2_diff_party_' + str(t_id) + '_density.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_diff_party_' + str(t_id) + '_density.png|alt text | width = \"500px\"}} ||'+
        #                wiki_n + 'tweet_judgment_incentive/claims_sp_2_judgment_diff_party_' + str(t_id) + '_hist.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_diff_party_' + str(t_id) + '_hist.png|alt text | width = \"500px\"}} ||')
        #
        #     outF.write('\n')

        outF.write('|| || Text || Lable ||' +
                   'AMT MPB' + '||' + 'AMT incentive MPB' + '||' +
                   ' AMT TPB ' + '||' + 'AMT incentive TPB' + '||' +
                   'Calim\' Leaning||' +
                   'Which Party benefit ||' +
                   'Claim judgment by all AMT workers (incentive) diff parties(normalized histogram)||Claim judgment by all AMT workers diff parties (normalized histogram) ||' +
                   'Claim judgment by all AMT workers (incentive) diff parties(histogram)||Claim judgment by all AMT workers diff parties (histogram) ||\n')

        for t_id in tweet_var_sort:
            count += 1
            outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] + '||' +
                       str(np.round(tweet_avg_3[t_id], 4)) + '||' + str(np.round(tweet_avg_2[t_id], 4)) + '||' +
                       str(np.round(tweet_abs_dev_avg_3[t_id], 4)) + '||' + str(
                np.round(tweet_abs_dev_avg_2[t_id], 4)) + '||' +
                       wiki_n + 'tweet_leaning/claims_leaning_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_leaning_ben/claims_leaning_ben_all_' + str(
                t_id) + '.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_judgment_incentive/claims_sp_incentive_diff_party_binary_normed_' + str(
                t_id) + '_hist.png|alt text | width = \"500px\"}} ||' + wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_diff_party_binary_normed_' + str(
                t_id) + '_hist.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_judgment_incentive/claims_sp_incentive_diff_party_binary_' + str(
                t_id) + '_hist.png|alt text | width = \"500px\"}} ||' + wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_diff_party_binary_' + str(
                t_id) + '_hist.png|alt text | width = \"500px\"}} ||')

            outF.write('\n')


    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_writing_wiki_workers_dist_judgments":

        for dataset in ['snopes_incentive']:  # , 'snopes_incentive','snopes_nonpol', 'politifact', 'snopes_2', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_3 = {}
            tweet_dev_med_3 = {}
            tweet_dev_var_3 = {}
            tweet_avg_3 = {}
            tweet_med_3 = {}
            tweet_var_3 = {}
            tweet_gt_var_3 = {}

            tweet_dev_avg_l_3 = []
            tweet_dev_med_l_3 = []
            tweet_dev_var_l_3 = []
            tweet_avg_l_3 = []
            tweet_med_l_3 = []
            tweet_var_l_3 = []
            tweet_gt_var_l_3 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_3 = {}
            tweet_abs_dev_med_3 = {}
            tweet_abs_dev_var_3 = {}

            tweet_abs_dev_avg_l_3 = []
            tweet_abs_dev_med_l_3 = []
            tweet_abs_dev_var_l_3 = []

            tweet_abs_dev_avg_rnd_3= {}
            tweet_dev_avg_rnd_3 = {}

            tweet_skew_3 = {}
            tweet_skew_l_3 = []

            tweet_vote_avg_med_var_3 = collections.defaultdict(list)
            tweet_vote_avg_3 = collections.defaultdict()
            tweet_vote_med_3 = collections.defaultdict()
            tweet_vote_var_3 = collections.defaultdict()

            tweet_avg_group_3 = collections.defaultdict()
            tweet_med_group_3 = collections.defaultdict()
            tweet_var_group_3 = collections.defaultdict()
            tweet_var_diff_group_3 = collections.defaultdict()

            tweet_kldiv_group_3 = collections.defaultdict()

            tweet_vote_avg_l_3 = []
            tweet_vote_med_l_3 = []
            tweet_vote_var_l_3 = []
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_3 = []
            worker_acc_true = collections.defaultdict(float)
            worker_acc_false = collections.defaultdict(float)
            worker_err_abs = collections.defaultdict()
            worker_err = collections.defaultdict()
            worker_judgment_dict = collections.defaultdict(list)
            worker_gull = collections.defaultdict()
            worker_cyn = collections.defaultdict()


            for ind in ind_l:
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_w_m = df_w[ind]
                df_w_m.loc[:,'err'] = df_w_m['worker_id'] * 0.0
                df_w_m.loc[:,'leaning'] = df_w_m['worker_id'] * 0.0
                df_w_m.loc[:,'cyn'] = df_w_m['worker_id'] * 0.0
                df_w_m.loc[:,'gull'] = df_w_m['worker_id'] * 0.0
                df_w_m.loc[:,'abs_err'] = df_w_m['worker_id'] * 0.0
                df_w_m.loc[:,'acc_true'] = df_w_m['worker_id'] * 0.0
                df_w_m.loc[:,'acc_false'] = df_w_m['worker_id'] * 0.0
                df_w_m.loc[:,'income_c'] = df_w_m['worker_id'] * 0.0

                df_m = df[ind].copy()

                groupby_ftr = 'worker_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
                #                          'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))
                count=0
                for w_id in grouped.groups.keys():
                    count+=1
                    print(count)
                    df_tmp = df_m[df_m['worker_id'] == w_id]
                    ind_t = df_tmp.index.tolist()[0]
                    ind_w = df_w_m[df_w_m['worker_id']==w_id].index.tolist()[0]
                    weights = []

                    pp=0; pf=0; tpp=0; tpf=0
                    err=0; err_abs=0;gull=0;cyn=0;
                    t_gull=0; t_cyn=0
                    for t_id in df_tmp['tweet_id']:

                        indexx = df_tmp[df_tmp['tweet_id']==t_id].index.tolist()[0]
                        pt_b = df_tmp['rel_v_b'][indexx]
                        # pt = np.mean(df_tmp['acc'])
                        gt_b = df_tmp['rel_gt_v'][indexx]
                        if gt_b > 0:
                            if pt_b > 0:
                                pp += 1
                            tpp += 1
                        if gt_b < 0:
                            if pt_b < 0:
                                pf += 1
                            tpf += 1
                        min_dist = 10

                        pt = df_tmp['rel_v'][indexx]
                        # pt = np.mean(df_tmp['acc'])
                        gt = df_tmp['rel_gt_v'][indexx]
                        err += (pt - gt)
                        if pt>gt:
                            gull += (pt - gt)
                            t_gull+=1
                        if gt>pt:
                            cyn += (gt - pt)
                            t_cyn+=1
                        err_abs += np.abs(pt-gt)
                        min_dist = 10

                    acc_true = pp / float(tpp)
                    acc_false = pf / float(tpf)
                    worker_acc_true[w_id] = acc_true
                    worker_acc_false[w_id] = acc_false
                    worker_err[w_id] = err/float(len(df_tmp))
                    worker_err_abs[w_id] = err_abs/float(len(df_tmp))
                    worker_gull[w_id] = gull/float(t_gull)
                    worker_cyn[w_id] = cyn/float(t_cyn)
                    worker_judgment_dict[w_id] = list(df_tmp['ra'].values)

                    df_w_m['cyn'][ind_w] = worker_cyn[w_id]
                    df_w_m['gull'][ind_w] = worker_gull[w_id]
                    df_w_m['acc_true'][ind_w] = acc_true
                    df_w_m['acc_false'][ind_w] = acc_false
                    df_w_m['err'][ind_w] = worker_err[w_id]
                    df_w_m['abs_err'][ind_w] = worker_err_abs[w_id]
                    df_w_m['leaning'][ind_w] = df_m['leaning'][ind_t]

                    df_w_m.loc[:, 'under 30000'] = df_w_m['worker_id'] * 0.0
                    df_w_m.loc[:, '30000-50000'] = df_w_m['worker_id'] * 0.0
                    df_w_m.loc[:, 'more than 50000'] = df_w_m['worker_id'] * 0.0

                    if df_w_m['income'][ind_w] == '100001ndash150000':
                        df_w_m['income_c'][ind_w] = 'more than 50000'

                    elif df_w_m['income'][ind_w] == '40001ndash50000':
                        df_w_m['income_c'][ind_w] ='30000-50000'
                    elif df_w_m['income'][ind_w] == '70001ndash100000':
                        df_w_m['income_c'][ind_w] ='more than 50000'
                    elif df_w_m['income'][ind_w] == '60001ndash70000':
                        df_w_m['income_c'][ind_w] ='more than 50000'
                    elif df_w_m['income'][ind_w] == '150001ormore':
                        df_w_m['income_c'][ind_w] ='more than 50000'
                    elif df_w_m['income'][ind_w] == '10000ndash20000':
                        df_w_m['income_c'][ind_w] ='under 30000'
                    elif df_w_m['income'][ind_w] == '50001ndash60000':
                        df_w_m['income_c'][ind_w] ='more than 50000'
                    elif df_w_m['income'][ind_w] == '20001ndash30000':
                        df_w_m['income_c'][ind_w] ='under 30000'
                    elif df_w_m['income'][ind_w] == 'under10000':
                        df_w_m['income_c'][ind_w] ='under 30000'
                    elif df_w_m['income'][ind_w] == '30001ndash40000':
                        df_w_m['income_c'][ind_w] ='30000-50000'
        # val_1 = 0;
                    # val_2 = 0;
                    # val_3 = 0;
                    # val_4 = 0;
                    # val_5 = 0;
                    # val_6 = 0;
                    # val_7 = 0;
                    # val_list_ra = df_tmp['ra']
                    #
                    # for mm in val_list_ra:
                    #     if mm == 1:
                    #         val_1 += 1
                    #     elif mm == 2:
                    #         val_2 += 1
                    #     elif mm == 3:
                    #         val_3 += 1
                    #     elif mm == 4:
                    #         val_4 += 1
                    #     elif mm == 5:
                    #         val_5 += 1
                    #     elif mm == 6:
                    #         val_6 += 1
                    #     elif mm == 7:
                    #         val_7 += 1
                    # width=0.3
                    # mplpl.figure()
                    # hist_list = [val_1/float(len(val_list_ra)), val_2/float(len(val_list_ra)), val_3/float(len(val_list_ra))
                    #              , val_4/float(len(val_list_ra)), val_5/float(len(val_list_ra)), val_6/float(len(val_list_ra))
                    #              , val_7/float(len(val_list_ra))]
                    # mplpl.bar([0,1,2,3,4,5,6], hist_list,width,  color='c', label='')
                    #
                    # df_rep = pd.DataFrame(np.array(val_list_ra), columns=['dist'])
                    # df_rep['dist'].plot(kind='kde', lw=6, color='c', label='')
                    #
                    #
                    # mplpl.legend(loc="upper left")
                    # # labels = ['Republican', 'Neutral', 'Democrat']
                    # labels = ['Confirm \nit to\nbe a false', '', 'Possibly\nfalse',
                    #           '','Possibly\ntrue', '', 'Confirm\nit to be\ntrue']
                    # x = range(0, 7)
                    # mplpl.xticks(x, labels)  # , rotation='90')
                    # mplpl.subplots_adjust(bottom=0.2)
                    # mplpl.xlabel('Worker\'s judgements', fontsize=18)
                    #
                    # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_all_' + str(t_id)
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_incentive/' + 'worker_sp_incentive_judgment_' + str(
                    #     w_id)
                    # # mplpl.savefig(pp + '.pdf', format='pdf')
                    # mplpl.savefig(pp + '.png', format='png')

        # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
        #                          'employment',
        #                          'income', 'leaning', 'race', 'marital_status','political_view']


        demographic_feat_list = ['leaning', 'gender', 'age', 'degree', 'marital_status', 'employment', 'income_c', 'income']

        col = ['r', 'g', 'b', 'c', 'y', 'orange', 'k', 'grey', 'm', 'salmon', 'indigo']
        mplpl.rc('legend', fontsize='small')
        for dem_feature in demographic_feat_list:
            count = 0
            mplpl.figure()
            diff_demog_f_set = set(df_w_m[dem_feature])
            if dem_feature=='leaning':
                diff_demog_f_set=[-1,0,1]
            if dem_feature=='income_c':
                print""
                # for dd_f in diff_demog_f_set:
                #     if dd_f==''
                # diff_demog_f_set=['under 30000', '30000-50000', 'more than 50000']

            for dem_f in diff_demog_f_set:
                df_tmp_f = df_w_m[df_w_m[dem_feature]==dem_f]
                if dem_feature=='income_c' and dem_f==0:
                    continue
                num_bins = len(df_tmp_f['cyn'])
                # counts, bin_edges = np.histogram(df_tmp_f['cyn'], bins=num_bins, normed=True)
                counts, bin_edges = np.histogram(df_tmp_f['gull'], bins=num_bins, normed=True)
                # counts, bin_edges = np.histogram(df_tmp_f['abs_err'], bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                if dem_f==-1:
                    label_t = 'Rep'
                elif dem_f==0:
                    label_t='Neut'
                elif dem_f==1:
                    label_t='Dem'
                elif dem_f=='under 30000':
                    label_t='under 30000'
                elif dem_f == '30000-50000':
                    label_t = '30000-50000'
                elif dem_f == 'more than 50000':
                    label_t = 'more than 50000'
                else:
                    label_t = modify_demographic(dem_f)
                if num_bins>5:
                    mplpl.plot(bin_edges[1:], ncdf, c=col[count],   lw=5, label=str(label_t)  + str(':')+ str(np.round(float(num_bins)/len(df_w[1]),2)))

                    count+=1

            mplpl.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3))
            mplpl.rc('legend', fontsize='small')
            #loc="upper left")
            mplpl.subplots_adjust(top=0.8)

            mplpl.grid()
            mplpl.xlabel('Woerker\'s Cynicallity', fontsize=18)
            # mplpl.xlabel('Woerker\'s Gullibility', fontsize=18)
            # mplpl.xlabel('Woerker\'s Absolute Error', fontsize=18)
            mplpl.ylabel('CDF', fontsize=18)
        #
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_all_' + str(t_id)
        #     pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_incentive_initial/' + 'worker_sp_incentive_cynicallity_' + str(dem_feature)
            pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_incentive_initial/' + 'worker_sp_incentive_gullibility_' + str(dem_feature)
        #     pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_incentive_initial/' + 'worker_sp_incentive_abs_err_' + str(dem_feature)
        #     pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_snopes2_initial/' + 'worker_sp_incentive_cynicallity_' + str(dem_feature)
        #     pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_snopes2_initial/' + 'worker_sp_incentive_gullibility_' + str(dem_feature)
            # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_snopes2_initial/' + 'worker_sp_incentive_abs_err_' + str(dem_feature)
            # mplpl.savefig(pp + '.pdf', format='pdf')
            mplpl.savefig(pp + '.png', format='png')

        exit()

        worker_acc_sort = sorted(worker_err_abs, key=worker_err_abs.get, reverse=True)

        outF = open(remotedir + 'workers_dist_judgments.txt', 'w')
        count = 0
        wiki_n = "{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/"

        outF.write('|| || worker id|| Absolute error|| Accuracy (True claims)|| Accuracy(False claims)|| Gullibility ||'
                   'Cynicality||Worker judgment || political_view|| gender||degree||age||employment||income||'
                   'marital_status||race||nationality||residence||')
        outF.write('\n')

        count = 0
        for w_id in worker_acc_sort:
            df_tmp = df_m[df_m['worker_id'] == w_id]
            ind_t = df_tmp.index.tolist()[0]
            count+=1
            outF.write('||' + str(count) + '||' + str(w_id) + '||' +
                       str(np.round(worker_err_abs[w_id],4)) + '||' + str(np.round(worker_acc_true[w_id],4)) + '||'+str(np.round(worker_acc_false[w_id],4)) + '||'+
                       str(np.round(worker_gull[w_id],4))+ '||' +str(np.round(worker_cyn[w_id],4)) + '||'+
                       wiki_n + 'worker_judgment_incentive/worker_sp_incentive_judgment_' + str(w_id)+'.png|alt text | width = \"500px\"}} ||' +
                       str(df_tmp['political_view'][ind_t]) + '||'+str(df_tmp['gender'][ind_t]) + '||'+str(df_tmp['degree'][ind_t]) + '||'+
                       str(df_tmp['age'][ind_t]) + '||' +str(df_tmp['employment'][ind_t]) + '||'+str(df_tmp['income'][ind_t]) + '||'+
                       str(df_tmp['marital_status'][ind_t]) + '||' +str(df_tmp['race'][ind_t]) + '||'+str(df_tmp['nationality'][ind_t]) + '||'+
                       str(df_tmp['residence'][ind_t]) + '||\n')
            # outF.write('\n')




        exit()
        for dataset in ['snopes_2']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_2 = {}
            tweet_dev_med_2 = {}
            tweet_dev_var_2 = {}
            tweet_avg_2 = {}
            tweet_med_2 = {}
            tweet_var_2 = {}
            tweet_gt_var_2 = {}

            tweet_dev_avg_l_2 = []
            tweet_dev_med_l_2 = []
            tweet_dev_var_l_2 = []
            tweet_avg_l_2 = []
            tweet_med_l_2 = []
            tweet_var_l_2 = []
            tweet_gt_var_l_2 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_2 = {}
            tweet_abs_dev_med_2 = {}
            tweet_abs_dev_var_2 = {}

            tweet_abs_dev_avg_l_2 = []
            tweet_abs_dev_med_l_2 = []
            tweet_abs_dev_var_l_2 = []

            tweet_abs_dev_avg_rnd_2 = {}
            tweet_dev_avg_rnd_2 = {}

            tweet_skew_2 = {}
            tweet_skew_l_2 = []

            tweet_vote_avg_med_var_2 = collections.defaultdict(list)
            tweet_vote_avg_2 = collections.defaultdict()
            tweet_vote_med_2 = collections.defaultdict()
            tweet_vote_var_2 = collections.defaultdict()

            tweet_avg_group_2 = collections.defaultdict()
            tweet_med_group_2 = collections.defaultdict()
            tweet_var_group_2 = collections.defaultdict()
            tweet_var_diff_group_2 = collections.defaultdict()

            tweet_kldiv_group_2 = collections.defaultdict()

            tweet_vote_avg_l_2 = []
            tweet_vote_med_l_2 = []
            tweet_vote_var_l_2 = []
            tweet_chi_group_2 = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_2 = []
            for ind in ind_l:
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()



                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    dem_val_list = list(dem_df['ra'])
                    rep_val_list = list(rep_df['ra'])
                    neut_val_list = list(neut_df['ra'])

                    dem_num = len(dem_df['worker_id'])
                    rep_num = len(rep_df['worker_id'])
                    neut_num = len(neut_df['worker_id'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l_2 += list(df_tmp['ra'])
                    tweet_avg_group_2[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_2[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_2[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_2[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_2[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_2[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    val_list_ra = list(df_tmp['ra'])

                    tweet_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_2[t_id] = np.mean(val_list)
                    tweet_med_2[t_id] = np.median(val_list)
                    tweet_var_2[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_2.append(np.mean(val_list))
                    tweet_med_l_2.append(np.median(val_list))
                    tweet_var_l_2.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_2[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg_2[t_id] = np.mean(vot_list)
                    tweet_vote_med_2[t_id] = np.median(vot_list)
                    tweet_vote_var_2[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_2.append(np.mean(vot_list))
                    tweet_vote_med_l_2.append(np.median(vot_list))
                    tweet_vote_var_l_2.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg_2[t_id] = np.mean(val_list)
                    tweet_dev_med_2[t_id] = np.median(val_list)
                    tweet_dev_var_2[t_id] = np.var(val_list)

                    tweet_dev_avg_l_2.append(np.mean(val_list))
                    tweet_dev_med_l_2.append(np.median(val_list))
                    tweet_dev_var_l_2.append(np.var(val_list))

                    tweet_abs_dev_avg_2[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_2[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_2[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_2.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_2.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_2.append(np.var(abs_var_err))


        # tweet_var_sort = sorted(tweet_var_2, key=tweet_var_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
        tweet_var_sort = sorted(tweet_abs_dev_avg_3, key=tweet_abs_dev_avg_3.get, reverse=True)

        outF = open(remotedir + 'tweet_all_dist_judgments.txt','w')
        count = 0
        wiki_n = "{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/"

        outF.write('|| || Text || Lable ||' +
                   'AMT MPB' + '||' + 'AMT incentive MPB' + '||' +
                   ' AMT TPB ' + '||' + 'AMT incentive TPB' + '||' +
                   'Calim\' Leaning || Leaning judg by diff party ||' +
                   'Which Party benefit || Which Party benefit judgments by diff parties ||' +
                   'Claim judgment by all AMT workers (incentive)||Claim judgment by all AMT workers ||' +
                   'Claim judgment by all AMT workers (incentive) diff parties (density)||Claim judgment by all AMT workers diff parties(density) ||' +
                   'Claim judgment by all AMT workers (incentive) diff parties(histogram)||Claim judgment by all AMT workers diff parties (histogram) ||' )




        for t_id in tweet_var_sort:
            count+=1
            outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] + '||'+
                       str(np.round(tweet_avg_3[t_id],4)) + '||' + str(np.round(tweet_avg_2[t_id],4)) + '||'+
                       str(np.round(tweet_abs_dev_avg_3[t_id],4))+ '||' +str(np.round(tweet_abs_dev_avg_2[t_id],4)) + '||'+
                       wiki_n + 'tweet_leaning/claims_leaning_all_' + str(t_id)+'.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_leaning/claims_leaning_diff_party_' + str(t_id)+'.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_leaning_ben/claims_leaning_ben_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' + wiki_n + 'tweet_leaning_ben/claims_leaning_ben_diff_party_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_judgment_incentive/claims_sp_incentive_judgment_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_judgment_incentive/claims_sp_2_diff_party_' + str(t_id) + '_density.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_diff_party_' + str(t_id) + '_density.png|alt text | width = \"500px\"}} ||'+
                       wiki_n + 'tweet_judgment_incentive/claims_sp_2_judgment_diff_party_' + str(t_id) + '_hist.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_diff_party_' + str(t_id) + '_hist.png|alt text | width = \"500px\"}} ||')

            outF.write('\n')

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_writing_wiki_workers_dist_judgments_diff_dataset":

        df = collections.defaultdict()
        df_w = collections.defaultdict()

        tweet_avg_med_var = collections.defaultdict()
        tweet_dev_avg_med_var = collections.defaultdict()
        tweet_dev_avg = collections.defaultdict()
        tweet_dev_med = collections.defaultdict()
        tweet_dev_var = collections.defaultdict()
        tweet_avg = collections.defaultdict()
        tweet_med = collections.defaultdict()
        tweet_var = collections.defaultdict()
        tweet_gt_var = collections.defaultdict()

        tweet_dev_avg_l = collections.defaultdict()
        tweet_dev_med_l = collections.defaultdict()
        tweet_dev_var_l = collections.defaultdict()
        tweet_avg_l = collections.defaultdict()
        tweet_med_l = collections.defaultdict()
        tweet_var_l = collections.defaultdict()
        tweet_gt_var_l = collections.defaultdict()
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = collections.defaultdict()
        tweet_abs_dev_med = collections.defaultdict()
        tweet_abs_dev_var = collections.defaultdict()

        tweet_abs_dev_avg_l = collections.defaultdict()
        tweet_abs_dev_med_l = collections.defaultdict()
        tweet_abs_dev_var_l = collections.defaultdict()

        tweet_abs_dev_avg_rnd = collections.defaultdict()
        tweet_dev_avg_rnd = collections.defaultdict()

        tweet_skew = collections.defaultdict()
        tweet_skew_l = collections.defaultdict()

        tweet_vote_avg_med_var = collections.defaultdict()
        tweet_vote_avg = collections.defaultdict()
        tweet_vote_med = collections.defaultdict()
        tweet_vote_var = collections.defaultdict()

        tweet_avg_group = collections.defaultdict()
        tweet_med_group = collections.defaultdict()
        tweet_var_group = collections.defaultdict()
        tweet_var_diff_group = collections.defaultdict()

        tweet_kldiv_group = collections.defaultdict()

        tweet_vote_avg_l = collections.defaultdict()
        tweet_vote_med_l = collections.defaultdict()
        tweet_vote_var_l = collections.defaultdict()
        tweet_chi_group = collections.defaultdict()
        tweet_chi_group = collections.defaultdict()
        tweet_chi_group = collections.defaultdict()
        tweet_skew = collections.defaultdict()
        ind_l = collections.defaultdict()
        ans_l = collections.defaultdict()
        worker_acc_true = collections.defaultdict()
        worker_acc_false = collections.defaultdict()
        worker_err_abs = collections.defaultdict()
        worker_err = collections.defaultdict()
        worker_judgment_dict = collections.defaultdict()
        worker_gull = collections.defaultdict()
        worker_cyn = collections.defaultdict()
        df_w_m = collections.defaultdict()
        for dataset in ['snopes_incentive', 'snopes_2','snopes', 'snopes_incentive_notimer', 'snopes_noincentive_timer']:#,'snopes_nonpol', 'politifact', 'snopes_2', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive'or dataset == 'snopes_2'or dataset == 'snopes_incentive_notimer'or dataset == 'snopes_noincentive_timer':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'


            elif dataset == 'snopes_incentive_notimer':
                data_n = 'sp_incentive_notimer'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive_notimer'


            elif dataset == 'snopes_noincentive_timer':

                data_n = 'sp_noincentive_timer'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_noincentive_timer'
            elif dataset == 'snopes_ssi':
                        data_n = 'sp_ssi'
                        data_addr = 'snopes'
                        ind_l = [1, 2, 3]
                        data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df[dataset] = collections.defaultdict()
            df_w[dataset] = collections.defaultdict()
            tweet_avg_med_var[dataset] = collections.defaultdict(list)
            tweet_dev_avg_med_var[dataset] = collections.defaultdict(list)
            tweet_dev_avg[dataset] = {}
            tweet_dev_med[dataset] = {}
            tweet_dev_var[dataset] = {}
            tweet_avg[dataset] = {}
            tweet_med[dataset] = {}
            tweet_var[dataset] = {}
            tweet_gt_var[dataset] = {}

            tweet_dev_avg_l[dataset] = []
            tweet_dev_med_l[dataset] = []
            tweet_dev_var_l[dataset] = []
            tweet_avg_l[dataset] = []
            tweet_med_l[dataset] = []
            tweet_var_l[dataset] = []
            tweet_gt_var_l[dataset] = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg[dataset] = {}
            tweet_abs_dev_med[dataset] = {}
            tweet_abs_dev_var[dataset] = {}

            tweet_abs_dev_avg_l[dataset] = []
            tweet_abs_dev_med_l[dataset] = []
            tweet_abs_dev_var_l[dataset] = []

            tweet_abs_dev_avg_rnd[dataset]= {}
            tweet_dev_avg_rnd[dataset] = {}

            tweet_skew[dataset] = {}
            tweet_skew_l[dataset] = []

            tweet_vote_avg_med_var[dataset] = collections.defaultdict(list)
            tweet_vote_avg[dataset] = collections.defaultdict()
            tweet_vote_med[dataset] = collections.defaultdict()
            tweet_vote_var[dataset] = collections.defaultdict()

            tweet_avg_group[dataset] = collections.defaultdict()
            tweet_med_group[dataset] = collections.defaultdict()
            tweet_var_group[dataset] = collections.defaultdict()
            tweet_var_diff_group[dataset] = collections.defaultdict()

            tweet_kldiv_group[dataset] = collections.defaultdict()

            tweet_vote_avg_l[dataset] = []
            tweet_vote_med_l[dataset] = []
            tweet_vote_var_l[dataset] = []
            tweet_chi_group[dataset] = {}
            tweet_chi_group[dataset] = {}
            tweet_chi_group[dataset] = {}
            tweet_skew[dataset] = {}
            ind_l = [1]
            ans_l[dataset] = []
            worker_acc_true[dataset] = collections.defaultdict(float)
            worker_acc_false[dataset] = collections.defaultdict(float)
            worker_err_abs[dataset] = collections.defaultdict()
            worker_err[dataset] = collections.defaultdict()
            worker_judgment_dict[dataset] = collections.defaultdict(list)
            worker_gull[dataset] = collections.defaultdict()
            worker_cyn[dataset] = collections.defaultdict()


            for ind in ind_l:
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_w_m[dataset] = df_w[ind]
                df_w_m[dataset].loc[:,'err'] = df_w_m[dataset]['worker_id'] * 0.0
                df_w_m[dataset].loc[:,'leaning'] = df_w_m[dataset]['worker_id'] * 0.0
                df_w_m[dataset].loc[:,'cyn'] = df_w_m[dataset]['worker_id'] * 0.0
                df_w_m[dataset].loc[:,'gull'] = df_w_m[dataset]['worker_id'] * 0.0
                df_w_m[dataset].loc[:,'abs_err'] = df_w_m[dataset]['worker_id'] * 0.0
                df_w_m[dataset].loc[:,'acc_true'] = df_w_m[dataset]['worker_id'] * 0.0
                df_w_m[dataset].loc[:,'acc_false'] = df_w_m[dataset]['worker_id'] * 0.0
                df_w_m[dataset].loc[:,'income_c'] = df_w_m[dataset]['worker_id'] * 0.0

                df_m = df[ind].copy()

                groupby_ftr = 'worker_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
                #                          'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))
                count=0
                for w_id in grouped.groups.keys():
                    count+=1
                    print(count)
                    df_tmp = df_m[df_m['worker_id'] == w_id]
                    ind_t = df_tmp.index.tolist()[0]
                    ind_w = df_w_m[dataset][df_w_m[dataset]['worker_id']==w_id].index.tolist()[0]
                    weights = []

                    pp=0; pf=0; tpp=0; tpf=0
                    err=0; err_abs=0;gull=0;cyn=0;
                    t_gull=0; t_cyn=0
                    for t_id in df_tmp['tweet_id']:

                        indexx = df_tmp[df_tmp['tweet_id']==t_id].index.tolist()[0]
                        pt_b = df_tmp['rel_v_b'][indexx]
                        # pt = np.mean(df_tmp['acc'])
                        gt_b = df_tmp['rel_gt_v'][indexx]
                        if gt_b > 0:
                            if pt_b > 0:
                                pp += 1
                            tpp += 1
                        if gt_b < 0:
                            if pt_b < 0:
                                pf += 1
                            tpf += 1
                        min_dist = 10

                        pt = df_tmp['rel_v'][indexx]
                        # pt = np.mean(df_tmp['acc'])
                        gt = df_tmp['rel_gt_v'][indexx]
                        err += (pt - gt)
                        if pt>gt:
                            gull += (pt - gt)
                            t_gull+=1
                        if gt>pt:
                            cyn += (gt - pt)
                            t_cyn+=1
                        err_abs += np.abs(pt-gt)
                        min_dist = 10

                    acc_true = pp / float(tpp)
                    acc_false = pf / float(tpf)
                    worker_acc_true[dataset][w_id] = acc_true
                    worker_acc_false[dataset][w_id] = acc_false
                    worker_err[dataset][w_id] = err/float(len(df_tmp))
                    worker_err_abs[dataset][w_id] = err_abs/float(len(df_tmp))
                    worker_gull[dataset][w_id] = gull/float(t_gull)
                    worker_cyn[dataset][w_id] = cyn/float(t_cyn)
                    worker_judgment_dict[dataset][w_id] = list(df_tmp['ra'].values)

                    df_w_m[dataset]['cyn'][ind_w] = worker_cyn[dataset][w_id]
                    df_w_m[dataset]['gull'][ind_w] = worker_gull[dataset][w_id]
                    df_w_m[dataset]['acc_true'][ind_w] = acc_true
                    df_w_m[dataset]['acc_false'][ind_w] = acc_false
                    df_w_m[dataset]['err'][ind_w] = worker_err[dataset][w_id]
                    df_w_m[dataset]['abs_err'][ind_w] = worker_err_abs[dataset][w_id]
                    df_w_m[dataset]['leaning'][ind_w] = df_m['leaning'][ind_t]

                    df_w_m[dataset].loc[:, 'under 30000'] = df_w_m[dataset]['worker_id'] * 0.0
                    df_w_m[dataset].loc[:, '30000-50000'] = df_w_m[dataset]['worker_id'] * 0.0
                    df_w_m[dataset].loc[:, 'more than 50000'] = df_w_m[dataset]['worker_id'] * 0.0

                    if df_w_m[dataset]['income'][ind_w] == '100001ndash150000':
                        df_w_m[dataset]['income_c'][ind_w] = 'more than 50000'

                    elif df_w_m[dataset]['income'][ind_w] == '40001ndash50000':
                        df_w_m[dataset]['income_c'][ind_w] ='30000-50000'
                    elif df_w_m[dataset]['income'][ind_w] == '70001ndash100000':
                        df_w_m[dataset]['income_c'][ind_w] ='more than 50000'
                    elif df_w_m[dataset]['income'][ind_w] == '60001ndash70000':
                        df_w_m[dataset]['income_c'][ind_w] ='more than 50000'
                    elif df_w_m[dataset]['income'][ind_w] == '150001ormore':
                        df_w_m[dataset]['income_c'][ind_w] ='more than 50000'
                    elif df_w_m[dataset]['income'][ind_w] == '10000ndash20000':
                        df_w_m[dataset]['income_c'][ind_w] ='under 30000'
                    elif df_w_m[dataset]['income'][ind_w] == '50001ndash60000':
                        df_w_m[dataset]['income_c'][ind_w] ='more than 50000'
                    elif df_w_m[dataset]['income'][ind_w] == '20001ndash30000':
                        df_w_m[dataset]['income_c'][ind_w] ='under 30000'
                    elif df_w_m[dataset]['income'][ind_w] == 'under10000':
                        df_w_m[dataset]['income_c'][ind_w] ='under 30000'
                    elif df_w_m[dataset]['income'][ind_w] == '30001ndash40000':
                        df_w_m[dataset]['income_c'][ind_w] ='30000-50000'
        # val_1 = 0;
                    # val_2 = 0;
                    # val_3 = 0;
                    # val_4 = 0;
                    # val_5 = 0;
                    # val_6 = 0;
                    # val_7 = 0;
                    # val_list_ra = df_tmp['ra']
                    #
                    # for mm in val_list_ra:
                    #     if mm == 1:
                    #         val_1 += 1
                    #     elif mm == 2:
                    #         val_2 += 1
                    #     elif mm == 3:
                    #         val_3 += 1
                    #     elif mm == 4:
                    #         val_4 += 1
                    #     elif mm == 5:
                    #         val_5 += 1
                    #     elif mm == 6:
                    #         val_6 += 1
                    #     elif mm == 7:
                    #         val_7 += 1
                    # width=0.3
                    # mplpl.figure()
                    # hist_list = [val_1/float(len(val_list_ra)), val_2/float(len(val_list_ra)), val_3/float(len(val_list_ra))
                    #              , val_4/float(len(val_list_ra)), val_5/float(len(val_list_ra)), val_6/float(len(val_list_ra))
                    #              , val_7/float(len(val_list_ra))]
                    # mplpl.bar([0,1,2,3,4,5,6], hist_list,width,  color='c', label='')
                    #
                    # df_rep = pd.DataFrame(np.array(val_list_ra), columns=['dist'])
                    # df_rep['dist'].plot(kind='kde', lw=6, color='c', label='')
                    #
                    #
                    # mplpl.legend(loc="upper left")
                    # # labels = ['Republican', 'Neutral', 'Democrat']
                    # labels = ['Confirm \nit to\nbe a false', '', 'Possibly\nfalse',
                    #           '','Possibly\ntrue', '', 'Confirm\nit to be\ntrue']
                    # x = range(0, 7)
                    # mplpl.xticks(x, labels)  # , rotation='90')
                    # mplpl.subplots_adjust(bottom=0.2)
                    # mplpl.xlabel('Worker\'s judgements', fontsize=18)
                    #
                    # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_all_' + str(t_id)
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_incentive/' + 'worker_sp_incentive_judgment_' + str(
                    #     w_id)
                    # # mplpl.savefig(pp + '.pdf', format='pdf')
                    # mplpl.savefig(pp + '.png', format='png')

        # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
        #                          'employment',
        #                          'income', 'leaning', 'race', 'marital_status','political_view']


        demographic_feat_list = ['leaning', 'gender', 'age', 'degree', 'marital_status', 'employment', 'income_c', 'income']

        # col = ['r', 'g', 'b', 'c', 'y', 'orange', 'k', 'grey', 'm', 'salmon', 'indigo']
        col = ['g', 'k', 'salmon', 'grey', 'indigo']
        mplpl.rc('legend', fontsize='small')
        m_dict = {}
        for dem_feature in [0]:#demographic_feat_list:
            # print('==== ' + dem_feature + ' ====')
            count = 0
            # diff_demog_f_set = set(df_w_m['snopes'][dem_feature])
            # if dem_feature=='leaning':
            #     diff_demog_f_set=[-1,0,1]
            # if dem_feature=='income_c':
            #     print""

                # for dd_f in diff_demog_f_set:
                #     if dd_f==''
                # diff_demog_f_set=['under 30000', '30000-50000', 'more than 50000']

            for dem_f in [0]:#diff_demog_f_set:
                mplpl.figure()

                count=0
                for dataset in ['snopes_incentive', 'snopes_2','snopes','snopes_incentive_notimer','snopes_noincentive_timer']:
                    m_dict[dataset] = {}
                    # ,'snopes_nonpol', 'politifact', 'snopes_2', 'politifact', 'snopes', 'mia', 'politifact']:
                    # df_tmp_f = df_w_m[dataset][df_w_m[dataset][dem_feature]==dem_f]
                    # df_tmp_f = df_w_m[dataset][df_w_m[dataset][dem_feature]==dem_f]
                    df_tmp_f = df_w_m[dataset]
                    # if dem_feature=='income_c' and dem_f==0:
                    #     continue
                    num_bins = len(df_tmp_f['gull'])
                    if num_bins<1:
                        continue
                    # counts, bin_edges = np.histogram(df_tmp_f['cyn'], bins=num_bins, normed=True)
                    counts, bin_edges = np.histogram(df_tmp_f['gull'], bins=num_bins, normed=True)
                    # counts, bin_edges = np.histogram(df_tmp_f['abs_err'], bins=num_bins, normed=True)
                    for w_idm in df_tmp_f['worker_id']:
                        m_dict[dataset][w_idm]=df_tmp_f['cyn']
                    cdf = np.cumsum(counts)
                    scale = 1.0 / cdf[-1]
                    ncdf = scale * cdf
                    # if dem_f==-1:
                    #     label_t = 'Rep'
                    # elif dem_f==0:
                    #     label_t='Neut'
                    # elif dem_f==1:
                    #     label_t='Dem'
                    # elif dem_f=='under 30000':
                    #     label_t='under 30000'
                    # elif dem_f == '30000-50000':
                    #     label_t = '30000-50000'
                    # elif dem_f == 'more than 50000':
                    #     label_t = 'more than 50000'
                    # else:
                    #     label_t = modify_demographic(dem_f)
                    if num_bins>5:
                        mplpl.plot(bin_edges[1:], ncdf, c=col[count],   lw=5, label=str(dataset))

                    count+=1

                    mplpl.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3))
                    mplpl.rc('legend', fontsize='small')
                    #loc="upper left")
                    mplpl.subplots_adjust(top=0.8)

                    mplpl.grid()
                    # mplpl.xlabel('Woerker\'s Cynicallity', fontsize=18)
                    # mplpl.title( dem_feature + ' : '+ str(dem_f), fontsize=18)
                    mplpl.title( 'All data set', fontsize=18)
                    mplpl.xlabel('Woerker\'s Gullibility', fontsize=18)
                    # mplpl.xlabel('Woerker\'s Absolute Error', fontsize=18)
                    mplpl.ylabel('CDF', fontsize=18)
                #
                # print('' + str(np.round(sklearn.metrics.mutual_info_score(m_dict['snopes_incentive'].values(), m_dict['snopes'].values())), 4))
                # print('' + str(np.round(sklearn.metrics.mutual_info_score(m_dict['snopes_incentive'].values, m_dict['snopes_2'].values)), 4))
                # print('' + str(np.round(sklearn.metrics.mutual_info_score(m_dict['snopes'].values, m_dict['snopes_2'].values)), 4))
                # # entropy_estimator.sim_cosine(n)
                #
                # print('' + str(np.round(entropy_estimator.sim_cosine(m_dict['snopes_incentive'].values, m_dict['snopes'].values)), 4))
                # print('' + str(np.round(entropy_estimator.sim_cosine(m_dict['snopes_incentive'].values, m_dict['snopes_2'].values)), 4))
                # print('' + str(np.round(entropy_estimator.sim_cosine(m_dict['snopes'].values, m_dict['snopes_2'].values)), 4))
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_all_' + str(t_id)
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_initial/' + 'worker_sp_cynicallity' +str(dem_feature) + ':'+str(dem_f)
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_initial/' + 'worker_sp_gullibility_'+ str(dem_feature) + ':'+str(dem_f)
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_initial/' + 'worker_sp_abs_err_' + str(dem_feature)+ ':'+str(dem_f)
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_initial/' + 'worker_sp_cynicallity_all'# + str(dem_feature)
                pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_initial/' + 'worker_sp_gullibility_all'# + str(dem_feature)
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/worker_judgment_initial/' + 'worker_sp_abs_err_all'# + str(dem_feature)
                # mplpl.savefig(pp + '.pdf', format='pdf')
                mplpl.savefig(pp + '.png', format='png')
                # print('||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/worker_judgment_initial/worker_sp_abs_err_'+ str(dem_feature)+ ':'+str(dem_f)+'.png|alt text | width = "500px"}}||')
                # print(
                # '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/worker_judgment_initial/worker_sp_gullibility_' + str(
                #     dem_feature) + ':' + str(dem_f) + '.png|alt text | width = "500px"}}||')
                print(
                '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/worker_judgment_initial/worker_sp_cynicallity_' + str(
                    dem_feature) + ':' + str(dem_f) + '.png|alt text | width = "500px"}}||')

        exit()

        worker_acc_sort = sorted(worker_err_abs, key=worker_err_abs.get, reverse=True)

        outF = open(remotedir + 'workers_dist_judgments.txt', 'w')
        count = 0
        wiki_n = "{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/"

        outF.write('|| || worker id|| Absolute error|| Accuracy (True claims)|| Accuracy(False claims)|| Gullibility ||'
                   'Cynicality||Worker judgment || political_view|| gender||degree||age||employment||income||'
                   'marital_status||race||nationality||residence||')
        outF.write('\n')

        count = 0
        for w_id in worker_acc_sort:
            df_tmp = df_m[df_m['worker_id'] == w_id]
            ind_t = df_tmp.index.tolist()[0]
            count+=1
            outF.write('||' + str(count) + '||' + str(w_id) + '||' +
                       str(np.round(worker_err_abs[w_id],4)) + '||' + str(np.round(worker_acc_true[w_id],4)) + '||'+str(np.round(worker_acc_false[w_id],4)) + '||'+
                       str(np.round(worker_gull[w_id],4))+ '||' +str(np.round(worker_cyn[w_id],4)) + '||'+
                       wiki_n + 'worker_judgment_incentive/worker_sp_incentive_judgment_' + str(w_id)+'.png|alt text | width = \"500px\"}} ||' +
                       str(df_tmp['political_view'][ind_t]) + '||'+str(df_tmp['gender'][ind_t]) + '||'+str(df_tmp['degree'][ind_t]) + '||'+
                       str(df_tmp['age'][ind_t]) + '||' +str(df_tmp['employment'][ind_t]) + '||'+str(df_tmp['income'][ind_t]) + '||'+
                       str(df_tmp['marital_status'][ind_t]) + '||' +str(df_tmp['race'][ind_t]) + '||'+str(df_tmp['nationality'][ind_t]) + '||'+
                       str(df_tmp['residence'][ind_t]) + '||\n')
            # outF.write('\n')




        exit()
        for dataset in ['snopes_2']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            if dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            if dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_med_var_2 = collections.defaultdict(list)
            tweet_dev_avg_2 = {}
            tweet_dev_med_2 = {}
            tweet_dev_var_2 = {}
            tweet_avg_2 = {}
            tweet_med_2 = {}
            tweet_var_2 = {}
            tweet_gt_var_2 = {}

            tweet_dev_avg_l_2 = []
            tweet_dev_med_l_2 = []
            tweet_dev_var_l_2 = []
            tweet_avg_l_2 = []
            tweet_med_l_2 = []
            tweet_var_l_2 = []
            tweet_gt_var_l_2 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_2 = {}
            tweet_abs_dev_med_2 = {}
            tweet_abs_dev_var_2 = {}

            tweet_abs_dev_avg_l_2 = []
            tweet_abs_dev_med_l_2 = []
            tweet_abs_dev_var_l_2 = []

            tweet_abs_dev_avg_rnd_2 = {}
            tweet_dev_avg_rnd_2 = {}

            tweet_skew_2 = {}
            tweet_skew_l_2 = []

            tweet_vote_avg_med_var_2 = collections.defaultdict(list)
            tweet_vote_avg_2 = collections.defaultdict()
            tweet_vote_med_2 = collections.defaultdict()
            tweet_vote_var_2 = collections.defaultdict()

            tweet_avg_group_2 = collections.defaultdict()
            tweet_med_group_2 = collections.defaultdict()
            tweet_var_group_2 = collections.defaultdict()
            tweet_var_diff_group_2 = collections.defaultdict()

            tweet_kldiv_group_2 = collections.defaultdict()

            tweet_vote_avg_l_2 = []
            tweet_vote_med_l_2 = []
            tweet_vote_var_l_2 = []
            tweet_chi_group_2 = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_2 = []
            for ind in ind_l:
                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()



                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['rel_v'])
                    rep_val_list = list(rep_df['rel_v'])
                    neut_val_list = list(neut_df['rel_v'])

                    dem_val_list = list(dem_df['ra'])
                    rep_val_list = list(rep_df['ra'])
                    neut_val_list = list(neut_df['ra'])

                    dem_num = len(dem_df['worker_id'])
                    rep_num = len(rep_df['worker_id'])
                    neut_num = len(neut_df['worker_id'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    ans_l_2 += list(df_tmp['ra'])
                    tweet_avg_group_2[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_2[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_2[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_2[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_2[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_2[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    val_list_ra = list(df_tmp['ra'])

                    tweet_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_2[t_id] = np.mean(val_list)
                    tweet_med_2[t_id] = np.median(val_list)
                    tweet_var_2[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_2.append(np.mean(val_list))
                    tweet_med_l_2.append(np.median(val_list))
                    tweet_var_l_2.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_2[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
                    tweet_vote_avg_2[t_id] = np.mean(vot_list)
                    tweet_vote_med_2[t_id] = np.median(vot_list)
                    tweet_vote_var_2[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_2.append(np.mean(vot_list))
                    tweet_vote_med_l_2.append(np.median(vot_list))
                    tweet_vote_var_l_2.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_dev_avg_2[t_id] = np.mean(val_list)
                    tweet_dev_med_2[t_id] = np.median(val_list)
                    tweet_dev_var_2[t_id] = np.var(val_list)

                    tweet_dev_avg_l_2.append(np.mean(val_list))
                    tweet_dev_med_l_2.append(np.median(val_list))
                    tweet_dev_var_l_2.append(np.var(val_list))

                    tweet_abs_dev_avg_2[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_2[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_2[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_2.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_2.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_2.append(np.var(abs_var_err))


        # tweet_var_sort = sorted(tweet_var_2, key=tweet_var_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
        tweet_var_sort = sorted(tweet_abs_dev_avg_3, key=tweet_abs_dev_avg_3.get, reverse=True)

        outF = open(remotedir + 'tweet_all_dist_judgments.txt','w')
        count = 0
        wiki_n = "{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig/"

        outF.write('|| || Text || Lable ||' +
                   'AMT MPB' + '||' + 'AMT incentive MPB' + '||' +
                   ' AMT TPB ' + '||' + 'AMT incentive TPB' + '||' +
                   'Calim\' Leaning || Leaning judg by diff party ||' +
                   'Which Party benefit || Which Party benefit judgments by diff parties ||' +
                   'Claim judgment by all AMT workers (incentive)||Claim judgment by all AMT workers ||' +
                   'Claim judgment by all AMT workers (incentive) diff parties (density)||Claim judgment by all AMT workers diff parties(density) ||' +
                   'Claim judgment by all AMT workers (incentive) diff parties(histogram)||Claim judgment by all AMT workers diff parties (histogram) ||' )




        for t_id in tweet_var_sort:
            count+=1
            outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] + '||'+
                       str(np.round(tweet_avg_3[t_id],4)) + '||' + str(np.round(tweet_avg_2[t_id],4)) + '||'+
                       str(np.round(tweet_abs_dev_avg_3[t_id],4))+ '||' +str(np.round(tweet_abs_dev_avg_2[t_id],4)) + '||'+
                       wiki_n + 'tweet_leaning/claims_leaning_all_' + str(t_id)+'.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_leaning/claims_leaning_diff_party_' + str(t_id)+'.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_leaning_ben/claims_leaning_ben_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' + wiki_n + 'tweet_leaning_ben/claims_leaning_ben_diff_party_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_judgment_incentive/claims_sp_incentive_judgment_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_all_' + str(t_id) + '.png|alt text | width = \"500px\"}} ||' +
                       wiki_n + 'tweet_judgment_incentive/claims_sp_2_diff_party_' + str(t_id) + '_density.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_diff_party_' + str(t_id) + '_density.png|alt text | width = \"500px\"}} ||'+
                       wiki_n + 'tweet_judgment_incentive/claims_sp_2_judgment_diff_party_' + str(t_id) + '_hist.png|alt text | width = \"500px\"}} ||' +wiki_n + 'tweet_judgment_amt_sp_2/claims_sp_2_judgment_diff_party_' + str(t_id) + '_hist.png|alt text | width = \"500px\"}} ||')

            outF.write('\n')

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_dist_judgement":

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []


        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        # for dataset in ['snopes_ssi']:#, 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
        #     if dataset == 'snopes' or dataset == 'snopes_ssi':
        #         claims_list = []
        #         remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
        #         # remotedir=''
        #         inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
        #         news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
        #         news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']
        #
        #         sample_tweets_exp1 = []
        #
        #         tweet_txt_dict = {}
        #         tweet_date_dict = {}
        #         tweet_lable_dict = {}
        #         tweet_publisher_dict = {}
        #         print(inp_all)
        #
        #         for i in range(0, 5):
        #             df_cat = news_cat_list[i]
        #             df_cat_f = news_cat_list_f[i]
        #             # remotedir='snopes/'
        #             inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
        #             cat_count = 0
        #             for line in inF:
        #                 claims_list.append(line)
        #                 cat_count += 1
        #
        #         for line in claims_list:
        #             line_splt = line.split('<<||>>')
        #             publisher_name = int(line_splt[2])
        #             tweet_txt = line_splt[3]
        #             tweet_id = publisher_name
        #             cat_lable = line_splt[4]
        #             dat = line_splt[5]
        #             dt_splt = dat.split(' ')[0].split('-')
        #             m_day = int(dt_splt[2])
        #             m_month = int(dt_splt[1])
        #             m_year = int(dt_splt[0])
        #             m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
        #             tweet_txt_dict[tweet_id] = tweet_txt
        #             tweet_date_dict[tweet_id] = m_date
        #             tweet_lable_dict[tweet_id] = cat_lable
        #             # outF = open(remotedir + 'table_out.txt', 'w')
        #     if dataset == 'snopes':
        #         data_n = 'sp'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         data_name = 'Snopes'
        #     if dataset == 'snopes_ssi':
        #         data_n = 'sp_ssi'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         data_name = 'Snopes_ssi'
        #     if dataset == 'snopes_nonpol':
        #         data_n = 'sp_nonpol'
        #         data_addr = 'snopes'
        #         ind_l = [1]
        #         data_name = 'Snopes_nonpol'
        #     elif dataset == 'politifact':
        #         data_addr = 'politifact/fig/'
        #         data_n = 'pf'
        #         ind_l = [1, 2, 3]
        #         data_name = 'PolitiFact'
        #     elif dataset == 'mia':
        #         data_addr = 'mia/fig/'
        #
        #         data_n = 'mia'
        #         ind_l = [1]
        #         data_name = 'Rumors/Non-Rumors'
        #
        #     df = collections.defaultdict()
        #     df_w = collections.defaultdict()
        #     tweet_avg_med_var = collections.defaultdict(list)
        #     tweet_dev_avg_med_var = collections.defaultdict(list)
        #     tweet_dev_avg = {}
        #     tweet_dev_med = {}
        #     tweet_dev_var = {}
        #     tweet_avg = {}
        #     tweet_med = {}
        #     tweet_var = {}
        #     tweet_gt_var = {}
        #
        #     tweet_dev_avg_l = []
        #     tweet_dev_med_l = []
        #     tweet_dev_var_l = []
        #     tweet_avg_l = []
        #     tweet_med_l = []
        #     tweet_var_l = []
        #     tweet_gt_var_l = []
        #     avg_susc = 0
        #     avg_gull = 0
        #     avg_cyn = 0
        #
        #     tweet_abs_dev_avg = {}
        #     tweet_abs_dev_med = {}
        #     tweet_abs_dev_var = {}
        #
        #     tweet_abs_dev_avg_l = []
        #     tweet_abs_dev_med_l = []
        #     tweet_abs_dev_var_l = []
        #
        #     tweet_abs_dev_avg_rnd = {}
        #     tweet_dev_avg_rnd = {}
        #
        #     tweet_skew = {}
        #     tweet_skew_l = []
        #
        #     tweet_vote_avg_med_var = collections.defaultdict(list)
        #     tweet_vote_avg = collections.defaultdict()
        #     tweet_vote_med = collections.defaultdict()
        #     tweet_vote_var = collections.defaultdict()
        #
        #     tweet_avg_group = collections.defaultdict()
        #     tweet_med_group = collections.defaultdict()
        #     tweet_var_group = collections.defaultdict()
        #     tweet_var_diff_group = collections.defaultdict()
        #
        #     tweet_kldiv_group = collections.defaultdict()
        #
        #     tweet_vote_avg_l = []
        #     tweet_vote_med_l = []
        #     tweet_vote_var_l = []
        #     tweet_chi_group = {}
        #     tweet_chi_group_1 = {}
        #     tweet_chi_group_2 = {}
        #     tweet_skew = {}
        #     ind_l = [1]
        #     ans_l = []
        #     for ind in ind_l:
        #         if balance_f == 'balanced':
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
        #         else:
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_new.csv'
        #             # inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
        #         inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
        #         df[ind] = pd.read_csv(inp1, sep="\t")
        #         # df_w[ind] = pd.read_csv(inp1_w, sep="\t")
        #
        #         df_m = df[ind].copy()
        #
        #
        #
        #         df_modify_1 = df_m[df_m['tweet_id']==2001]
        #         df_modify_2 = df_m[df_m['tweet_id']==2002]
        #         worker_id_1 = df_modify_1[df_modify_1['ra']>4]['worker_id']
        #         worker_id_2 = df_modify_2[df_modify_2['ra']<4]['worker_id']
        #         df_modify = df_m[df_m['worker_id'].isin(set(worker_id_1).intersection(set(worker_id_2)))]
        #         # df_modify = df_m[df_m['tweet_id'].isin([2001, 2002])]
        #         #
        #         # df_m = df_modify.copy()
        #         df_m = df_m[df_m['tweet_id']!=2001]
        #         df_m = df_m[df_m['tweet_id']!=2002]
        #         groupby_ftr = 'tweet_id'
        #         grouped = df_m.groupby(groupby_ftr, sort=False)
        #         grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()
        #
        #         # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment',
        #         #                          'income', 'political_view', 'race', 'marital_status']
        #         # for demographic_feat in demographic_feat_list:
        #         #     print('--------------------' + demographic_feat + '--------------------')
        #         #     for lean_f in set(df_m[demographic_feat]):
        #         #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
        #         #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))
        #
        #         for t_id in grouped.groups.keys():
        #             df_tmp = df_m[df_m['tweet_id'] == t_id]
        #             ind_t = df_tmp.index.tolist()[0]
        #             weights = []
        #
        #             dem_df = df_tmp[df_tmp['leaning'] == 1]
        #             rep_df = df_tmp[df_tmp['leaning'] == -1]
        #             neut_df = df_tmp[df_tmp['leaning'] == 0]
        #             dem_val_list = list(dem_df['rel_v'])
        #             rep_val_list = list(rep_df['rel_v'])
        #             neut_val_list = list(neut_df['rel_v'])
        #             # df_tmp = neut_df.copy()
        #
        #             val_list = list(df_tmp['rel_v'])
        #             ans_l += list(df_tmp['ra'])
        #
        #
        #             tweet_avg_group[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
        #             tweet_med_group[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
        #             tweet_var_diff_group[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
        #             tweet_var_group[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
        #             tweet_kldiv_group[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)
        #
        #
        #             # tweet_skew[t_id] = scipy.stats.skew(val_list)
        #             tweet_skew[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
        #             # tweet_skew_l.append(tweet_skew[t_id])
        #
        #             weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
        #             # val_list = list(df_tmp['rel_v_b'])
        #             val_list = list(df_tmp['rel_v'])
        #             tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
        #             tweet_avg[t_id] = np.mean(val_list)
        #             tweet_med[t_id] = np.median(val_list)
        #             tweet_var[t_id] = np.var(val_list)
        #             # tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]
        #
        #             tweet_avg_l.append(np.mean(val_list))
        #             tweet_med_l.append(np.median(val_list))
        #             tweet_var_l.append(np.var(val_list))
        #             # tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])
        #
        #             vot_list = []
        #             vot_list_tmp = list(df_tmp['vote'])
        #             # vot_list_tmp = []
        #
        #             for vot in vot_list_tmp:
        #                 if vot < 0:
        #                     vot_list.append(vot)
        #             tweet_vote_avg_med_var[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
        #             tweet_vote_avg[t_id] = np.mean(vot_list)
        #             tweet_vote_med[t_id] = np.median(vot_list)
        #             tweet_vote_var[t_id] = np.var(vot_list)
        #
        #             tweet_vote_avg_l.append(np.mean(vot_list))
        #             tweet_vote_med_l.append(np.median(vot_list))
        #             tweet_vote_var_l.append(np.var(vot_list))
        #
        #
        #
        #
        #             # val_list = list(df_tmp['susc'])
        #             # val_list = list(df_tmp['err_b'])
        #             val_list = list(df_tmp['err'])
        #             abs_var_err = [np.abs(x) for x in val_list]
        #             tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
        #             tweet_dev_avg[t_id] = np.mean(val_list)
        #             tweet_dev_med[t_id] = np.median(val_list)
        #             tweet_dev_var[t_id] = np.var(val_list)
        #
        #             tweet_dev_avg_l.append(np.mean(val_list))
        #             tweet_dev_med_l.append(np.median(val_list))
        #             tweet_dev_var_l.append(np.var(val_list))
        #
        #             tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
        #             tweet_abs_dev_med[t_id] = np.median(abs_var_err)
        #             tweet_abs_dev_var[t_id] = np.var(abs_var_err)
        #
        #             # tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
        #             # tweet_abs_dev_med_l.append(np.median(abs_var_err))
        #             # tweet_abs_dev_var_l.append(np.var(abs_var_err))
        # exit()
        print('')
        # for dataset in ['snopes_2']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
        #     if dataset == 'snopes' or dataset == 'snopes_2':
        #         claims_list = []
        #         remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
        #         # remotedir=''
        #         inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
        #         news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
        #         news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']
        #
        #         sample_tweets_exp1 = []
        #
        #         tweet_txt_dict = {}
        #         tweet_date_dict = {}
        #         tweet_lable_dict = {}
        #         tweet_publisher_dict = {}
        #         print(inp_all)
        #
        #         for i in range(0, 5):
        #             df_cat = news_cat_list[i]
        #             df_cat_f = news_cat_list_f[i]
        #             inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
        #             cat_count = 0
        #             for line in inF:
        #                 claims_list.append(line)
        #                 cat_count += 1
        #
        #         for line in claims_list:
        #             line_splt = line.split('<<||>>')
        #             publisher_name = int(line_splt[2])
        #             tweet_txt = line_splt[3]
        #             tweet_id = publisher_name
        #             cat_lable = line_splt[4]
        #             dat = line_splt[5]
        #             dt_splt = dat.split(' ')[0].split('-')
        #             m_day = int(dt_splt[2])
        #             m_month = int(dt_splt[1])
        #             m_year = int(dt_splt[0])
        #             m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
        #             tweet_txt_dict[tweet_id] = tweet_txt
        #             tweet_date_dict[tweet_id] = m_date
        #             tweet_lable_dict[tweet_id] = cat_lable
        #             # outF = open(remotedir + 'table_out.txt', 'w')
        #     if dataset == 'snopes_2':
        #         data_n = 'sp_2'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_2'
        #     if dataset == 'snopes_ssi':
        #         data_n = 'sp_ssi'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         data_name = 'Snopes_ssi'
        #     if dataset == 'snopes_nonpol':
        #         data_n = 'sp_nonpol'
        #         data_addr = 'snopes'
        #         ind_l = [1]
        #         data_name = 'Snopes_nonpol'
        #     elif dataset == 'politifact':
        #         data_addr = 'politifact/fig/'
        #         data_n = 'pf'
        #         ind_l = [1, 2, 3]
        #         data_name = 'PolitiFact'
        #     elif dataset == 'mia':
        #         data_addr = 'mia/fig/'
        #
        #         data_n = 'mia'
        #         ind_l = [1]
        #         data_name = 'Rumors/Non-Rumors'
        #
        #     df = collections.defaultdict()
        #     df_w = collections.defaultdict()
        #     tweet_avg_med_var_2 = collections.defaultdict(list)
        #     tweet_dev_avg_med_var_2 = collections.defaultdict(list)
        #     tweet_dev_avg_2 = {}
        #     tweet_dev_med_2 = {}
        #     tweet_dev_var_2 = {}
        #     tweet_avg_2 = {}
        #     tweet_med_2 = {}
        #     tweet_var_2 = {}
        #     tweet_gt_var_2 = {}
        #
        #     tweet_dev_avg_l_2 = []
        #     tweet_dev_med_l_2 = []
        #     tweet_dev_var_l_2 = []
        #     tweet_avg_l_2 = []
        #     tweet_med_l_2 = []
        #     tweet_var_l_2 = []
        #     tweet_gt_var_l_2 = []
        #     avg_susc = 0
        #     avg_gull = 0
        #     avg_cyn = 0
        #
        #     tweet_abs_dev_avg_2 = {}
        #     tweet_abs_dev_med_2 = {}
        #     tweet_abs_dev_var_2 = {}
        #
        #     tweet_abs_dev_avg_l_2 = []
        #     tweet_abs_dev_med_l_2 = []
        #     tweet_abs_dev_var_l_2 = []
        #
        #     tweet_abs_dev_avg_rnd_2 = {}
        #     tweet_dev_avg_rnd_2 = {}
        #
        #     tweet_skew_2 = {}
        #     tweet_skew_l_2 = []
        #
        #     tweet_vote_avg_med_var_2 = collections.defaultdict(list)
        #     tweet_vote_avg_2 = collections.defaultdict()
        #     tweet_vote_med_2 = collections.defaultdict()
        #     tweet_vote_var_2 = collections.defaultdict()
        #
        #     tweet_avg_group_2 = collections.defaultdict()
        #     tweet_med_group_2 = collections.defaultdict()
        #     tweet_var_group_2 = collections.defaultdict()
        #     tweet_var_diff_group_2 = collections.defaultdict()
        #
        #     tweet_kldiv_group_2 = collections.defaultdict()
        #
        #     tweet_vote_avg_l_2 = []
        #     tweet_vote_med_l_2 = []
        #     tweet_vote_var_l_2 = []
        #     tweet_chi_group_2 = {}
        #     tweet_chi_group_1 = {}
        #     tweet_chi_group_2 = {}
        #     tweet_skew = {}
        #     ind_l=[1]
        #     ans_l_2 = []
        #     for ind in ind_l:
        #         if balance_f == 'balanced':
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
        #         else:
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
        #         inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
        #         df[ind] = pd.read_csv(inp1, sep="\t")
        #         # df_w[ind] = pd.read_csv(inp1_w, sep="\t")
        #
        #         df_m = df[ind].copy()
        #
        #         groupby_ftr = 'tweet_id'
        #         grouped = df_m.groupby(groupby_ftr, sort=False)
        #         grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()
        #
        #
        #         # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree', 'employment',
        #         #                          'income','political_view', 'race', 'marital_status']
        #         # for demographic_feat in demographic_feat_list:
        #         #     print('--------------------' + demographic_feat + '--------------------')
        #         #     for lean_f in set(df_m[demographic_feat]):
        #         #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
        #         #         print(lean_f + ' : '+str(len(set(df_lean_f['worker_id']))))
        #
        #
        #
        #
        #
        #         for t_id in grouped.groups.keys():
        #             df_tmp = df_m[df_m['tweet_id'] == t_id]
        #             ind_t = df_tmp.index.tolist()[0]
        #             weights = []
        #
        #             dem_df = df_tmp[df_tmp['leaning'] == 1]
        #             rep_df = df_tmp[df_tmp['leaning'] == -1]
        #             neut_df = df_tmp[df_tmp['leaning'] == 0]
        #             dem_val_list = list(dem_df['rel_v'])
        #             rep_val_list = list(rep_df['rel_v'])
        #             neut_val_list = list(neut_df['rel_v'])
        #
        #             dem_val_list = list(dem_df['ra'])
        #             rep_val_list = list(rep_df['ra'])
        #             neut_val_list = list(neut_df['ra'])
        #
        #             dem_num = len(dem_df['worker_id'])
        #             rep_num = len(rep_df['worker_id'])
        #             neut_num = len(neut_df['worker_id'])
        #
        #
        #
        #             # df_tmp = dem_df.copy()
        #
        #             val_list = list(df_tmp['rel_v'])
        #             ans_l_2 += list(df_tmp['ra'])
        #             tweet_avg_group_2[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
        #             tweet_med_group_2[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
        #             tweet_var_diff_group_2[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
        #             tweet_var_group_2[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
        #             tweet_kldiv_group_2[t_id] = np.round(scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)
        #
        #             # tweet_skew[t_id] = scipy.stats.skew(val_list)
        #             tweet_skew_2[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
        #             # tweet_skew_l.append(tweet_skew[t_id])
        #
        #             weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
        #             # val_list = list(df_tmp['rel_v_b'])
        #             val_list = list(df_tmp['rel_v'])
        #             val_list_ra = list(df_tmp['ra'])
        #
        #             tweet_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
        #             tweet_avg_2[t_id] = np.mean(val_list)
        #             tweet_med_2[t_id] = np.median(val_list)
        #             tweet_var_2[t_id] = np.var(val_list)
        #             # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]
        #
        #             tweet_avg_l_2.append(np.mean(val_list))
        #             tweet_med_l_2.append(np.median(val_list))
        #             tweet_var_l_2.append(np.var(val_list))
        #             # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])
        #
        #             vot_list = []
        #             vot_list_tmp = list(df_tmp['vote'])
        #             # vot_list_tmp = []
        #
        #             for vot in vot_list_tmp:
        #                 if vot < 0:
        #                     vot_list.append(vot)
        #             tweet_vote_avg_med_var_2[t_id] = [np.mean(vot_list), np.median(vot_list), np.var(vot_list)]
        #             tweet_vote_avg_2[t_id] = np.mean(vot_list)
        #             tweet_vote_med_2[t_id] = np.median(vot_list)
        #             tweet_vote_var_2[t_id] = np.var(vot_list)
        #
        #             tweet_vote_avg_l_2.append(np.mean(vot_list))
        #             tweet_vote_med_l_2.append(np.median(vot_list))
        #             tweet_vote_var_l_2.append(np.var(vot_list))
        #
        #             # val_list = list(df_tmp['susc'])
        #             # val_list = list(df_tmp['err_b'])
        #             val_list = list(df_tmp['err'])
        #             abs_var_err = [np.abs(x) for x in val_list]
        #             tweet_dev_avg_med_var_2[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
        #             tweet_dev_avg_2[t_id] = np.mean(val_list)
        #             tweet_dev_med_2[t_id] = np.median(val_list)
        #             tweet_dev_var_2[t_id] = np.var(val_list)
        #
        #             tweet_dev_avg_l_2.append(np.mean(val_list))
        #             tweet_dev_med_l_2.append(np.median(val_list))
        #             tweet_dev_var_l_2.append(np.var(val_list))
        #
        #             tweet_abs_dev_avg_2[t_id] = np.mean(abs_var_err)
        #             tweet_abs_dev_med_2[t_id] = np.median(abs_var_err)
        #             tweet_abs_dev_var_2[t_id] = np.var(abs_var_err)
        #
        #             tweet_abs_dev_avg_l_2.append(np.mean(abs_var_err))
        #             tweet_abs_dev_med_l_2.append(np.median(abs_var_err))
        #             tweet_abs_dev_var_l_2.append(np.var(abs_var_err))
        #
        #             mplpl.figure()
        #             width=0.3
        #             # mplpl.hist([val_list_ra], normed=False, color=['c'])
        #             val_1=0;val_2=0;val_3=0;val_4=0;val_5=0;val_6=0;val_7=0;
        #             for mm in val_list_ra:
        #                 if mm==1:
        #                     val_1+=1
        #                 elif mm==2:
        #                     val_2+=1
        #                 elif mm == 3:
        #                     val_3 += 1
        #                 elif mm == 4:
        #                     val_4 += 1
        #                 elif mm == 5:
        #                     val_5 += 1
        #                 elif mm == 6:
        #                     val_6 += 1
        #                 elif mm == 7:
        #                     val_7 += 1
        #
        #             # hist_list = [val_1/float(len(val_list_ra)), val_2/float(len(val_list_ra)), val_3/float(len(val_list_ra))
        #             #              , val_4/float(len(val_list_ra)), val_5/float(len(val_list_ra)), val_6/float(len(val_list_ra))
        #             #              , val_7/float(len(val_list_ra))]
        #             # mplpl.bar([0,1,2,3,4,5,6], hist_list,width,  color='c', label='all workers')
        #             #
        #             # df_rep = pd.DataFrame(np.array(val_list_ra), columns=['dist'])
        #             # df_rep['dist'].plot(kind='kde', lw=6, color='c', label='')
        #             #
        #             #
        #             # mplpl.legend(loc="upper left")
        #             # # labels = ['Republican', 'Neutral', 'Democrat']
        #             # labels = ['Confirm \nit to\nbe a false', '', 'Possibly\nfalse',
        #             #           '','Possibly\ntrue', '', 'Confirm\nit to be\ntrue']
        #             # x = range(0, 7)
        #             # mplpl.xticks(x, labels)  # , rotation='90')
        #             # mplpl.subplots_adjust(bottom=0.2)
        #             # mplpl.xlabel('Workers judgements', fontsize=18)
        #             #
        #             # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_all_' + str(t_id)
        #             # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_amt_1/' + 'claims_sp_2_judgment_all_' + str(
        #             #     t_id)
        #             # # mplpl.savefig(pp + '.pdf', format='pdf')
        #             # mplpl.savefig(pp + '.png', format='png')
        #
        #             # mplpl.figure()
        #             Y = [0, 0, 0, 0, 0, 0, 0]
        #             width = 0.3
        #             count = 1
        #             dif_aprty_judg_l = [rep_val_list, neut_val_list,dem_val_list]
        #             col_l = ['r', 'g','b']
        #             lab_d = ['Rep', 'Neut', 'Dem']
        #
        #
        #             # lab_d = ['Rep', 'Other', 'Dem']
        #             # for categ in [0, 1, 2]:
        #             #     party_judgment = dif_aprty_judg_l[categ]
        #             #     categ_num = collections.defaultdict(float)
        #             #     if categ == 0:
        #             #         div = rep_num
        #             #     elif categ == 1:
        #             #         div = neut_num
        #             #     elif categ == 2:
        #             #         div = dem_num
        #             #     for el in party_judgment:
        #             #         categ_num[el] += float(1) / div
        #             #
        #             #     mplpl.bar([0, 1, 2, 3, 4, 5, 6], [categ_num[1], categ_num[2], categ_num[3],0, categ_num[5], categ_num[6],categ_num[7]], width, bottom=np.array(Y),
        #             #               color=col_l[count - 1], label=lab_d[count - 1])
        #             #     Y = np.array(Y) + np.array([categ_num[1], categ_num[2], categ_num[3], 0, categ_num[5], categ_num[6],categ_num[7]])
        #             #     count += 1
        #
        #
        #             for categ in [0, 1, 2]:
        #                 party_judgment = dif_aprty_judg_l[categ]
        #                 categ_num = collections.defaultdict(float)
        #                 if categ == 0:
        #                     div = rep_num
        #                 elif categ == 1:
        #                     div = neut_num
        #                 elif categ == 2:
        #                     div = dem_num
        #                 for el in party_judgment:
        #                     categ_num[el] += float(1) / div
        #
        #                 if categ == 0:
        #                     rep_l = [categ_num[1], categ_num[2], categ_num[3],0, categ_num[5], categ_num[6],categ_num[7]]
        #                     rep_l_j = party_judgment
        #                 if categ == 1:
        #                     neut_l = [categ_num[1], categ_num[2], categ_num[3], 0, categ_num[5], categ_num[6],
        #                              categ_num[7]]
        #                     neut_l_j = party_judgment
        #
        #                 if categ == 2:
        #                     dem_l = [categ_num[1], categ_num[2], categ_num[3], 0, categ_num[5], categ_num[6],
        #                              categ_num[7]]
        #                     dem_l_j = party_judgment
        #
        #             widtgh = 0.2
        #             # mplpl.figure()
        #             # # mplpl.hist([rep_l_j, neut_l_j, dem_l_j], color=['r', 'g', 'b'])
        #             # mplpl.bar([0-width,1-width,2-width,3-width,4-width,5-width,6-width], rep_l, width, color='r', label='Rep')
        #             # mplpl.bar([0,1,2,3,4,5,6], neut_l,width,  color='g', label='Neut')
        #             # mplpl.bar([0 + width,1+ width,2+ width,3+ width,4+ width,5+ width,6+ width],  dem_l,width, color='b', label='Dem')
        #             #
        #             # mplpl.legend(loc="upper center")
        #             # # labels = ['Republican', 'Neutral', 'Democrat']
        #             # labels = ['Confirm \nit to\nbe a false', 'Very likely\nto be\na false', 'Possibly\nfalse',
        #             #           'Can\'t tell','Possibly\ntrue', 'Very likely\nto be\ntrue', 'Confirm\nit to be\ntrue']
        #             # x = [0, 1, 2, 3, 4, 5, 6]
        #             # mplpl.xticks(x, labels)  # , rotation='90')
        #             # mplpl.subplots_adjust(bottom=0.2)
        #             # mplpl.xlabel('Workers judgements', fontsize=18)
        #             #
        #             # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_diff_party_' + str(t_id)
        #             # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_amt_1/' + 'claims_sp_2_judgment_diff_party_' + str(
        #             #     t_id) + '_hist'
        #             # # mplpl.savefig(pp + '.pdf', format='pdf')
        #             # mplpl.savefig(pp + '.png', format='png')
        #             #
        #             # mplpl.figure()
        #             #
        #             # df_rep = pd.DataFrame(np.array(rep_l_j), columns=['Rep'])
        #             # df_rep['Rep'].plot(kind='kde', lw=6, color='r', label='Rep')
        #             #
        #             # df_neut = pd.DataFrame(np.array(neut_l_j), columns=['Neut'])
        #             # df_neut['Neut'].plot(kind='kde', lw=6, color='g', label='Neut')
        #             #
        #             # df_dem = pd.DataFrame(np.array(dem_l_j), columns=['Dem'])
        #             # df_dem['Dem'].plot(kind='kde', lw=6, color='b', label='Dem')
        #             #
        #             #
        #             # mplpl.legend(loc="upper center")
        #             # # labels = ['Republican', 'Neutral', 'Democrat']
        #             # labels = ['Confirm \nit to\nbe a false', '', '',
        #             #           'Can\'t tell','', '', 'Confirm\nit to be\ntrue']
        #             # x = [0, 1, 2, 3, 4, 5, 6]
        #             # mplpl.xticks(x, labels)  # , rotation='90')
        #             # mplpl.subplots_adjust(bottom=0.2)
        #             # mplpl.xlabel('Workers judgements', fontsize=18)
        #             #
        #             # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_diff_party_' + str(t_id)
        #             # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_amt_1/' + 'claims_sp_2_diff_party_' + str(
        #             #     t_id) + '_density'
        #             # # mplpl.savefig(pp + '.pdf', format='pdf')
        #             # mplpl.savefig(pp + '.png', format='png')


        # exit()

        for dataset in ['snopes_incentive']:#'snopes_2' ,'snopes_incentive', 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi'or dataset == 'snopes_incentive' or dataset == 'snopes_2':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_med_var_3 = collections.defaultdict(list)
            tweet_dev_avg_3 = {}
            tweet_dev_med_3 = {}
            tweet_dev_var_3 = {}
            tweet_avg_3 = {}
            tweet_med_3 = {}
            tweet_var_3 = {}
            tweet_gt_var_3 = {}

            tweet_dev_avg_l_3 = []
            tweet_dev_med_l_3 = []
            tweet_dev_var_l_3 = []
            tweet_avg_l_3 = []
            tweet_med_l_3 = []
            tweet_var_l_3 = []
            tweet_gt_var_l_3 = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_3 = {}
            tweet_abs_dev_med_3 = {}
            tweet_abs_dev_var_3 = {}

            tweet_abs_dev_avg_l_3 = []
            tweet_abs_dev_med_l_3 = []
            tweet_abs_dev_var_l_3 = []

            tweet_abs_dev_avg_rnd_3= {}
            tweet_dev_avg_rnd_3 = {}

            tweet_skew_3 = {}
            tweet_skew_l_3 = []

            tweet_vote_avg_med_var_3 = collections.defaultdict(list)
            tweet_vote_avg_3 = collections.defaultdict()
            tweet_vote_med_3 = collections.defaultdict()
            tweet_vote_var_3 = collections.defaultdict()

            tweet_avg_group_3 = collections.defaultdict()
            tweet_med_group_3 = collections.defaultdict()
            tweet_var_group_3 = collections.defaultdict()
            tweet_var_diff_group_3 = collections.defaultdict()

            tweet_kldiv_group_3 = collections.defaultdict()

            tweet_vote_avg_l_3 = []
            tweet_vote_med_l_3 = []
            tweet_vote_var_l_3 = []
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_chi_group_3 = {}
            tweet_skew = {}
            ind_l = [1]
            ans_l_3 = []
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
                #                          'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    # dem_val_list = list(dem_df['rel_v'])
                    # rep_val_list = list(rep_df['rel_v'])
                    # neut_val_list = list(neut_df['rel_v'])

                    dem_val_list = list(dem_df['ra'])
                    rep_val_list = list(rep_df['ra'])
                    neut_val_list = list(neut_df['ra'])

                    dem_num = len(dem_df['worker_id'])
                    rep_num = len(rep_df['worker_id'])
                    neut_num = len(neut_df['worker_id'])

                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['rel_v'])
                    val_list = list(df_tmp['ra'])
                    ans_l_3 += list(df_tmp['ra'])
                    tweet_avg_group_3[t_id] = np.abs(np.mean(dem_val_list) - np.mean(rep_val_list))
                    tweet_med_group_3[t_id] = np.abs(np.median(dem_val_list) - np.median(rep_val_list))
                    tweet_var_diff_group_3[t_id] = np.abs(np.var(dem_val_list) - np.var(rep_val_list))
                    tweet_var_group_3[t_id] = np.abs(np.var(dem_val_list) + np.var(rep_val_list))
                    tweet_kldiv_group_3[t_id] = np.round(
                        scipy.stats.ks_2samp(dem_val_list, rep_val_list)[1], 4)

                    # tweet_skew[t_id] = scipy.stats.skew(val_list)
                    tweet_skew_3[t_id] = scipy.stats.skew(dem_val_list) + scipy.stats.skew(rep_val_list)
                    # tweet_skew_l.append(tweet_skew[t_id])

                    weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
                    # val_list = list(df_tmp['rel_v_b'])
                    val_list = list(df_tmp['rel_v'])
                    val_list_ra = list(df_tmp['ra'])
                    tweet_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_3[t_id] = np.mean(val_list)
                    tweet_med_3[t_id] = np.median(val_list)
                    tweet_var_3[t_id] = np.var(val_list)
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_3.append(np.mean(val_list))
                    tweet_med_l_3.append(np.median(val_list))
                    tweet_var_l_3.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])

                    vot_list = []
                    vot_list_tmp = list(df_tmp['vote'])
                    # vot_list_tmp = []

                    for vot in vot_list_tmp:
                        if vot < 0:
                            vot_list.append(vot)
                    tweet_vote_avg_med_var_3[t_id] = [np.mean(vot_list), np.median(vot_list),
                                                      np.var(vot_list)]
                    tweet_vote_avg_3[t_id] = np.mean(vot_list)
                    tweet_vote_med_3[t_id] = np.median(vot_list)
                    tweet_vote_var_3[t_id] = np.var(vot_list)

                    tweet_vote_avg_l_3.append(np.mean(vot_list))
                    tweet_vote_med_l_3.append(np.median(vot_list))
                    tweet_vote_var_l_3.append(np.var(vot_list))

                    # val_list = list(df_tmp['susc'])
                    # val_list = list(df_tmp['err_b'])
                    val_list = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list]
                    tweet_dev_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list),
                                                     np.var(val_list)]
                    tweet_dev_avg_3[t_id] = np.mean(val_list)
                    tweet_dev_med_3[t_id] = np.median(val_list)
                    tweet_dev_var_3[t_id] = np.var(val_list)

                    tweet_dev_avg_l_3.append(np.mean(val_list))
                    tweet_dev_med_l_3.append(np.median(val_list))
                    tweet_dev_var_l_3.append(np.var(val_list))

                    tweet_abs_dev_avg_3[t_id] = np.mean(abs_var_err)
                    tweet_abs_dev_med_3[t_id] = np.median(abs_var_err)
                    tweet_abs_dev_var_3[t_id] = np.var(abs_var_err)

                    tweet_abs_dev_avg_l_3.append(np.mean(abs_var_err))
                    tweet_abs_dev_med_l_3.append(np.median(abs_var_err))
                    tweet_abs_dev_var_l_3.append(np.var(abs_var_err))

                    mplpl.figure()
                    width=0.3
                    # mplpl.hist([val_list_ra], normed=False, color=['c'])
                    val_1=0;val_2=0;val_3=0;val_4=0;val_5=0;val_6=0;val_7=0;
                    for mm in val_list_ra:
                        # if mm==1:
                        #     val_1+=1
                        # elif mm==2:
                        #     val_2+=1
                        # elif mm == 3:
                        #     val_3 += 1
                        # elif mm == 4:
                        #     val_4 += 1
                        # elif mm == 5:
                        #     val_5 += 1
                        # elif mm == 6:
                        #     val_6 += 1
                        # elif mm == 7:
                        #     val_7 += 1

                        #binary
                        if mm == 1:
                            val_1 += 1
                        elif mm == 2:
                            val_1 += 1
                        elif mm == 3:
                            val_1 += 1
                        # elif mm == 4:
                        #     val_4 += 1
                        elif mm == 5:
                            val_2 += 1
                        elif mm == 6:
                            val_2 += 1
                        elif mm == 7:
                            val_2 += 1

                    # hist_list = [val_1/float(len(val_list_ra)), val_2/float(len(val_list_ra)), val_3/float(len(val_list_ra))
                    #              , val_4/float(len(val_list_ra)), val_5/float(len(val_list_ra)), val_6/float(len(val_list_ra))
                    #              , val_7/float(len(val_list_ra))]

                    hist_list = [val_1/float(len(val_list_ra)), val_2/float(len(val_list_ra))]

                    # mplpl.bar([0,1,2,3,4,5,6], hist_list,width,  color='c', label='all workers')
                    #
                    # df_rep = pd.DataFrame(np.array(val_list_ra), columns=['dist'])
                    # df_rep['dist'].plot(kind='kde', lw=6, color='c', label='')
                    #
                    #
                    # mplpl.legend(loc="upper left")
                    # # labels = ['Republican', 'Neutral', 'Democrat']
                    # labels = ['Confirm \nit to\nbe a false', '', 'Possibly\nfalse',
                    #           '','Possibly\ntrue', '', 'Confirm\nit to be\ntrue']
                    # x = range(0, 7)
                    # mplpl.xticks(x, labels)  # , rotation='90')
                    # mplpl.subplots_adjust(bottom=0.2)
                    # mplpl.xlabel('Workers judgements', fontsize=18)
                    #
                    # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_all_' + str(t_id)
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment/' + 'claims_sp_incentive_judgment_all_' + str(
                    #     t_id)
                    # # mplpl.savefig(pp + '.pdf', format='pdf')
                    # mplpl.savefig(pp + '.png', format='png')

                    # mplpl.figure()
                    Y = [0, 0, 0]
                    width = 0.3
                    count = 1
                    dif_aprty_judg_l = [rep_val_list, neut_val_list,dem_val_list]
                    col_l = ['r', 'g','b']
                    lab_d = ['Rep', 'Neut', 'Dem']


                    for categ in [0, 1, 2]:
                        party_judgment = dif_aprty_judg_l[categ]
                        categ_num = collections.defaultdict(float)
                        if categ == 0:
                            div = rep_num
                        elif categ == 1:
                            div = neut_num
                        elif categ == 2:
                            div = dem_num
                        for el in party_judgment:
                            categ_num[el] += float(1)/div
                            # categ_num[el] += 1

                        if categ == 0:
                            rep_l = [categ_num[1]+categ_num[2]+categ_num[3],0, categ_num[5]+ categ_num[6]+categ_num[7]]
                            rep_l_j = party_judgment
                        if categ == 1:
                            neut_l = [categ_num[1]+ categ_num[2]+ categ_num[3], 0, categ_num[5]+ categ_num[6]+categ_num[7]]
                            neut_l_j = party_judgment

                        if categ == 2:
                            dem_l = [categ_num[1]+ categ_num[2]+ categ_num[3], 0, categ_num[5]+ categ_num[6]+categ_num[7]]
                            dem_l_j = party_judgment

                    widtgh = 0.2
                    mplpl.figure()
                    # mplpl.hist([rep_l_j, neut_l_j, dem_l_j], color=['r', 'g', 'b'])
                    mplpl.bar([0-width,1-width,2-width], rep_l, width, color='r', label='Rep')
                    mplpl.bar([0,1,2], neut_l,width,  color='g', label='Neut')
                    mplpl.bar([0 + width,1+ width,2+ width],  dem_l,width, color='b', label='Dem')

                    mplpl.legend(loc="upper center")
                    # labels = ['Republican', 'Neutral', 'Democrat']
                    labels = ['False','', 'True']
                    x = [0, 1, 2]
                    mplpl.xticks(x, labels)  # , rotation='90')
                    mplpl.subplots_adjust(bottom=0.2)
                    mplpl.xlabel('Workers judgements (Binary)', fontsize=18)
                    mplpl.ylim([0,1])
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_diff_party_' + str(t_id)
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_incentive/' + 'claims_sp_incentive_diff_party_binary_' + str(
                    #     t_id) + '_hist'
                    pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_incentive/' + 'claims_sp_incentive_diff_party_binary_normed_' + str(
                        t_id) + '_hist'
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_amt_sp_2/' + 'claims_sp_2_judgment_diff_party_binary_' + str(
                    #     t_id) + '_hist'
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_amt_sp_2/' + 'claims_sp_2_judgment_diff_party_binary_normed_' + str(
                    #     t_id) + '_hist'
                    # mplpl.savefig(pp + '.pdf', format='pdf')
                    mplpl.savefig(pp + '.png', format='png')

                    # widtgh = 0.2
                    # mplpl.figure()
                    # # mplpl.hist([rep_l_j, neut_l_j, dem_l_j], color=['r', 'g', 'b'])
                    # mplpl.bar([0-width,1-width,2-width,3-width,4-width,5-width,6-width], rep_l, width, color='r', label='Rep')
                    # mplpl.bar([0,1,2,3,4,5,6], neut_l,width,  color='g', label='Neut')
                    # mplpl.bar([0 + width,1+ width,2+ width,3+ width,4+ width,5+ width,6+ width],  dem_l,width, color='b', label='Dem')
                    #
                    # mplpl.legend(loc="upper center")
                    # # labels = ['Republican', 'Neutral', 'Democrat']
                    # labels = ['Confirm \nit to\nbe a false', 'Very likely\nto be\na false', 'Possibly\nfalse',
                    #           'Can\'t tell','Possibly\ntrue', 'Very likely\nto be\ntrue', 'Confirm\nit to be\ntrue']
                    # x = [0, 1, 2, 3, 4, 5, 6]
                    # mplpl.xticks(x, labels)  # , rotation='90')
                    # mplpl.subplots_adjust(bottom=0.2)
                    # mplpl.xlabel('Workers judgements', fontsize=18)
                    #
                    # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_diff_party_' + str(t_id)
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment/' + 'claims_sp_2_judgment_diff_party_' + str(
                    #     t_id) + '_hist'
                    # # mplpl.savefig(pp + '.pdf', format='pdf')
                    # mplpl.savefig(pp + '.png', format='png')



                    #
                    # try:
                    #     mplpl.figure()
                    #
                    #     df_rep = pd.DataFrame(np.array(rep_l_j), columns=['Rep'])
                    #     df_rep['Rep'].plot(kind='kde', lw=6, color='r', label='Rep')
                    #
                    #     df_neut = pd.DataFrame(np.array(neut_l_j), columns=['Neut'])
                    #     df_neut['Neut'].plot(kind='kde', lw=6, color='g', label='Neut')
                    #
                    #     df_dem = pd.DataFrame(np.array(dem_l_j), columns=['Dem'])
                    #     df_dem['Dem'].plot(kind='kde', lw=6, color='b', label='Dem')
                    #
                    #
                    #     mplpl.legend(loc="upper center")
                    #     # labels = ['Republican', 'Neutral', 'Democrat']
                    #     # labels = ['Confirm \nit to\nbe a false', '', '',
                    #     #           'Can\'t tell','', '', 'Confirm\nit to be\ntrue']
                    #     labels = ['False', 'True']
                    #     x = [-1, 1]
                    #     mplpl.xticks(x, labels)  # , rotation='90')
                    #     mplpl.subplots_adjust(bottom=0.2)
                    #     mplpl.xlabel('Workers judgements', fontsize=18)
                    #
                    #     # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_diff_party_' + str(t_id)
                    #     pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_judgment_incentive/' + 'claims_sp_2_diff_party_binary_' + str(
                    #         t_id)# + '_density'
                    #     # mplpl.savefig(pp + '.pdf', format='pdf')
                    #     mplpl.savefig(pp + '.png', format='png')
                    #
                    # except:
                    #     print(t_id)
                    #     continue

        exit()
        for dataset in ['snopes_leaning_ben']:  # , 'snopes_nonpol', 'politifact', 'mia', 'politifact', 'snopes', 'mia', 'politifact']:
            if dataset == 'snopes' or dataset == 'snopes_ssi' or dataset == 'snopes_incentive'or dataset == 'snopes_leaning'or dataset == 'snopes_leaning_ben':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                # remotedir=''
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')
            if dataset == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            elif dataset == 'snopes_leaning':
                data_n = 'sp_leaning'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_leaning'


            elif dataset == 'snopes_leaning_ben':
                data_n = 'sp_leaning_ben'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_leaning_ben'


            elif dataset == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                data_name = 'Snopes_ssi'
            elif dataset == 'snopes_nonpol':
                data_n = 'sp_nonpol'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_nonpol'
            elif dataset == 'politifact':
                data_addr = 'politifact/fig/'
                data_n = 'pf'
                ind_l = [1, 2, 3]
                data_name = 'PolitiFact'
            elif dataset == 'mia':
                data_addr = 'mia/fig/'

                data_n = 'mia'
                ind_l = [1]
                data_name = 'Rumors/Non-Rumors'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var_leaning = collections.defaultdict(list)
            tweet_dev_avg_med_var_leaning = collections.defaultdict(list)
            tweet_dev_avg_leaning = {}
            tweet_dev_med_leaning = {}
            tweet_dev_var_leaning = {}
            tweet_avg_leaning = {}
            tweet_med_leaning = {}
            tweet_var_leaning = {}
            tweet_gt_var_leaning = {}

            tweet_dev_avg_l_leaning = []
            tweet_dev_med_l_leaning = []
            tweet_dev_var_l_leaning = []
            tweet_avg_l_leaning = []
            tweet_med_l_leaning = []
            tweet_var_l_leaning = []
            tweet_gt_var_l_leaning = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg_leaning = {}
            tweet_abs_dev_med_leaning = {}
            tweet_abs_dev_var_leaning = {}

            tweet_abs_dev_avg_l_leaning = []
            tweet_abs_dev_med_l_leaning = []
            tweet_abs_dev_var_l_leaning = []

            tweet_abs_dev_avg_rnd_leaning = {}
            tweet_dev_avg_rnd_leaning = {}

            tweet_skew_leaning = {}
            tweet_skew_l_leaning = []

            tweet_vote_avg_med_var_leaning = collections.defaultdict(list)
            tweet_vote_avg_leaning = collections.defaultdict()
            tweet_vote_med_leaning = collections.defaultdict()
            tweet_vote_var_leaning = collections.defaultdict()

            tweet_avg_group_leaning = collections.defaultdict()
            tweet_med_group_leaning = collections.defaultdict()
            tweet_var_group_leaning = collections.defaultdict()
            tweet_var_diff_group_leaning = collections.defaultdict()

            tweet_kldiv_group_leaning = collections.defaultdict()

            tweet_vote_avg_l_leaning = []
            tweet_vote_med_l_leaning = []
            tweet_vote_var_l_leaning = []
            tweet_chi_group_leaning = {}
            tweet_chi_group_leaning = {}
            tweet_chi_group_leaning = {}
            tweet_leaning_list_leaning = collections.defaultdict(list)
            tweet_skew = {}
            ind_l = [1]
            ans_l_leaning = []
            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # demographic_feat_list = ['nationality', 'residence', 'gender', 'age', 'degree',
                #                          'employment',
                #                          'income', 'political_view', 'race', 'marital_status']
                # for demographic_feat in demographic_feat_list:
                #     print('--------------------' + demographic_feat + '--------------------')
                #     for lean_f in set(df_m[demographic_feat]):
                #         df_lean_f = df_m[df_m[demographic_feat] == lean_f]
                #         print(lean_f + ' : ' + str(len(set(df_lean_f['worker_id']))))

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    ind_t = df_tmp.index.tolist()[0]
                    weights = []

                    dem_df = df_tmp[df_tmp['leaning'] == 1]
                    rep_df = df_tmp[df_tmp['leaning'] == -1]
                    neut_df = df_tmp[df_tmp['leaning'] == 0]
                    dem_val_list = list(dem_df['tweet_leaning'])
                    rep_val_list = list(rep_df['tweet_leaning'])
                    neut_val_list = list(neut_df['tweet_leaning'])


                    dem_num = len(dem_df['worker_id'])
                    rep_num = len(rep_df['worker_id'])
                    neut_num = len(neut_df['worker_id'])

                    dif_aprty_judg_l = [rep_val_list, neut_val_list,dem_val_list]
                    col_l = ['r', 'g','b']
                    # df_tmp = dem_df.copy()

                    val_list = list(df_tmp['tweet_leaning'])
                    ans_l_leaning += list(df_tmp['ra'])


                    val_list = list(df_tmp['tweet_leaning'])
                    # tweet_avg_med_var_3[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
                    tweet_avg_leaning[t_id] = np.mean(val_list)
                    tweet_med_leaning[t_id] = np.median(val_list)
                    tweet_var_leaning[t_id] = np.var(val_list)
                    tweet_leaning_list_leaning[t_id] = val_list
                    # tweet_gt_var_2[t_id] = df_tmp['rel_gt_v'][ind_t]

                    tweet_avg_l_leaning.append(np.mean(val_list))
                    tweet_med_l_leaning.append(np.median(val_list))
                    tweet_var_l_leaning.append(np.var(val_list))
                    # tweet_gt_var_l_2.append(df_tmp['rel_gt_v'][ind_t])
                    # mplpl.figure()
                    #
                    # mplpl.hist([val_list], normed=False, color=['c'])
                    # mplpl.legend(loc="upper left")
                    # # labels = ['Republican', 'Neutral', 'Democrat']
                    # labels = ['Republican', 'Other', 'Democrat']
                    # x = [-1,0,1]
                    # mplpl.xticks(x, labels)  # , rotation='90')
                    # mplpl.subplots_adjust(bottom=0.2)
                    # mplpl.xlabel('Workers judgements (Which party benefit)', fontsize=18)

                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_all_' + str(t_id)
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning_ben/' + 'claims_leaning_ben_all_' + str(t_id)
                    # mplpl.savefig(pp + '.pdf', format='pdf')
                    # mplpl.savefig(pp + '.png', format='png')



                    # mplpl.figure()
                    # Y = [0,0,0]
                    # width = 0.3
                    # count=1
                    # lab_d = ['Rep', 'Neut', 'Dem']
                    # # lab_d = ['Rep', 'Other', 'Dem']
                    # for categ in [0,1,2]:
                    #     party_judgment = dif_aprty_judg_l[categ]
                    #     categ_num = collections.defaultdict(float)
                    #     if categ==0:
                    #         div = rep_num
                    #     elif categ==1:
                    #         div = neut_num
                    #     elif categ==2:
                    #         div = dem_num
                    #     for el in party_judgment:
                    #         categ_num[el]+=float(1)/div
                    #
                    #     mplpl.bar([0, 1, 2], [categ_num[-1],categ_num[0],categ_num[1]], width, bottom=np.array(Y), color=col_l[count - 1], label=lab_d[count-1])
                    #     Y = np.array(Y) + np.array([categ_num[-1],categ_num[0],categ_num[1]])
                    #     count+=1
                    #
                    # mplpl.legend(loc="upper left")
                    # # labels = ['Republican', 'Neutral', 'Democrat']
                    # labels = ['Republican', 'Other', 'Democrat']
                    # x = range(0, 3)
                    # mplpl.xticks(x, labels)  # , rotation='90')
                    # mplpl.subplots_adjust(bottom=0.2)
                    # mplpl.xlabel('Workers judgements (Which party benefit)', fontsize=18)
                    #
                    # # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_diff_party_' + str(t_id)
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning_ben/' + 'claims_leaning_ben_diff_party_' + str(t_id)
                    # # mplpl.savefig(pp + '.pdf', format='pdf')
                    # mplpl.savefig(pp + '.png', format='png')






                num_bins = len(tweet_avg_leaning)
                # counts, bin_edges = np.histogram(df_tmp_f['cyn'], bins=num_bins, normed=True)
                counts, bin_edges = np.histogram(tweet_avg_leaning.values(), bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='c', lw=5,label='')

                # count += 1

                # mplpl.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3))
                mplpl.legend(loc='upper center')
                mplpl.rc('legend', fontsize='small')
                # loc="upper left")
                # mplpl.subplots_adjust(top=0.8)

                mplpl.grid()
                mplpl.xlabel('Workers judgments (which party benefit)', fontsize=18)
                # mplpl.xlabel('Workers judgments (Claims\' leaning)', fontsize=18)
                mplpl.ylabel('CDF', fontsize=18)
                pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning_ben/' + 'claims_leaning_ben_CDF'
                # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/tweet_leaning/' + 'claims_leaning_CDF'
                # # mplpl.savefig(pp + '.pdf', format='pdf')
                mplpl.savefig(pp + '.png', format='png')


                ##################################################
        exit()
        len_cat_dict = {}

        # tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)
        tweet_vote_sort = sorted(tweet_vote_avg, key=tweet_vote_avg.get, reverse=False)

        # tweet_vote_sort = sorted(tweet_abs_dev_avvg, key=tweet_abs_dev_avg.get, reverse=True)
        tweet_var_sort = sorted(tweet_var_2, key=tweet_var_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)
        # tweet_var_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)

        dispt_diff = collections.defaultdict()
        TPB_diff = collections.defaultdict()
        tweet_avg_ll = []
        tweet_avg_ll_2 = []
        tweet_var_ll = []
        tweet_var_ll_2 = []
        tweet_avg_ll_3 = []
        tweet_var_ll_3 = []
        TPB_l=[]
        TPB_l_2=[]
        TPB_l_3=[]
        for t_id in tweet_var_sort:
            tweet_avg_ll_2.append(tweet_avg_2[t_id])
            tweet_avg_ll.append(tweet_avg[t_id])

            tweet_var_ll_2.append(tweet_var_2[t_id])
            tweet_var_ll.append(tweet_var[t_id])

            tweet_avg_ll_3.append(tweet_avg_3[t_id])

            tweet_var_ll_3.append(tweet_var_3[t_id])
            TPB_l.append(tweet_abs_dev_avg[t_id])
            TPB_l_2.append(tweet_abs_dev_avg_2[t_id])
            TPB_l_3.append(tweet_abs_dev_avg_3[t_id])


            dispt_diff[t_id] = np.abs(tweet_var_2[t_id]-tweet_var[t_id])

            TPB_diff[t_id] = np.abs(tweet_abs_dev_avg_2[t_id] - tweet_abs_dev_avg[t_id])
        # mplpl.scatter(range(len(tweet_avg_ll_2)),tweet_avg_ll_2,color='r', label='PTL(AMT)')
        # mplpl.scatter(range(len(tweet_avg_ll)),tweet_avg_ll,color='b', label='PTL(SSI)')
        # # mplpl.xlim([-.02, 1.02])
        # # mplpl.ylim([0, 1.02])
        # mplpl.xlabel('Snopes news stories sorted based on PTL(AMT)', fontsize=18)
        # mplpl.ylabel('PTL', fontsize=18)
        # # #
        # mplpl.legend(loc="upper right")
        # # #
        # mplpl.figure()
        #
        # mplpl.scatter(range(len(tweet_var_ll_2)),tweet_var_ll_2,color='r', label='PTL(AMT)')
        # mplpl.scatter(range(len(tweet_var_ll)),tweet_var_ll,color='b', label='PTL(SSI)')
        # # mplpl.xlim([-.02, 1.02])
        # # mplpl.ylim([0, 1.02])
        # mplpl.xlabel('Snopes news stories sorted based on Disputability(AMT)', fontsize=18)
        # mplpl.ylabel('Disputability', fontsize=18)
        # # #
        # mplpl.legend(loc="upper right")


        # mplpl.figure()
        mplpl.hist([ans_l,ans_l_2, ans_l_3],normed=True, color=['r','b', 'g'], label=['SSI','AMT', 'AMT_incentive'])
        # mplpl.figure()
        # mplpl.hist(ans_l_2,color='b')
        # # mplpl.grid()
        # # mplpl.title(data_name + ' , avg : ' + str(np.round(np.mean((tweet_abs_dev_avg_rnd.values())),4)))
        # # pp = remotedir + '/fig/fig_exp1/news_based/initial/' + data_n + '_disput_nabs_perception'
        # # mplpl.savefig(pp, format='png')
        # # mplpl.figure()
        mplpl.legend(loc="upper left")
        labels = ['Confirm \nit to\nbe a false', 'Very likely\nto be\na false', 'Possibly\nfalse',
                  'Can\'t tell', 'Possibly\ntrue', 'Very likely\nto be\ntrue', 'Confirm\nit to be\ntrue']
        x = range(1, 8)
        mplpl.xticks(x, labels)#, rotation='90')
        mplpl.subplots_adjust(bottom=0.2)
        mplpl.xlabel('Workers judgements', fontsize=18)

        # mplpl.show()
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + 'SSI_AMT_judgment'
        pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/' + 'SSI_AMT_judgment_new_1'
        mplpl.savefig(pp + '.pdf', format='pdf')
        mplpl.savefig(pp + '.png', format='png')

        # tweet_var_sort = sorted(dispt_diff, key=dispt_diff.get, reverse=True)
        # tweet_var_sort = sorted(TPB_diff, key=TPB_diff.get, reverse=True)
        tweet_var_sort = sorted(tweet_avg_2, key=tweet_avg_2.get, reverse=True)

        # for t_id in tweet_var_sort:
        #     print('||' + str(t_id) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id]+'||' + str(tweet_avg_2[t_id]) +
        #           '||' + str(tweet_avg[t_id]) + '||' + str(tweet_var_2[t_id]) + '||'
        #           +str(tweet_var[t_id]) + '||'+str(tweet_abs_dev_avg_2[t_id]) + '||' + str(tweet_abs_dev_avg[t_id])+'||')





    if args.t == "AMT_dataset_reliable_user-level_processing_all_dataset_weighted_visualisation_initial_stastistics_tpb_FPB_FNB_cdf_toghether":



        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []

        dataset = 'snopes'
        # dataset = 'mia'
        # dataset = 'politifact'

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)


        line_count = 0
        tmp_dict = {}
        claims_list = []


        if dataset == 'snopes':
            remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
            news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

            sample_tweets_exp1 = []

            tweet_txt_dict = {}
            tweet_date_dict = {}
            tweet_lable_dict = {}
            tweet_publisher_dict = {}
            print(inp_all)

            for i in range(0, 5):
                df_cat = news_cat_list[i]
                df_cat_f = news_cat_list_f[i]
                inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                cat_count = 0
                for line in inF:
                    claims_list.append(line)
                    cat_count += 1

            for line in claims_list:
                line_splt = line.split('<<||>>')
                publisher_name = int(line_splt[2])
                tweet_txt = line_splt[3]
                tweet_id = publisher_name
                cat_lable = line_splt[4]
                dat = line_splt[5]
                dt_splt = dat.split(' ')[0].split('-')
                m_day = int(dt_splt[2])
                m_month = int(dt_splt[1])
                m_year = int(dt_splt[0])
                m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                tweet_txt_dict[tweet_id] = tweet_txt
                tweet_date_dict[tweet_id] = m_date
                tweet_lable_dict[tweet_id] = cat_lable


        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l= []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []


        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False



        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        for dataset in  ['snopes']:

            if dataset == 'snopes':
                claims_list = []
                col = 'r'
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['MOSTLY FALSE', 'FALSE', 'MIXTURE', 'TRUE', 'MOSTLY TRUE']
                news_cat_list_f = ['mostly_false', 'false', 'mixture', 'true', 'mostly_true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                # outF = open(remotedir + 'table_out.txt', 'w')


            if dataset=='snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1,2,3]
                col = 'purple'

                data_name = 'Snopes'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            for ind in ind_l:
                if balance_f == 'balanced':
                    inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp' + str(ind) + '_final_balanced.csv'
                else:
                    inp1 = remotedir + 'amt_answers_'+data_n+'_claims_exp' + str(ind) + '_final.csv'
                inp1_w = remotedir + 'worker_amt_answers_'+data_n+'_claims_exp' + str(ind) + '.csv'
                df[ind] = pd.read_csv(inp1, sep="\t")
                df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()
                df[ind].loc[:, 'abs_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_err'] = df[ind]['tweet_id'] * 0.0
                df[ind].loc[:, 'norm_abs_err'] = df[ind]['tweet_id'] * 0.0

                groupby_ftr = 'tweet_id'
                grouped = df[ind].groupby(groupby_ftr, sort=False)
                grouped_sum = df[ind].groupby(groupby_ftr, sort=False).sum()


                for ind_t in df[ind].index.tolist():
                    t_id = df[ind]['tweet_id'][ind_t]
                    err = df[ind]['err'][ind_t]
                    abs_err = np.abs(err)
                    df[ind]['abs_err'][ind_t] = abs_err
                    sum_rnd_abs_perc = 0
                    sum_rnd_perc = 0
                    for val in [-1, -2/float(3), -1/float(3), 0, 1/float(3), 2/float(3),1]:
                        sum_rnd_perc+= (val - df[ind]['rel_gt_v'][ind_t])
                        sum_rnd_abs_perc += np.abs(val - df[ind]['rel_gt_v'][ind_t])
                    random_perc = np.abs(sum_rnd_perc / float(7))
                    random_abs_perc = sum_rnd_abs_perc / float(7)


                    norm_err = err / float(random_perc)
                    norm_abs_err = abs_err / float(random_abs_perc)
                    df[ind]['norm_err'][ind_t] = norm_err
                    df[ind]['norm_abs_err'][ind_t] = norm_abs_err

                # df[ind] = df[ind].copy()

            w_pt_avg_l = []
            w_err_avg_l = []
            w_abs_err_avg_l = []
            w_norm_err_avg_l = []
            w_norm_abs_err_avg_l = []
            w_acc_avg_l = []

            w_pt_std_l = []
            w_err_std_l = []
            w_abs_err_std_l = []
            w_norm_err_std_l = []
            w_norm_abs_err_std_l = []
            w_acc_std_l = []

            w_pt_avg_dict = collections.defaultdict()
            w_err_avg_dict = collections.defaultdict()
            w_abs_err_avg_dict = collections.defaultdict()
            w_norm_err_avg_dict = collections.defaultdict()
            w_norm_abs_err_avg_dict = collections.defaultdict()
            w_acc_avg_dict = collections.defaultdict()

            w_pt_std_dict = collections.defaultdict()
            w_err_std_dict = collections.defaultdict()
            w_abs_err_std_dict = collections.defaultdict()
            w_norm_err_std_dict = collections.defaultdict()
            w_norm_abs_err_std_dict = collections.defaultdict()
            w_acc_std_dict = collections.defaultdict()

            all_w_pt_list  = []
            all_w_err_list = []
            all_w_abs_err_list = []
            all_w_norm_err_list = []
            all_w_norm_abs_err_list  = []
            all_w_acc_list = []

            all_w_cyn_list = []
            all_w_gull_list = []
            w_cyn_avg_l = []
            w_gull_avg_l = []
            w_cyn_std_l= []
            w_gull_std_l = []
            w_cyn_avg_dict =collections.defaultdict()
            w_gull_avg_dict =collections.defaultdict()
            w_cyn_std_dict =collections.defaultdict()
            w_gull_std_dict = collections.defaultdict()
            for ind in ind_l:

                df_m = df[ind].copy()
                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                for t_id in grouped.groups.keys():
                    df_tmp = df_m[df_m['tweet_id'] == t_id]


                    w_pt_list = list(df_tmp['rel_v'])
                    w_err_list = list(df_tmp['err'])
                    # w_abs_err_list = list(df_tmp['abs_err'])
                    w_abs_err_list = list(df_tmp['susc'])
                    w_norm_err_list = list(df_tmp['norm_err'])
                    w_norm_abs_err_list = list(df_tmp['norm_abs_err'])
                    df_cyn = df_tmp[df_tmp['cyn']>0]
                    df_gull = df_tmp[df_tmp['gull']>0]

                    w_cyn_list = list(df_cyn['cyn'])
                    w_gull_list = list(df_gull['gull'])
                    w_acc_list_tmp = list(df_tmp['acc'])
                    w_acc_list = []
                    # w_ind_acc_list
                    acc_c = 0
                    nacc_c = 0




                    # w_acc_avg_l.append(w_ind_acc_list)

                    w_pt_std_l.append(np.std(w_pt_list))
                    w_err_std_l.append(np.std(w_err_list))
                    w_abs_err_std_l.append(np.std(w_abs_err_list))
                    w_norm_err_std_l.append(np.std(w_norm_err_list))
                    w_norm_abs_err_std_l.append(np.std(w_norm_abs_err_list))
                    w_cyn_std_l.append(np.std(w_cyn_list))
                    w_gull_std_l.append(np.std(w_gull_list))
                    # w_acc_std_l.append(np.std(w_ind_acc_list))


                    w_pt_avg_dict[t_id] = np.mean(w_pt_list)
                    w_err_avg_dict[t_id] = np.mean(w_err_list)
                    w_abs_err_avg_dict[t_id] = np.mean(w_abs_err_list)
                    w_norm_err_avg_dict[t_id] = np.mean(w_norm_err_list)
                    w_norm_abs_err_avg_dict[t_id] = np.mean(w_norm_abs_err_list)
                    w_cyn_avg_dict[t_id] = np.mean(w_cyn_list)
                    w_gull_avg_dict[t_id] = np.mean(w_gull_list)
                    # w_acc_avg_dict[t_id] = w_ind_acc_list



            # fig_f = True
            fig_f = False
            # fig_f_1 = True
            fig_f_1 = False
            fig_f_together = True



            if fig_f_together==True:
                out_dict = w_abs_err_avg_dict
                # out_dict = w_gull_avg_dict
                # out_dict = w_cyn_avg_dict
                ####ptl_cdf
                mplpl.rcParams['figure.figsize'] = 4.5, 2.5
                mplpl.rc('xtick', labelsize='large')
                mplpl.rc('ytick', labelsize='large')
                mplpl.rc('legend', fontsize='medium')
                # w_err_avg_dict
                # tweet_l_sort = sorted(w_norm_abs_err_avg_dict, key=w_norm_abs_err_avg_dict.get, reverse=False)
                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                # tweet_l_sort = [x for x in tweet_l_sort if x >= 0 or x < 0]
                acc_l = []
                for t_id in tweet_l_sort:
                    if out_dict[t_id] >=0 or out_dict[t_id]<0:
                        acc_l.append(out_dict[t_id])

                num_bins = len(acc_l)
                counts, bin_edges = np.histogram(acc_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='r', lw=5, label='TPB')

                # out_dict = w_abs_err_avg_dict
                # out_dict = w_gull_avg_dict
                out_dict = w_cyn_avg_dict
                # ####ptl_cdf
                # mplpl.rcParams['figure.figsize'] = 4.5, 2.5
                # mplpl.rc('xtick', labelsize='large')
                # mplpl.rc('ytick', labelsize='large')
                # mplpl.rc('legend', fontsize='medium')
                # w_err_avg_dict
                # tweet_l_sort = sorted(w_norm_abs_err_avg_dict, key=w_norm_abs_err_avg_dict.get, reverse=False)
                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                # tweet_l_sort = [x for x in tweet_l_sort if x >= 0 or x < 0]
                acc_l = []
                for t_id in tweet_l_sort:
                    if out_dict[t_id] >= 0 or out_dict[t_id] < 0:
                        acc_l.append(out_dict[t_id])

                num_bins = len(acc_l)
                counts, bin_edges = np.histogram(acc_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='orange', lw=5, label='FNB')

                # out_dict = w_abs_err_avg_dict
                out_dict = w_gull_avg_dict
                # out_dict = w_cyn_avg_dict
                ####ptl_cdf
                # mplpl.rcParams['figure.figsize'] = 4.5, 2.5
                # mplpl.rc('xtick', labelsize='large')
                # mplpl.rc('ytick', labelsize='large')
                # mplpl.rc('legend', fontsize='medium')
                # tweet_l_sort = sorted(w_norm_abs_err_avg_dict, key=w_norm_abs_err_avg_dict.get, reverse=False)
                tweet_l_sort = sorted(out_dict, key=out_dict.get, reverse=False)
                # tweet_l_sort = [x for x in tweet_l_sort if x >= 0 or x < 0]
                acc_l = []
                for t_id in tweet_l_sort:
                    if out_dict[t_id] >= 0 or out_dict[t_id] < 0:
                        acc_l.append(out_dict[t_id])

                num_bins = len(acc_l)
                counts, bin_edges = np.histogram(acc_l, bins=num_bins, normed=True)
                cdf = np.cumsum(counts)
                scale = 1.0 / cdf[-1]
                ncdf = scale * cdf
                mplpl.plot(bin_edges[1:], ncdf, c='c', lw=5, label='FPB')



        legend_properties = {'weight': 'bold'}

        #
        mplpl.ylabel('CDF', fontsize=20, fontweight = 'bold')
        # mplpl.xlabel('Total Perception Bias', fontsize=20, fontweight = 'bold')
        # mplpl.xlabel('False Positive Bias', fontsize=20, fontweight = 'bold')
        mplpl.xlabel('Perception Bias Measures', fontsize=20, fontweight = 'bold')
        # mplpl.xlabel('False Negative Bias', fontsize=20, fontweight = 'bold')
        mplpl.legend(loc="lower right", prop=legend_properties, fontsize='medium', ncol=1)
        # mplpl.title(data_name)
        # mplpl.legend(loc="upper left",fontsize = 'large')
        mplpl.xlim([0, 2])
        mplpl.ylim([0, 1])
        mplpl.grid()
        mplpl.subplots_adjust(bottom=0.24)
        mplpl.subplots_adjust(left=0.18)
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/NAPB_cdf_alldataset'
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/APB_cdf_alldataset_news'
        # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/FPB_cdf_alldataset_new'
        pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/fig/news_based/TPB_FPB_FNB_cdf_alldataset_new'
        mplpl.savefig(pp + '.pdf', format='pdf')
        mplpl.savefig(pp + '.png', format='png')

        exit()

        # elif fig_f_1==True:
            #     balance_f = 'un_balanced'
            #     # balance_f = 'balanced'
            #
            #     # fig_f = True
            #     # fig_f = False
            #     if dataset == 'snopes':
            #         data_n = 'sp'
            #     elif dataset == 'politifact':
            #         data_n = 'pf'
            #     elif dataset == 'mia':
            #         data_n = 'mia'
            #
            #     fig_p = 7
            #
            #     for ind in ind_l:
            #
            #         df_m = df[ind].copy()
            #
            #         groupby_ftr = 'worker_id'
            #         grouped = df_m.groupby(groupby_ftr, sort=False)
            #         grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()
            #
            #         # df_tmp = df_m[df_m['tweet_id'] == t_id]
            #         w_cc=0
            #         for w_id in grouped.groups.keys():
            #             print(w_cc)
            #             w_cc+=1
            #             weights = []
            #
            #             w_pt_list = []
            #
            #             df_tmp = df_m[df_m['worker_id'] == w_id]
            #             ind_t = df_tmp.index.tolist()[0]
            #             weights = []
            #             w_acc_list_tmp = list(df_tmp['acc'])
            #             w_acc_list = []
            #             for el in w_acc_list_tmp:
            #                 if el == 0:
            #                     w_acc_list.append(-1)
            #                 elif el == 1:
            #                     w_acc_list.append(1)
            #                 else:
            #                     w_acc_list.append(0)
            #             df_tt = pd.DataFrame({'val' : w_acc_list})
            #
            #             weights.append(np.ones_like(list(df_tt['val'])) / float(len(list(df_tt['val']))))
            #
            #             # tweet_avg_med_var[t_id] = [np.mean(w_acc_list), np.median(w_acc_list), np.var(w_acc_list)]
            #             # tweet_avg[t_id] = np.mean(w_acc_list)
            #             # tweet_med[t_id] = np.median(w_acc_list)
            #             # tweet_var[t_id] = np.var(w_acc_list)
            #             #
            #             # tweet_avg_l.append(np.mean(w_acc_list))
            #             # tweet_med_l.append(np.median(w_acc_list))
            #             # tweet_var_l.append(np.var(w_acc_list))
            #             accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
            #
            #             if fig_p==1:
            #                 all_acc.append(accuracy)
            #                 try:
            #                     df_tt['val'].plot(kind='kde', lw=4, color='g', label='Accuracy')
            #                 except:
            #                     print('hmm')
            #
            #                 mplpl.hist(list(df_tt['val']), weights=weights, color='g')
            #
            #
            #                 mplpl.ylabel('Frequency')
            #                 mplpl.xlabel('Accuracy')
            #                 mplpl.title(' Accuracy : ' + str(np.round(accuracy, 3)) + '\n Avg : '
            #                             + str(np.round(np.mean(w_acc_list), 3))+', Var : '
            #                             + str(np.round(np.var(w_acc_list), 3)))
            #                 mplpl.legend(loc="upper right")
            #                 mplpl.xlim([-2, 2])
            #                 mplpl.ylim([0, 1])
            #                 if balance_f == 'balanced':
            #                     pp = remotedir + '/fig/fig_exp1/news_based/balanced/ind_users/' + str(w_id) + '_acc_dist'
            #                 else:
            #                     pp = remotedir + '/fig/fig_exp1/user_based/ind_users/' + str(w_id) + '_acc_dist'
            #                 mplpl.savefig(pp, format='png')
            #                 mplpl.figure()
            #
            #
            #
            #
            #
            #             weights = []
            #
            #             w_pt_list = []
            #
            #
            #             weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(list(df_tmp['rel_v']))))
            #
            #             # tweet_avg_med_var[t_id] = [np.mean(list(df_tmp['rel_v'])), np.median(list(df_tmp['rel_v'])), np.var(list(df_tmp['rel_v']))]
            #             # tweet_avg[t_id] = np.mean(list(df_tmp['rel_v']))
            #             # tweet_med[t_id] = np.median(list(df_tmp['rel_v']))
            #             # tweet_var[t_id] = np.var(list(df_tmp['rel_v']))
            #             #
            #             # tweet_avg_l.append(np.mean(list(df_tmp['rel_v'])))
            #             # tweet_med_l.append(np.median(list(df_tmp['rel_v'])))
            #             # tweet_var_l.append(np.var(list(df_tmp['rel_v'])))
            #             accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
            #             if fig_p==6:
            #
            #                 all_acc.append(accuracy)
            #                 try:
            #                     df_tmp['rel_v'].plot(kind='kde', lw=4, color='c', label='Perceive truth value')
            #                 except:
            #                     print('hmm')
            #
            #                 mplpl.hist(list(df_tmp['rel_v']), weights=weights, color='c')
            #
            #                 mplpl.ylabel('Frequency')
            #                 mplpl.xlabel('Perceive truth value')
            #                 mplpl.title(' Accuracy : ' + str(np.round(accuracy, 3)) + '\n Avg : '
            #                             + str(np.round(np.mean(list(df_tmp['rel_v'])), 3)) + ', Var : '
            #                             + str(np.round(np.var(list(df_tmp['rel_v'])), 3)))
            #                 mplpl.legend(loc="upper right")
            #                 mplpl.xlim([-2, 2])
            #                 mplpl.ylim([0, 1])
            #                 if balance_f == 'balanced':
            #                     pp = remotedir + '/fig/fig_exp1/news_based/balanced/ind_users/' + str(w_id) + '_pt_dist'
            #                 else:
            #                     pp = remotedir + '/fig/fig_exp1/user_based/ind_users/' + str(w_id) + '_pt_dist'
            #                 mplpl.savefig(pp, format='png')
            #                 mplpl.figure()
            #
            #
            #
            #             weights = []
            #
            #             w_pt_list = []
            #
            #             weights.append(np.ones_like(list(df_tmp['err'])) / float(len(list(df_tmp['err']))))
            #
            #             # tweet_avg_med_var[t_id] = [np.mean(list(df_tmp['err'])), np.median(list(df_tmp['err'])),
            #             #                            np.var(list(df_tmp['err']))]
            #             # tweet_avg[t_id] = np.mean(list(df_tmp['err']))
            #             # tweet_med[t_id] = np.median(list(df_tmp['err']))
            #             # tweet_var[t_id] = np.var(list(df_tmp['err']))
            #             #
            #             # tweet_avg_l.append(np.mean(list(df_tmp['err'])))
            #             # tweet_med_l.append(np.median(list(df_tmp['err'])))
            #             # tweet_var_l.append(np.var(list(df_tmp['err'])))
            #             accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
            #
            #             if fig_p==2:
            #                 all_acc.append(accuracy)
            #                 try:
            #                     df_tmp['err'].plot(kind='kde', lw=4, color='y', label='Perception bias value')
            #                 except:
            #                     print('hmm')
            #
            #                 mplpl.hist(list(df_tmp['err']), weights=weights, color='y')
            #
            #                 mplpl.ylabel('Frequency')
            #                 mplpl.xlabel('Perception bias value')
            #                 mplpl.title(' Accuracy : ' + str(np.round(accuracy, 3)) + '\n Avg : '
            #                             + str(np.round(np.mean(list(df_tmp['err'])), 3)) + ', Var : '
            #                             + str(np.round(np.var(list(df_tmp['err'])), 3)))
            #                 mplpl.legend(loc="upper right")
            #                 mplpl.xlim([-2, 2])
            #                 mplpl.ylim([0, 1])
            #                 if balance_f == 'balanced':
            #                     pp = remotedir + '/fig/fig_exp1/news_based/balanced/ind_users/' + str(w_id) + '_pb_dist'
            #                 else:
            #                     pp = remotedir + '/fig/fig_exp1/user_based/ind_users/' + str(w_id) + '_pb_dist'
            #                 mplpl.savefig(pp, format='png')
            #                 mplpl.figure()
            #
            #
            #             weights = []
            #
            #             w_pt_list = []
            #
            #             weights.append(np.ones_like(list(df_tmp['abs_err'])) / float(len(list(df_tmp['abs_err']))))
            #
            #             # tweet_avg_med_var[t_id] = [np.mean(list(df_tmp['abs_err'])), np.median(list(df_tmp['abs_err'])),
            #             #                            np.var(list(df_tmp['abs_err']))]
            #             # tweet_avg[t_id] = np.mean(list(df_tmp['abs_err']))
            #             # tweet_med[t_id] = np.median(list(df_tmp['abs_err']))
            #             # tweet_var[t_id] = np.var(list(df_tmp['abs_err']))
            #             #
            #             # tweet_avg_l.append(np.mean(list(df_tmp['abs_err'])))
            #             # tweet_med_l.append(np.median(list(df_tmp['abs_err'])))
            #             # tweet_var_l.append(np.var(list(df_tmp['abs_err'])))
            #             accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
            #             if fig_p==3:
            #
            #                 all_acc.append(accuracy)
            #                 try:
            #                     df_tmp['abs_err'].plot(kind='kde', lw=4, color='y', label='Absolute perception bias value')
            #                 except:
            #                     print('hmm')
            #
            #                 mplpl.hist(list(df_tmp['abs_err']), weights=weights, color='y')
            #
            #                 mplpl.ylabel('Frequency')
            #                 mplpl.xlabel('Absolute perception bias value')
            #                 mplpl.title(' Accuracy : ' + str(np.round(accuracy, 3)) + '\n Avg : '
            #                             + str(np.round(np.mean(list(df_tmp['abs_err'])), 3)) + ', Var : '
            #                             + str(np.round(np.var(list(df_tmp['abs_err'])), 3)))
            #                 mplpl.legend(loc="upper right")
            #                 mplpl.xlim([-2, 2])
            #                 mplpl.ylim([0, 1])
            #                 if balance_f == 'balanced':
            #                     pp = remotedir + '/fig/fig_exp1/news_based/balanced/ind_users/' + str(w_id) + '_apb_dist'
            #                 else:
            #                     pp = remotedir + '/fig/fig_exp1/user_based/ind_users/' + str(w_id) + '_apb_dist'
            #                 mplpl.savefig(pp, format='png')
            #                 mplpl.figure()
            #
            #
            #
            #
            #             weights = []
            #
            #             w_pt_list = []
            #
            #             weights.append(np.ones_like(list(df_tmp['gull'])) / float(len(list(df_tmp['gull']))))
            #
            #             # tweet_avg_med_var[t_id] = [np.mean(list(df_tmp['gull'])), np.median(list(df_tmp['gull'])),
            #             #                            np.var(list(df_tmp['gull']))]
            #             # tweet_avg[t_id] = np.mean(list(df_tmp['gull']))
            #             # tweet_med[t_id] = np.median(list(df_tmp['gull']))
            #             # tweet_var[t_id] = np.var(list(df_tmp['gull']))
            #             #
            #             # tweet_avg_l.append(np.mean(list(df_tmp['gull'])))
            #             # tweet_med_l.append(np.median(list(df_tmp['gull'])))
            #             # tweet_var_l.append(np.var(list(df_tmp['gull'])))
            #             accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
            #             if fig_p==4:
            #
            #                 all_acc.append(accuracy)
            #                 try:
            #                     df_tmp['gull'].plot(kind='kde', lw=4, color='k', label='Gullibility value')
            #                 except:
            #                     print('hmm')
            #
            #                 mplpl.hist(list(df_tmp['gull']), weights=weights, color='k')
            #
            #                 mplpl.ylabel('Frequency')
            #                 mplpl.xlabel('Gullibility value')
            #                 mplpl.title(' Accuracy : ' + str(np.round(accuracy, 3)) + '\n Avg : '
            #                             + str(np.round(np.mean(list(df_tmp['gull'])), 3)) + ', Var : '
            #                             + str(np.round(np.var(list(df_tmp['gull'])), 3)))
            #                 mplpl.legend(loc="upper right")
            #                 mplpl.xlim([-2, 2])
            #                 mplpl.ylim([0, 1])
            #                 if balance_f == 'balanced':
            #                     pp = remotedir + '/fig/fig_exp1/news_based/balanced/ind_users/' + str(w_id) + '_gull_dist'
            #                 else:
            #                     pp = remotedir + '/fig/fig_exp1/user_based/ind_users/' + str(w_id) + '_gull_dist'
            #                 mplpl.savefig(pp, format='png')
            #                 mplpl.figure()
            #
            #             weights = []
            #
            #             w_pt_list = []
            #
            #             weights.append(np.ones_like(list(df_tmp['cyn'])) / float(len(list(df_tmp['cyn']))))
            #
            #             # tweet_avg_med_var[t_id] = [np.mean(list(df_tmp['cyn'])), np.median(list(df_tmp['cyn'])),
            #             #                            np.var(list(df_tmp['cyn']))]
            #             # tweet_avg[t_id] = np.mean(list(df_tmp['cyn']))
            #             # tweet_med[t_id] = np.median(list(df_tmp['cyn']))
            #             # tweet_var[t_id] = np.var(list(df_tmp['cyn']))
            #             #
            #             # tweet_avg_l.append(np.mean(list(df_tmp['cyn'])))
            #             # tweet_med_l.append(np.median(list(df_tmp['cyn'])))
            #             # tweet_var_l.append(np.var(list(df_tmp['cyn'])))
            #             accuracy = np.sum(df_tmp['acc']) / float(len(df_tmp))
            #             if fig_p==5:
            #
            #                 all_acc.append(accuracy)
            #                 try:
            #                     df_tmp['cyn'].plot(kind='kde', lw=4, color='m', label='Cynicality value')
            #                 except:
            #                     print('hmm')
            #
            #                 mplpl.hist(list(df_tmp['cyn']), weights=weights, color='m')
            #
            #                 mplpl.ylabel('Frequency')
            #                 mplpl.xlabel('Cynicality value')
            #                 mplpl.title(' Accuracy : ' + str(np.round(accuracy, 3)) + '\n Avg : '
            #                             + str(np.round(np.mean(list(df_tmp['cyn'])), 3)) + ', Var : '
            #                             + str(np.round(np.var(list(df_tmp['cyn'])), 3)))
            #                 mplpl.legend(loc="upper right")
            #                 mplpl.xlim([-2, 2])
            #                 mplpl.ylim([0, 1])
            #                 if balance_f == 'balanced':
            #                     pp = remotedir + '/fig/fig_exp1/news_based/balanced/ind_users/' + str(w_id) + '_cyn_dist'
            #                 else:
            #                     pp = remotedir + '/fig/fig_exp1/user_based/ind_users/' + str(w_id) + '_cyn_dist'
            #                 mplpl.savefig(pp, format='png')
            #                 mplpl.figure()
            #
            #
            #
            #
            #
            #     exit()
            # else:
            #
            #     AVG_list = []
            #     print(np.mean(all_acc))
            #     outF = open(remotedir + 'output.txt', 'w')
            #
            #     tweet_all_var = {}
            #     tweet_all_dev_avg = {}
            #     tweet_all_avg = {}
            #     tweet_all_gt_var = {}
            #     tweet_all_dev_avg_l = []
            #     tweet_all_dev_med_l = []
            #     tweet_all_dev_var_l = []
            #     tweet_all_avg_l = []
            #     tweet_all_med_l = []
            #     tweet_all_var_l = []
            #     tweet_all_gt_var_l = []
            #     diff_group_disp_l = []
            #     dem_disp_l = []
            #     rep_disp_l = []
            #
            #     tweet_all_dev_avg = {}
            #     tweet_all_dev_med = {}
            #     tweet_all_dev_var = {}
            #
            #     tweet_all_dev_avg_l = []
            #     tweet_all_dev_med_l = []
            #     tweet_all_dev_var_l = []
            #
            #     tweet_all_abs_dev_avg = {}
            #     tweet_all_abs_dev_med = {}
            #     tweet_all_abs_dev_var = {}
            #
            #     tweet_all_abs_dev_avg_l = []
            #     tweet_all_abs_dev_med_l = []
            #     tweet_all_abs_dev_var_l = []
            #     tweet_all_dev_avg_rnd = {}
            #     tweet_all_abs_dev_avg_rnd = {}
            #
            #     diff_group_disp_dict = {}
            #     if dataset == 'snopes':
            #         data_n = 'sp'
            #         news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
            #         ind_l = [1, 2, 3]
            #     elif dataset == 'politifact':
            #         data_n = 'pf'
            #         news_cat_list = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            #         ind_l = [1, 2, 3]
            #     elif dataset == 'mia':
            #         data_n = 'mia'
            #         news_cat_list = ['rumor', 'non-rumor']
            #         ind_l = [1]
            #
            #     for cat_l in news_cat_list:
            #         outF.write('== ' + str(cat_l) + ' ==\n\n')
            #         print('== ' + str(cat_l) + ' ==')
            #         tweet_dev_avg = {}
            #         tweet_dev_med = {}
            #         tweet_dev_var = {}
            #         tweet_abs_dev_avg = {}
            #         tweet_abs_dev_med = {}
            #         tweet_abs_dev_var = {}
            #
            #         tweet_avg = {}
            #         tweet_med = {}
            #         tweet_var = {}
            #         tweet_gt_var = {}
            #
            #         tweet_dev_avg_rnd = {}
            #         tweet_abs_dev_avg_rnd = {}
            #
            #
            #         tweet_dev_avg_l = []
            #         tweet_dev_med_l = []
            #         tweet_dev_var_l = []
            #         tweet_abs_dev_avg_l = []
            #         tweet_abs_dev_med_l = []
            #         tweet_abs_dev_var_l = []
            #
            #         tweet_avg_l = []
            #         tweet_med_l = []
            #         tweet_var_l = []
            #         tweet_gt_var_l = []
            #         AVG_susc_list = []
            #         AVG_wl_list = []
            #         all_acc = []
            #         AVG_dev_list = []
            #         # for lean in [-1, 0, 1]:
            #
            #             # AVG_susc_list = []
            #             # AVG_wl_list = []
            #             # all_acc = []
            #             # df_m = df_m[df_m['leaning'] == lean]
            #             # if lean == 0:
            #             #     col = 'g'
            #             #     lean_cat = 'neutral'
            #             # elif lean == 1:
            #             #     col = 'b'
            #             #     lean_cat = 'democrat'
            #             # elif lean == -1:
            #             #     col = 'r'
            #             #     lean_cat = 'republican'
            #             # print(lean_cat)
            #         for ind in ind_l:
            #
            #             if balance_f == 'balanced':
            #                 inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_balanced.csv'
            #             else:
            #                 inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final.csv'
            #
            #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '.csv'
            #             df[ind] = pd.read_csv(inp1, sep="\t")
            #             df_w[ind] = pd.read_csv(inp1_w, sep="\t")
            #
            #             df_m = df[ind].copy()
            #             df_mm = df_m.copy()
            #
            #             df_m = df_m[df_m['ra_gt'] == cat_l]
            #             # df_mm = df_m[df_m['ra_gt']==cat_l]
            #             # df_m = df_m[df_m['leaning'] == lean]
            #
            #             groupby_ftr = 'tweet_id'
            #             grouped = df_m.groupby(groupby_ftr, sort=False)
            #             grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()
            #
            #             for t_id in grouped.groups.keys():
            #                 df_tmp = df_m[df_m['tweet_id'] == t_id]
            #
            #                 df_tmp_m = df_mm[df_mm['tweet_id'] == t_id]
            #                 df_tmp_dem = df_tmp_m[df_tmp_m['leaning'] == 1]
            #                 df_tmp_rep = df_tmp_m[df_tmp_m['leaning'] == -1]
            #                 ind_t = df_tmp.index.tolist()[0]
            #                 weights = []
            #                 df_tmp = df_m[df_m['tweet_id'] == t_id]
            #                 ind_t = df_tmp.index.tolist()[0]
            #                 weights = []
            #
            #                 weights.append(np.ones_like(list(df_tmp['rel_v'])) / float(len(df_tmp)))
            #                 val_list = list(df_tmp['rel_v'])
            #                 tweet_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
            #                 tweet_avg[t_id] = np.mean(val_list)
            #                 tweet_med[t_id] = np.median(val_list)
            #                 tweet_var[t_id] = np.var(val_list)
            #                 tweet_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]
            #
            #                 tweet_avg_l.append(np.mean(val_list))
            #                 tweet_med_l.append(np.median(val_list))
            #                 tweet_var_l.append(np.var(val_list))
            #                 tweet_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])
            #
            #
            #
            #
            #                 tweet_all_avg[t_id] = np.mean(val_list)
            #                 tweet_all_var[t_id] = np.var(val_list)
            #                 tweet_all_gt_var[t_id] = df_tmp['rel_gt_v'][ind_t]
            #
            #                 tweet_all_avg_l.append(np.mean(val_list))
            #                 tweet_all_med_l.append(np.median(val_list))
            #                 tweet_all_var_l.append(np.var(val_list))
            #                 tweet_all_gt_var_l.append(df_tmp['rel_gt_v'][ind_t])
            #
            #
            #
            #                 val_list = list(df_tmp['err'])
            #                 abs_var_err = [np.abs(x) for x in val_list]
            #                 tweet_dev_avg_med_var[t_id] = [np.mean(val_list), np.median(val_list), np.var(val_list)]
            #                 tweet_dev_avg[t_id] = np.mean(val_list)
            #                 tweet_dev_med[t_id] = np.median(val_list)
            #                 tweet_dev_var[t_id] = np.var(val_list)
            #
            #                 tweet_dev_avg_l.append(np.mean(val_list))
            #                 tweet_dev_med_l.append(np.median(val_list))
            #                 tweet_dev_var_l.append(np.var(val_list))
            #
            #                 tweet_abs_dev_avg[t_id] = np.mean(abs_var_err)
            #                 tweet_abs_dev_med[t_id] = np.median(abs_var_err)
            #                 tweet_abs_dev_var[t_id] = np.var(abs_var_err)
            #
            #                 tweet_abs_dev_avg_l.append(np.mean(abs_var_err))
            #                 tweet_abs_dev_med_l.append(np.median(abs_var_err))
            #                 tweet_abs_dev_var_l.append(np.var(abs_var_err))
            #
            #
            #                 tweet_all_dev_avg[t_id] = np.mean(val_list)
            #                 tweet_all_dev_med[t_id] = np.median(val_list)
            #                 tweet_all_dev_var[t_id] = np.var(val_list)
            #
            #                 tweet_all_dev_avg_l.append(np.mean(val_list))
            #                 tweet_all_dev_med_l.append(np.median(val_list))
            #                 tweet_all_dev_var_l.append(np.var(val_list))
            #
            #                 tweet_all_abs_dev_avg[t_id] = np.mean(abs_var_err)
            #                 tweet_all_abs_dev_med[t_id] = np.median(abs_var_err)
            #                 tweet_all_abs_dev_var[t_id] = np.var(abs_var_err)
            #
            #                 tweet_all_abs_dev_avg_l.append(np.mean(abs_var_err))
            #                 tweet_all_abs_dev_med_l.append(np.median(abs_var_err))
            #                 tweet_all_abs_dev_var_l.append(np.var(abs_var_err))
            #
            #
            #
            #                 sum_rnd_abs_perc = 0
            #                 sum_rnd_perc = 0
            #                 for val in [-1, -2/float(3), -1/float(3), 0, 1/float(3), 2/float(3),1]:
            #                     sum_rnd_perc+= val - df_tmp['rel_gt_v'][ind_t]
            #                     sum_rnd_abs_perc += np.abs(val - df_tmp['rel_gt_v'][ind_t])
            #                 random_perc = np.abs(sum_rnd_perc / float(7))
            #                 random_abs_perc = sum_rnd_abs_perc / float(7)
            #
            #                 tweet_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
            #                 tweet_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
            #
            #                 tweet_all_dev_avg_rnd[t_id] = np.mean(val_list) / float(random_perc)
            #                 tweet_all_abs_dev_avg_rnd[t_id] = np.mean(abs_var_err) / float(random_abs_perc)
            #
            #         gt_l = []
            #         pt_l = []
            #         disputability_l = []
            #         perc_l = []
            #         abs_perc_l = []
            #         # for t_id in tweet_l_sort:
            #         #     gt_l.append(tweet_gt_var[t_id])
            #         #     pt_l.append(tweet_avg[t_id])
            #         #     disputability_l.append(tweet_var[t_id])
            #         #     perc_l.append(tweet_dev_avg[t_id])
            #         #     abs_perc_l.append(tweet_abs_dev_avg[t_id])
            #
            #
            #
            #         # tweet_l_sort = sorted(tweet_avg, key=tweet_avg.get, reverse=True)
            #         tweet_l_sort = sorted(tweet_var, key=tweet_var.get, reverse=True)
            #         # tweet_l_sort = sorted(tweet_dev_avg, key=tweet_dev_avg.get, reverse=True)
            #         # tweet_l_sort = sorted(tweet_abs_dev_avg, key=tweet_abs_dev_avg.get, reverse=True)
            #         # tweet_l_sort = sorted(tweet_dev_avg_rnd, key=tweet_dev_avg_rnd.get, reverse=True)
            #         # tweet_l_sort = sorted(tweet_abs_dev_avg_rnd, key=tweet_abs_dev_avg_rnd.get, reverse=True)
            #
            #         # tweet_l_sort = sorted(tweet_avg, key=tweet_avg.get, reverse=True)
            #
            #
            #         if dataset == 'snopes':
            #             data_addr = 'snopes'
            #         elif dataset == 'politifact':
            #             data_addr = 'politifact/fig'
            #         elif dataset == 'mia':
            #             data_addr = 'mia/fig'
            #
            #         count = 0
            #         outF.write(
            #             '|| || news || Category|| Perception bias <<BR>> Absolute perception bias||Perception bias <<BR>> Absolute perception bias (rnd)||All rederas judgment dist || Democrats || Republicans || Neutrals ||\n')
            #         # '|| || news || Category|| grouped disputablity||All rederas judgment dist || Democrats || Republicans || Neutrals ||\n')
            #
            #         for t_id in tweet_l_sort:
            #             count+=1
            #             if balance_f=='balanced':
            #                 outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] + '||'
            #                            + str(np.round(diff_group_disp_dict[t_id], 3)) + '||'+ str(tweet_all_dev_avg[t_id]) +'<<BR>>' + str(tweet_all_abs_dev_avg[t_id]) +'||'
            #                            + '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/balanced/' +
            #                            str(t_id) + '_rel_dist| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/balanced/' +
            #                            str(t_id) + '_rel_dist_democrat| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/balanced/' +
            #                            str(t_id) + '_rel_dist_republican| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/balanced/' +
            #                            str(t_id) + '_rel_dist_neutral| alt text| width = 500px}} ||\n')
            #                 # +
            #
            #             else:
            #                 outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] +
            #                            # str(np.round(diff_group_disp_dict[t_id], 3)) +
            #                            '||'+  str(tweet_all_dev_avg[t_id]) +'<<BR>>' + str(tweet_all_abs_dev_avg[t_id])+'||'
            #                             + str(tweet_all_dev_avg_rnd[t_id]) + '<<BR>>' + str(tweet_all_abs_dev_avg_rnd[t_id]) +
            #                             '||{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/' +
            #                            str(t_id) + '_rel_dist| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/' +
            #                            str(t_id) + '_rel_dist_democrat| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/' +
            #                            str(t_id) + '_rel_dist_republican| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/' +
            #                            str(t_id) + '_rel_dist_neutral| alt text| width = 500px}} ||\n')
            #
            #
            #
            #
            #     if dataset == 'snopes':
            #         data_addr = 'snopes'
            #     elif dataset == 'politifact':
            #         data_addr = 'politifact/fig'
            #     elif dataset == 'mia':
            #         data_addr = 'mia/fig'
            #
            #     # tweet_l_sort = sorted(diff_group_disp_dict, key=diff_group_disp_dict.get, reverse=True)
            #     # tweet_l_sort = sorted(tweet_all_avg, key=tweet_all_avg.get, reverse=True)
            #     tweet_l_sort = sorted(tweet_all_var, key=tweet_all_var.get, reverse=True)
            #     # tweet_l_sort = sorted(tweet_all_dev_avg, key=tweet_all_dev_avg.get, reverse=True)
            #     # tweet_l_sort = sorted(tweet_all_abs_dev_avg, key=tweet_all_abs_dev_avg.get, reverse=True)
            #     # tweet_l_sort = sorted(tweet_all_dev_avg_rnd, key=tweet_all_dev_avg_rnd.get, reverse=True)
            #     # tweet_l_sort = sorted(tweet_all_dev_avg_rnd, key=tweet_all_dev_avg_rnd.get, reverse=True)
            #
            #     # tweet_l_sort = sorted(tweet_all_var, key=tweet_all_var.get, reverse=True)
            #     # tweet_l_sort = sorted(tweet_all_dev_avg, key=tweet_all_dev_avg.get, reverse=True)
            #
            #     tweet_napb_dict_high_disp = {}
            #     tweet_napb_dict_low_disp = {}
            #     for t_id in tweet_l_sort[:20]:
            #         # tweet_napb_dict_high_disp[t_id] = tweet_all_abs_dev_avg_rnd[t_id]
            #         tweet_napb_dict_high_disp[t_id] = tweet_all_abs_dev_avg[t_id]
            #
            #     for t_id in tweet_l_sort[-20:]:
            #         # tweet_napb_dict_low_disp[t_id] = tweet_all_abs_dev_avg_rnd[t_id]
            #         tweet_napb_dict_low_disp[t_id] = tweet_all_abs_dev_avg[t_id]
            #
            #     kk = 0
            #
            #     for tweet_dict in [tweet_napb_dict_high_disp, tweet_napb_dict_low_disp]:
            #         if kk==0:
            #             tweet_l_sort = sorted(tweet_dict, key=tweet_dict.get, reverse=False)
            #         else:
            #             tweet_l_sort = sorted(tweet_dict, key=tweet_dict.get, reverse=True)
            #
            #         kk+=1
            #         count = 0
            #         outF.write(
            #             '|| || news || Category|| Perception bias <<BR>> Absolute perception bias||Perception bias <<BR>> Absolute perception bias (rnd)||All rederas judgment dist || Democrats || Republicans || Neutrals ||\n')
            #         for t_id in tweet_l_sort:
            #             count += 1
            #             # ind_t = df_tmp_m[df_tmp_m['tweet_id']=t_id].index.tolist()
            #             if balance_f == 'balanced':
            #                 outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] + '||'
            #                            + str(np.round(diff_group_disp_dict[t_id], 3)) + '||' +
            #                            str(tweet_all_dev_avg[t_id]) +'<<BR>>' + str(tweet_all_abs_dev_avg[t_id]) +'||'+
            #                            str(tweet_all_dev_avg_rnd[t_id]) +'<<BR>>' + str(tweet_all_abs_dev_avg_rnd[t_id]) +'||'+
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/balanced/' +
            #                            str(t_id) + '_rel_dist| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/balanced/' +
            #                            str(t_id) + '_rel_dist_democrat| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/balanced/' +
            #                            str(t_id) + '_rel_dist_republican| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/balanced/' +
            #                            str(t_id) + '_rel_dist_neutral| alt text| width = 500px}} ||\n')
            #                 # +
            #                 #            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig_exp1/news_based/balanced/' +
            #                 #            str(t_id) + '_p_susc_dist| alt text| width = 500px}} ||\n')
            #
            #             else:
            #                 outF.write('||' + str(count) + '||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id] + '||' +
            #                            str(tweet_all_dev_avg[t_id]) + '<<BR>>' + str(tweet_all_abs_dev_avg[t_id]) + '||' +
            #                            str(tweet_all_dev_avg_rnd[t_id]) +'<<BR>>' + str(tweet_all_abs_dev_avg_rnd[t_id]) +'||'+
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/' +
            #                            str(t_id) + '_rel_dist| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/' +
            #                            str(t_id) + '_rel_dist_democrat| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/' +
            #                            str(t_id) + '_rel_dist_republican| alt text| width = 500px}} ||' +
            #                            '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/' + data_addr + '/fig_exp1/news_based/leaning_demographic/' +
            #                            str(t_id) + '_rel_dist_neutral| alt text| width = 500px}} ||\n')
            #             # +
            #             # '{{http://www.mpi-sws.mpg.de/~babaei/result-fig/reliability_news/snopes/fig_exp1/news_based/' +
            #             # str(t_id) + '_p_susc_dist| alt text| width = 500px}} ||\n')



    if args.t == "scratch_bounus":
        import hashlib

        algo = 'sha1'
        data = 'twitter app survey crowd signal.'

        dataset = 'snopes'
        wrk_corr = collections.defaultdict(int)
        if dataset == 'snopes':
            query1 = "select workerid, ra from mturk_sp_claim_incentive_correct_response_exp1_1"
        # elif dataset == 'snopes_ssi':
            # query1 = "select workerid, count(*) from mturk_sp_claim_ssi_response_exp" + str(experiment) + "_recovery_full group by workerid;"
            # query1 = "select workerid, ra from mturk_sp_claim_incentive_10_correct_response_exp1_1;"
        # elif dataset == 'snopes_nonpol':
        #     query1 = "select workerid, count(*) from mturk_sp_claim_nonpol_response_exp1_recovery group by workerid;"
        # elif dataset == 'politifact':
        #     query1 = "select workerid, count(*) from mturk_pf_claim_response_exp1_" + str(
        #         experiment) + "_recovery group by workerid;"
        # elif dataset == 'mia':
        #     query1 = "select workerid, count(*) from mturk_m_claim_response_exp1_recovery group by workerid;"

        cursor.execute(query1)
        res_exp2 = cursor.fetchall()
        for el in res_exp2:
            if el[1]==1:
                wrk_corr[el[0]]+=1



        h = hashlib.new(algo)
        for w_id in wrk_corr:
            h.update(data+str(w_id))

            result = h.hexdigest()
            print(result + '----> ' + str(wrk_corr[w_id]) + '    : workerId  : ' + str(w_id))

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_analysis_plan":


        remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
        output = open(remotedir + '/comparing_diff_surveys.txt', 'w')
        # output.write('||Surveys || Distribution of answering|| Chi square of dist of answering survey 1 <<BR>> survey 2'
        #              '|| Chi square test (dependency) of survey 1 and 2 ||Chi square proportion of correct answering|| Chi square of TPB of survey 1<<BR>> survey 2'
        #              '|| Chi square test(dependency) of survey 1 and 2||MWU test to compare TPB|| Pearson Corr of TPB|| Spearman Corr of TPB||\n')



        output.write('||Surveys || Distribution of answering|| Chi square of dist of answering survey 1 <<BR>> survey 2'
                     '|| Chi square test (dependency) of survey 1 and 2 ||Chi square proportion of correct answering'
                     '|| Chi square test(dependency) of survey 1 and 2|| Pearson Corr of TPB|| Spearman Corr of TPB||\n')

        # output.write('|| Survey || pearson corr between TPB and Disp|| pearson corr between TPB and Disp||\n')

        data_n = 'sp';ind=1
        inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
        inp2 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
        inp3 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
        inp4 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'

        df_gt = pd.read_csv(inp1, sep='\t')
        df_gt_exp = pd.read_csv(inp2, sep='\t')
        df_inc_gt = pd.read_csv(inp3, sep='\t')
        df_inc_gt_exp = pd.read_csv(inp4, sep='\t')



        inp1 = remotedir + 'gt.csv'
        inp2 = remotedir + 'gt_experts.csv'
        inp3 = remotedir + 'incentivize_gt.csv'
        inp4 = remotedir + 'incentivize_gt_experts.csv'

        df_gt_amt = pd.read_csv(inp1,sep=',')
        df_gt_exp_amt = pd.read_csv(inp2, sep=',')
        df_inc_gt_amt = pd.read_csv(inp3, sep=',')
        df_inc_gt_exp_amt = pd.read_csv(inp4, sep=',')


        w_id_gt = df_gt_amt['WorkerId']
        w_id_gt_exp = df_gt_exp_amt['WorkerId']
        w_id_inc_gt = df_inc_gt_amt['WorkerId']
        w_id_inc_gt_exp = df_inc_gt_exp_amt['WorkerId']


        w1_gt_l = list(set(w_id_gt) & set(w_id_gt_exp))
        w2_gt_l = list(set(w_id_gt) & set(w_id_inc_gt_exp))

        w1_gt_exp_l = list(set(w_id_gt_exp) & set(w_id_gt))
        w2_gt_exp_l = list(set(w_id_gt_exp) & set(w_id_inc_gt))



        w1_inc_gt_l = list(set(w_id_inc_gt) & set(w_id_gt_exp))
        w2_inc_gt_l = list(set(w_id_inc_gt) & set(w_id_inc_gt_exp))

        w1_inc_gt_exp_l = list(set(w_id_inc_gt_exp) & set(w_id_gt))
        w2_inc_gt_exp_l = list(set(w_id_inc_gt_exp) & set(w_id_inc_gt))



        inters_gt_l = []
        inters_gt_exp_l = []
        inters_inc_gt_l = []
        inters_inc_gt_exp_l = []

        inters_gt_l+=w1_gt_l
        inters_gt_l+=w2_gt_l


        inters_gt_exp_l+=w1_gt_exp_l
        inters_gt_exp_l+=w2_gt_exp_l


        inters_inc_gt_l+=w1_inc_gt_l
        inters_inc_gt_l+=w2_inc_gt_l


        inters_inc_gt_exp_l+=w1_inc_gt_exp_l
        inters_inc_gt_exp_l+=w2_inc_gt_exp_l



        import hashlib


        cod_id_gt_list = []
        for w_id in inters_gt_l:
            try:
                ind_tmp = df_gt_amt[df_gt_amt['WorkerId']==w_id].index.tolist()[0]
                cod_id = df_gt_amt['Answer.Q2age'][ind_tmp]
                cod_id_gt_list.append(cod_id)
            except:
                continue

        w_id_gt_filter = []
        for w_id in range(0,1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_gt_list:
                w_id_gt_filter.append(w_id+1000)


        cod_id_gt_exp_list = []
        for w_id in inters_gt_exp_l:
            try:
                ind_tmp = df_gt_exp_amt[df_gt_exp_amt['WorkerId']==w_id].index.tolist()[0]
                cod_id = df_gt_exp_amt['Answer.Q2age'][ind_tmp]
                cod_id_gt_exp_list.append(cod_id)
            except:
                continue

        w_id_gt_exp_filter = []
        for w_id in range(0,1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_gt_exp_list:
                w_id_gt_exp_filter.append(w_id+1000)




        cod_id_inc_gt_list = []
        for w_id in inters_inc_gt_l:
            try:
                ind_tmp = df_inc_gt_amt[df_inc_gt_amt['WorkerId']==w_id].index.tolist()[0]
                cod_id = df_inc_gt_amt['Answer.Q2age'][ind_tmp]
                cod_id_inc_gt_list.append(cod_id)
            except:
                continue

        w_id_inc_gt_filter = []
        for w_id in range(0,1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_inc_gt_list:
                w_id_inc_gt_filter.append(w_id+1000)




        cod_id_inc_gt_exp_list = []
        for w_id in inters_inc_gt_exp_l:
            try:
                ind_tmp = df_inc_gt_exp_amt[df_inc_gt_exp_amt['WorkerId']==w_id].index.tolist()[0]
                cod_id = df_inc_gt_exp_amt['Answer.Q2age'][ind_tmp]
                cod_id_inc_gt_exp_list.append(cod_id)
            except:
                continue

        w_id_inc_gt_exp_filter = []
        for w_id in range(0,1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_inc_gt_exp_list:
                w_id_inc_gt_exp_filter.append(w_id+1000)




        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        # exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'

        # inp10 = remotedir + 'amt_answers_' + 'sp_gt_10' + '_claims_exp' + str(1) + '_final_weighted_gt.csv'
        inp = remotedir + 'amt_answers_' + 'sp' + '_claims_exp' + str(1) + '_final_weighted.csv'
        df = pd.read_csv(inp, sep="\t")

        claims_list_all = set(df['tweet_id'])

        data_c = 0
        data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt', 'snopes_gt_experts'
            , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer']
        # data_list1 = ['snopes',  'snopes_ssi', 'snopes_incentive', 'snopes_gt', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2', 'snopes_incentive_notimer', 'snopes_noincentive_timer']

        # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt','snopes_gt_2','snopes_gt_3', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer','snopes_gt_10']
        # data_list1 = ['snopes',  'snopes_ssi', 'snopes_incentive', 'snopes_gt','snopes_gt_2','snopes_gt_3',  'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2', 'snopes_incentive_notimer', 'snopes_noincentive_timer','snopes_gt_10']
        # #

        # data_list = ['snopes_gt', 'snopes_gt_2', 'snopes_gt_3' ]
        # data_list1 = ['snopes_gt',  'snopes_gt_2', 'snopes_gt_3' ]


        TPB_dict = collections.defaultdict(list)

        Disp_dict = collections.defaultdict(list)
        for dataset1 in data_list:#, 'snopes_incentive_10', 'snopes_2', 'snopes_ssi', 'snopes_nonpol', 'snopes','snopes_gt','snope_gt_experts', 'snopes_gt_incentive', 'snopes_gt_incentive_experts']:
            # dataset1 = dataset
            data_c+=1
            if dataset1 == 'snopes' or dataset1 == 'snopes_ssi' or dataset1 == 'snopes_incentive' \
                    or dataset1 == 'snopes_gt'or dataset1 == 'snopes_gt_1'or dataset1 == 'snopes_gt_experts'or dataset1 == 'snopes_gt_incentive'\
                    or dataset1 == 'snopes_gt_incentive_experts'or dataset1 == 'snopes_2' or dataset1 == 'snopes_incentive_notimer' \
                    or dataset1 == 'snopes_noincentive_timer' or dataset1 == 'snopes_gt_10'or dataset1 == 'snopes_gt_2'or dataset1 == 'snopes_gt_3':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')



            print('-------------------' + dataset1 + '------------------')

            if dataset1 == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset1 == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'

            elif dataset1 == 'snopes_gt':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt'
            elif dataset1 == 'snopes_gt_1':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_1'
            elif dataset1 == 'snopes_gt_2':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_2'
            elif dataset1 == 'snopes_gt_3':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_3'
            elif dataset1 == 'snopes_gt_10':
                data_n = 'sp_gt_10'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_10'

            elif dataset1 == 'snopes_gt_experts':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_exp'

            elif dataset1 == 'snopes_gt_incentive':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_inc'


            elif dataset1 == 'snopes_gt_incentive_experts':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_inc_exp'


            elif dataset1 == 'snopes_incentive_notimer':
                data_n = 'sp_incentive_notimer'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive_notimer'


            elif dataset1 == 'snopes_noincentive_timer':

                data_n = 'sp_noincentive_timer'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_noincentive_timer'


            elif dataset1 == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            # elif dataset == 'snopes_incentive_10':
            #     data_n = 'sp_incentive_10'
            #     data_addr = 'snopes'
            #     ind_l = [1, 2, 3]
            #     ind_l = [1]
            #     data_name = 'Snopes_incentive_10'

            elif dataset1 == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_ssi'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            news_cat_list_tf = [4, 2, 3, 1]
            t_f_dict_len = collections.defaultdict(int)
            t_f_dict = {}

            # if dataset == 'snopes' or dataset == 'snopes_nonpol' or dataset == 'snopes_ssi' or dataset == 'snopes_incentive' or dataset == 'snopes_incentive_10' or dataset == 'snopes_2':
            #     news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
            #     news_cat_list_v = [-1, -.5, 0, 0.5, 1]


            w_fnb_dict = collections.defaultdict()
            w_fpb_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()
            gt_acc = collections.defaultdict()
            # for cat in news_cat_list_v:
            #     gt_acc[cat] = [0] * (len(news_cat_list_t_f))
            weight_list = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
            # weight_list = [-2.5, -0.84, 0.25, 1, -1.127, 1.1, 1.05]
            # weight_list = [-2.36, -0.73, 0.53, 0.87, -0.87, 0.93, 1.53]
            pt_list = []
            gt_list = []
            pp = 0
            pf = 0
            tpp = 0
            tpf = 0
            chi_sq_l = []
            f_filter = []
            ind_l = [1]
            dist_list1 = []

            for ind in ind_l:

                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted.csv'
                # df[ind] = pd.read_csv(inp1, sep="\t")


                if dataset1=='snopes_gt':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted_gt.csv'
                    f_filter = w_id_gt_filter
                elif dataset1 == 'snopes_gt_1':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp1_2_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp1_2_weighted_gt.csv'
                    # f_filter = w_id_gt_filter
                elif dataset1 == 'snopes_gt_2':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp2_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp2_weighted_gt.csv'
                elif dataset1 == 'snopes_gt_3':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp3_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp3_weighted_gt.csv'
                    # f_filter = w_id_gt_filter
                elif dataset1 == 'snopes_gt_10':
                    inp1 = remotedir + 'amt_answers_sp_gt_10_claims_exp' + str(ind) + '_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_sp_gt_10_claims_exp' + str(
                        ind) + '_weighted_gt.csv'
                elif dataset1=='snopes_gt_experts':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted_gt_experts.csv'
                    f_filter = w_id_gt_exp_filter

                elif dataset1=='snopes_gt_incentive':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_weighted_gt.csv'

                    f_filter = w_id_inc_gt_filter

                elif dataset1=='snopes_gt_incentive_experts':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_weighted_gt_experts.csv'
                    f_filter = w_id_inc_gt_exp_filter

                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                worker_id_list = set(df_m['worker_id'])
                filter_l = list(worker_id_list - set(f_filter))
                df_m=df_m[df_m['worker_id'].isin(filter_l)]



                df_tt = df_m[df_m['tweet_id']==2001]
                df_t2 = df_tt[df_tt['ra']<4]
                stp_workers1 = df_t2['worker_id']

                df_tt = df_m[df_m['tweet_id']==2002]
                df_t2 = df_tt[df_tt['ra']>2]
                stp_workers2 = df_t2['worker_id']


                # worker_id_list = set(df_m['worker_id'])
                # filter_l = list(worker_id_list - set(list(stp_workers1) + list(stp_workers2)))
                # df_m=df_m[df_m['worker_id'].isin(filter_l)]

                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                # df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # tmp_list = df_m[df_m['ra'] == 1].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[0]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 2].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[1]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 3].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[2]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 4].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[3]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 5].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[4]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 6].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[5]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 7].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[6]] * len(tmp_list)

                # df_mm = df_m[df_m['tweet_id'].isin(claims_list_all)]
                df_mm = df_m[df_m['tweet_id'].isin(grouped.groups.keys())]



                if 'gt' in dataset1:
                    dist_list1.append(len(df_mm[df_mm['ra'] == 1]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 2]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 3]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 4]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 5]))

                else:
                    dist_list1.append(len(df_mm[df_mm['ra'] == 1]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 2]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 3]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 4]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 5]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 6]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 7]))




                # for t_id in grouped.groups.keys():
                for t_id in list(claims_list_all):
                    # if t_id == 1027:
                    print (t_id)
                    # if t_id==2001:

                    if t_id>=2000:
                        continue
                    pp_tmp = 0
                    pf_tmp = 0
                    tpp_tmp = 0
                    tpf_tmp = 0

                    df_tmp = df_m[df_m['tweet_id'] == t_id]
                    pt = np.mean(df_tmp['rel_v_b'])
                    # pt = np.mean(df_tmp['acc'])
                    gt = list(df_tmp['rel_gt_v'])[0]
                    if gt!=0:
                    #     continue
                        if gt > 0:
                            if pt > 0:
                                pp += 1
                            tpp += 1
                        if gt < 0:
                            if pt < 0:
                                pf += 1
                            tpf += 1

                        for el_val in list(df_tmp['rel_v_b']):
                            if gt > 0:
                                if el_val > 0:
                                    pp_tmp += 1
                                tpp_tmp += 1
                            if gt < 0:
                                if el_val < 0:
                                    pf_tmp += 1
                                tpf_tmp += 1

                        if gt>0:
                            chi_sq_l.append(pp_tmp)
                            chi_sq_l.append(tpp_tmp - pp_tmp)
                        elif gt<0:
                            chi_sq_l.append(pf_tmp)
                            chi_sq_l.append(tpf_tmp - pf_tmp)


                    val_list = list(df_tmp['rel_v'])
                    val_list_ra = list(df_tmp['ra'])
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_avg_l.append(np.var(val_list))


                    val_list1 = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list1]

                    tweet_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_dev_med[t_id] = np.median(abs_var_err)
                    tweet_dev_var[t_id] = np.var(abs_var_err)
                    tweet_dev_avg_l.append(np.mean(abs_var_err))


                TPB_dict[dataset1] = tweet_dev_avg_l
                Disp_dict[dataset1] = tweet_avg_l
                print(np.corrcoef(tweet_dev_avg_l,tweet_avg_l))
                print(scipy.stats.spearmanr(tweet_dev_avg_l, tweet_avg_l))
                print(pp / float(tpp))
                print(pp)
                print(pf / float(tpf))
                print(pf)
                # print()
        #         TPB_dict[dataset1] = tweet_dev_avg_l
        #
        #
        #
        # index_list_m = range(len(TPB_dict[data_list[0]]+TPB_dict[data_list[1]]+TPB_dict[data_list[2]]))
        # data_list = ['snopes_gt','snopes_gt_2','snopes_gt_3']
        #
        # df_w = pd.DataFrame({'sp_tpb': Series(TPB_dict[data_list[0]]+TPB_dict[data_list[1]]+TPB_dict[data_list[2]], index=index_list_m),
        #                      'sp_disp': Series(Disp_dict[data_list[0]]+Disp_dict[data_list[1]]+Disp_dict[data_list[2]], index=index_list_m),})
        #
        # print(remotedir)
        # df_w.to_csv(remotedir + 'TPB_Disp_diff_surveys_sp150.csv',
        #             columns=df_w.columns, sep=",", index=False)



        # index_list_m = range(len(TPB_dict[data_list[0]]))
        # data_list = ['snopes_gt','snopes_gt_2','snopes_gt_3']
        #
        # df_w = pd.DataFrame({data_list[0]+'_tpb': Series(TPB_dict[data_list[0]], index=index_list_m),
        #                      data_list[0]+ '_disp': Series(Disp_dict[data_list[0]], index=index_list_m),
        #                      data_list[1] + '_tpb': Series(TPB_dict[data_list[1]], index=index_list_m),
        #                      data_list[1] + '_disp': Series(Disp_dict[data_list[1]], index=index_list_m),
        #                      data_list[2]+'_tpb': Series(TPB_dict[data_list[2]], index=index_list_m),
        #                      data_list[2]+ '_disp': Series(Disp_dict[data_list[2]], index=index_list_m),})
        #
        # print(remotedir)
        # df_w.to_csv(remotedir + 'TPB_Disp_diff_surveys_sp1_2_3.csv',
        #             columns=df_w.columns, sep=",", index=False)



        index_list_m = range(len(TPB_dict[data_list[0]]))
        # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt','snopes_gt_1','snopes_gt_1','snopes_gt_2','snopes_gt_3', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer', 'snopes_gt_10']

        data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt', 'snopes_gt_experts'
            , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer']
        df_w = pd.DataFrame({data_list[0]+'_tpb': Series(TPB_dict[data_list[0]], index=index_list_m),
                             data_list[0]+ '_disp': Series(Disp_dict[data_list[0]], index=index_list_m),
                             data_list[1]+'_tpb': Series(TPB_dict[data_list[1]], index=index_list_m),
                             data_list[1]+ '_disp': Series(Disp_dict[data_list[1]], index=index_list_m),
                             data_list[2]+'_tpb': Series(TPB_dict[data_list[2]], index=index_list_m),
                             data_list[2]+ '_disp': Series(Disp_dict[data_list[2]], index=index_list_m),
                             data_list[3]+'_tpb': Series(TPB_dict[data_list[3]], index=index_list_m),
                             data_list[3]+ '_disp': Series(Disp_dict[data_list[3]], index=index_list_m),
                             data_list[4]+'_tpb': Series(TPB_dict[data_list[4]], index=index_list_m),
                             data_list[4]+ '_disp': Series(Disp_dict[data_list[4]], index=index_list_m),
                             data_list[5]+'_tpb': Series(TPB_dict[data_list[5]], index=index_list_m),
                             data_list[5]+ '_disp': Series(Disp_dict[data_list[5]], index=index_list_m),
                             data_list[6]+'_tpb': Series(TPB_dict[data_list[6]], index=index_list_m),
                             data_list[6]+ '_disp': Series(Disp_dict[data_list[6]], index=index_list_m),
                             data_list[7]+'_tpb': Series(TPB_dict[data_list[7]], index=index_list_m),
                             data_list[7]+ '_disp': Series(Disp_dict[data_list[7]], index=index_list_m),
                             data_list[8]+'_tpb': Series(TPB_dict[data_list[8]], index=index_list_m),
                             data_list[8]+ '_disp': Series(Disp_dict[data_list[8]], index=index_list_m),
                             data_list[9]+'_tpb': Series(TPB_dict[data_list[9]], index=index_list_m),
                             data_list[9]+ '_disp': Series(Disp_dict[data_list[9]], index=index_list_m),})

        print(remotedir)
        df_w.to_csv(remotedir + 'TPB_Disp_diff_surveys_50claims.csv',
                    columns=df_w.columns, sep=",", index=False)
        #


        exit()
                    # index_list_m = range(len(TPB_dict[data_list[0]]))
        # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer']
        #
        # df_w = pd.DataFrame({data_list[0]: Series(TPB_dict[data_list[0]], index=index_list_m),
        #                      data_list[1]: Series(TPB_dict[data_list[1]], index=index_list_m),
        #                      data_list[2]: Series(TPB_dict[data_list[2]], index=index_list_m),
        #                      data_list[3]: Series(TPB_dict[data_list[3]], index=index_list_m),
        #                      data_list[4]: Series(TPB_dict[data_list[4]], index=index_list_m),
        #                      data_list[5]: Series(TPB_dict[data_list[5]], index=index_list_m),
        #                      data_list[6]: Series(TPB_dict[data_list[6]], index=index_list_m),
        #                      data_list[7]: Series(TPB_dict[data_list[7]], index=index_list_m),
        #                      data_list[8]: Series(TPB_dict[data_list[8]], index=index_list_m),
        #                      data_list[9]: Series(TPB_dict[data_list[9]], index=index_list_m), })
        #
        # print(remotedir)
        # df_w.to_csv(remotedir + 'TPB_diff_surveys_1.csv',
        #             columns=df_w.columns, sep=",", index=False)
            ##################################################
            # print(pp / float(tpp))
            # print(pp)
            # print(pf / float(tpf))
            # print(pf)
            # output.write('||' + dataset1 )
            # output.write('|| ' + str(np.corrcoef(tweet_avg_l, tweet_dev_avg_l)[1]))
            # output.write('|| ' + str(scipy.stats.spearmanr(tweet_avg_l, tweet_dev_avg_l))+ '||\n')
            #
            # news_cat_list = ['pants-fire', 'false', 'mostly_false', 'half-true', 'mostly-true', 'true']
            # news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
            #
            #
            # for dataset2 in data_list1[data_c:]:#, 'snopes_incentive_10', 'snopes_2', 'snopes_ssi', 'snopes_nonpol', 'snopes','mia', 'mia', 'politifact']:
            #
            #     # dataset2 = dataset
            #     print('##############' + dataset2+'##############')
            #     if dataset2 == 'snopes' or dataset2 == 'snopes_ssi' or dataset2 == 'snopes_incentive' or dataset2 == 'snopes_incentive_10' \
            #             or dataset2 == 'snopes_gt' or dataset2 == 'snopes_gt_experts' or dataset2 == 'snopes_gt_incentive' \
            #             or dataset2 == 'snopes_gt_incentive_experts' or dataset2 == 'snopes_2' or dataset2 == 'snopes_incentive_notimer' \
            #             or dataset2 == 'snopes_noincentive_timer' or dataset2 == 'snopes_gt_10':
            #         claims_list = []
            #         remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
            #         inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
            #         news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
            #         news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']
            #
            #         sample_tweets_exp1 = []
            #
            #         tweet_txt_dict = {}
            #         tweet_date_dict = {}
            #         tweet_lable_dict = {}
            #         tweet_publisher_dict = {}
            #         print(inp_all)
            #
            #         for i in range(0, 5):
            #             df_cat = news_cat_list[i]
            #             df_cat_f = news_cat_list_f[i]
            #             inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
            #             cat_count = 0
            #             for line in inF:
            #                 claims_list.append(line)
            #                 cat_count += 1
            #
            #         for line in claims_list:
            #             line_splt = line.split('<<||>>')
            #             publisher_name = int(line_splt[2])
            #             tweet_txt = line_splt[3]
            #             tweet_id = publisher_name
            #             cat_lable = line_splt[4]
            #             dat = line_splt[5]
            #             dt_splt = dat.split(' ')[0].split('-')
            #             m_day = int(dt_splt[2])
            #             m_month = int(dt_splt[1])
            #             m_year = int(dt_splt[0])
            #             m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
            #             tweet_txt_dict[tweet_id] = tweet_txt
            #             tweet_date_dict[tweet_id] = m_date
            #             tweet_lable_dict[tweet_id] = cat_lable
            #             # outF = open(remotedir + 'table_out.txt', 'w')
            #
            #     if dataset2 == 'snopes':
            #         data_n = 'sp'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes'
            #     elif dataset2 == 'snopes_2':
            #         data_n = 'sp_2'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_2'
            #
            #     elif dataset2 == 'snopes_gt':
            #         data_n = 'sp'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_gt'
            #     elif dataset2 == 'snopes_gt_10':
            #         data_n = 'sp_gt_10'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_gt_10'
            #
            #     elif dataset2 == 'snopes_gt_experts':
            #         data_n = 'sp'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_gt_exp'
            #
            #     elif dataset2 == 'snopes_gt_inc':
            #         data_n = 'sp'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_gt_incentive'
            #
            #
            #     elif dataset2 == 'snopes_gt_incentive_experts':
            #         data_n = 'sp'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_gt_inc_exp'
            #
            #
            #     elif dataset2 == 'snopes_incentive_notimer':
            #         data_n = 'sp_incentive_notimer'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_incentive_notimer'
            #
            #
            #     elif dataset2 == 'snopes_noincentive_timer':
            #
            #         data_n = 'sp_noincentive_timer'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_noincentive_timer'
            #
            #
            #     elif dataset2 == 'snopes_incentive':
            #         data_n = 'sp_incentive'
            #         data_addr = 'snopes'
            #         ind_l = [1, 2, 3]
            #         ind_l = [1]
            #         data_name = 'Snopes_incentive'
            #
            #     # elif dataset == 'snopes_incentive_10':
            #     #     data_n = 'sp_incentive_10'
            #     #     data_addr = 'snopes'
            #     #     ind_l = [1, 2, 3]
            #     #     ind_l = [1]
            #     #     data_name = 'Snopes_incentive_10'
            #
            #     elif dataset2 == 'snopes_ssi':
            #         data_n = 'sp_ssi'
            #         data_addr = 'snopes'
            #         ind_l = [1]
            #         data_name = 'Snopes_ssi'
            #
            #     df = collections.defaultdict()
            #     df_w = collections.defaultdict()
            #     tweet_avg_med_var = collections.defaultdict(list)
            #     tweet_dev_avg_med_var = collections.defaultdict(list)
            #     tweet_dev_avg2 = {}
            #     tweet_dev_med2 = {}
            #     tweet_dev_var2 = {}
            #     tweet_avg2 = {}
            #     tweet_med2 = {}
            #     tweet_var2 = {}
            #     tweet_gt_var = {}
            #
            #     tweet_dev_avg_l2 = []
            #     tweet_dev_med_l2 = []
            #     tweet_dev_var_l2 = []
            #     tweet_avg_l2 = []
            #     tweet_med_l2 = []
            #     tweet_var_l2 = []
            #     tweet_gt_var_l = []
            #     avg_susc = 0
            #     avg_gull = 0
            #     avg_cyn = 0
            #
            #     tweet_abs_dev_avg2 = {}
            #     tweet_abs_dev_med2 = {}
            #     tweet_abs_dev_var2 = {}
            #
            #
            #     tweet_abs_dev_avg_rnd2 = {}
            #     tweet_dev_avg_rnd2 = {}
            #
            #     tweet_skew = {}
            #     tweet_skew_l = []
            #
            #
            #
            #     tweet_kldiv_group = collections.defaultdict()
            #
            #     news_cat_list_tf = [4, 2, 3, 1]
            #     t_f_dict_len = collections.defaultdict(int)
            #     t_f_dict = {}
            #
            #     # if dataset == 'snopes' or dataset == 'snopes_nonpol' or dataset == 'snopes_ssi' or dataset == 'snopes_incentive' or dataset == 'snopes_incentive_10' or dataset == 'snopes_2':
            #     #     news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
            #     #     news_cat_list_v = [-1, -.5, 0, 0.5, 1]
            #
            #     w_fnb_dict = collections.defaultdict()
            #     w_fpb_dict = collections.defaultdict()
            #     w_apb_dict = collections.defaultdict()
            #     gt_acc = collections.defaultdict()
            #     # for cat in news_cat_list_v:
            #     #     gt_acc[cat] = [0] * (len(news_cat_list_t_f))
            #     # weight_list = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
            #     # weight_list = [-2.5, -0.84, 0.25, 1, -1.127, 1.1, 1.05]
            #     # weight_list = [-2.36, -0.73, 0.53, 0.87, -0.87, 0.93, 1.53]
            #     pt_list = []
            #     gt_list = []
            #     pp2 = 0
            #     pf2 = 0
            #     tpp2 = 0
            #     tpf2 = 0
            #     chi_sq_l2=[]
            #     f_filter = []
            #     ind_l = [1]
            #     dist_list2 = []
            #
            #
            #     for ind in ind_l:
            #
            #         inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
            #         inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted.csv'
            #         # df[ind] = pd.read_csv(inp1, sep="\t")
            #
            #
            #         if dataset2=='snopes_gt':
            #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
            #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted_gt.csv'
            #             f_filter = w_id_gt_filter
            #         elif dataset2 == 'snopes_gt_10':
            #             inp1 = remotedir + 'amt_answers_sp_gt_10_claims_exp' + str(
            #                 ind) + '_final_weighted_gt.csv'
            #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(
            #                 ind) + '_weighted_gt.csv'
            #             f_filter = w_id_gt_filter
            #         elif dataset2=='snopes_gt_experts':
            #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
            #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted_gt_experts.csv'
            #             f_filter = w_id_gt_exp_filter
            #
            #         elif dataset2=='snopes_gt_incentive':
            #             inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt.csv'
            #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_weighted_gt.csv'
            #
            #             f_filter = w_id_inc_gt_filter
            #
            #         elif dataset2=='snopes_gt_incentive_experts':
            #             inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
            #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_weighted_gt_experts.csv'
            #             f_filter = w_id_inc_gt_exp_filter
            #
            #         df[ind] = pd.read_csv(inp1, sep="\t")
            #         # df_w[ind] = pd.read_csv(inp1_w, sep="\t")
            #
            #         df_m = df[ind].copy()
            #
            #         worker_id_list = set(df_m['worker_id'])
            #         filter_l = list(worker_id_list - set(f_filter))
            #         df_m=df_m[df_m['worker_id'].isin(filter_l)]
            #
            #         df_mm=df_m[df_m['tweet_id'].isin(claims_list_all)]
            #
            #
            #         if 'gt' in dataset2:
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 1]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 2]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 3]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 4]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 5]))
            #
            #         else:
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 1]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 2]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 3]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 4]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 5]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 6]))
            #             dist_list2.append(len(df_mm[df_mm['ra'] == 7]))
            #
            #
            #
            #         groupby_ftr = 'tweet_id'
            #         grouped = df_m.groupby(groupby_ftr, sort=False)
            #         grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()
            #
            #         # tmp_list = df_m[df_m['ra'] == 1].index.tolist()
            #         # df_m['rel_v'][tmp_list] = [weight_list[0]] * len(tmp_list)
            #         # tmp_list = df_m[df_m['ra'] == 2].index.tolist()
            #         # df_m['rel_v'][tmp_list] = [weight_list[1]] * len(tmp_list)
            #         # tmp_list = df_m[df_m['ra'] == 3].index.tolist()
            #         # df_m['rel_v'][tmp_list] = [weight_list[2]] * len(tmp_list)
            #         # tmp_list = df_m[df_m['ra'] == 4].index.tolist()
            #         # df_m['rel_v'][tmp_list] = [weight_list[3]] * len(tmp_list)
            #         # tmp_list = df_m[df_m['ra'] == 5].index.tolist()
            #         # df_m['rel_v'][tmp_list] = [weight_list[4]] * len(tmp_list)
            #         # tmp_list = df_m[df_m['ra'] == 6].index.tolist()
            #         # df_m['rel_v'][tmp_list] = [weight_list[5]] * len(tmp_list)
            #         # tmp_list = df_m[df_m['ra'] == 7].index.tolist()
            #         # df_m['rel_v'][tmp_list] = [weight_list[6]] * len(tmp_list)
            #
            #         # for t_id in grouped.groups.keys():
            #         for t_id in claims_list_all:
            #
            #             if t_id >= 2000:
            #                 continue
            #             pp_tmp2 = 0
            #             pf_tmp2 = 0
            #             tpp_tmp2 = 0
            #             tpf_tmp2 = 0
            #
            #             df_tmp = df_m[df_m['tweet_id'] == t_id]
            #             pt = np.mean(df_tmp['rel_v_b'])
            #             # pt = np.mean(df_tmp['acc'])
            #             gt = list(df_tmp['rel_gt_v'])[0]
            #             if gt!=0:
            #                 # continue
            #                 if gt > 0:
            #                     if pt > 0:
            #                         pp2 += 1
            #                     tpp2 += 1
            #                 if gt < 0:
            #                     if pt < 0:
            #                         pf2 += 1
            #                     tpf2 += 1
            #
            #                 for el_val in list(df_tmp['rel_v_b']):
            #                     if gt > 0:
            #                         if el_val > 0:
            #                             pp_tmp2 += 1
            #                         tpp_tmp2 += 1
            #                     if gt < 0:
            #                         if el_val < 0:
            #                             pf_tmp2 += 1
            #                         tpf_tmp2 += 1
            #
            #
            #                 if gt>0:
            #                     chi_sq_l2.append(pp_tmp2)
            #                     chi_sq_l2.append(tpp_tmp2 - pp_tmp2)
            #                 elif gt<0:
            #                     chi_sq_l2.append(pf_tmp2)
            #                     chi_sq_l2.append(tpf_tmp2 - pf_tmp2)
            #
            #
            #             val_list = list(df_tmp['rel_v'])
            #             val_list_ra = list(df_tmp['ra'])
            #             tweet_avg2[t_id] = np.mean(val_list)
            #             tweet_med2[t_id] = np.median(val_list)
            #             tweet_var2[t_id] = np.var(val_list)
            #
            #             tweet_avg_l2.append(np.var(val_list))
            #
            #             val_list = list(df_tmp['err'])
            #             abs_var_err = [np.abs(x) for x in val_list]
            #
            #             tweet_dev_avg2[t_id] = np.mean(abs_var_err)
            #             tweet_dev_med2[t_id] = np.median(abs_var_err)
            #             tweet_dev_var2[t_id] = np.var(abs_var_err)
            #             tweet_dev_avg_l2.append(np.mean(abs_var_err))
            #
            #     ##################################################
            #     print(pp2 / float(tpp2))
            #     print(pp)
            #     print(pf2 / float(tpf2))
            #     print(pf2)
            #     # print(np.corrcoef(pt_list, gt_list))
            #     # print(scipy.stats.spearmanr(pt_list, gt_list))
            #     # print(remotedir)
            #     # from scipy.stats import chisquare
            #     # Pearson Corr of TPB and Disp|| Spearman corr of TPB and Disp')
            #     print([pp + pf,tpp-pp + tpf-pf], [pp2 + pf2,tpp2-pp2 + tpf2-pf2])
            #     output.write('||' + dataset1 + ' and ' + dataset2 + '||')
            #
            #     if 'gt' in dataset1:
            #         output.write(str(dist_list1[0]) + ', ' + str(dist_list1[1]) + ', ' + str(dist_list1[2]) + ', ' +
            #                      str(dist_list1[3]) + ', ' + str(dist_list1[4]) + '<<BR>> <<BR>>')
            #     else:
            #         output.write(str(dist_list1[0])+ ', '+ str(dist_list1[1])+ ', '+str(dist_list1[2])+ ', '+
            #                      str(dist_list1[3])+ ', '+str(dist_list1[4])+ ', '+str(dist_list1[5])+ ', '+str(dist_list1[6]) + '<<BR>><<BR>>')
            #
            #
            #     if 'gt' in dataset2:
            #         output.write(str(dist_list2[0]) + ', ' + str(dist_list2[1]) + ', ' + str(dist_list2[2]) + ', ' +
            #                      str(dist_list2[3]) + ', ' + str(dist_list2[4]) + '||')
            #
            #     else:
            #         output.write(str(dist_list2[0])+ ', '+ str(dist_list2[1])+ ', '+str(dist_list2[2])+ ', '+
            #                      str(dist_list2[3])+ ', '+str(dist_list2[4])+ ', '+str(dist_list2[5])+ ', '+str(dist_list2[6]) + '||')
            #
            #     # scipy.stats.chisquare(dist_list1)
            #     output.write(str(scipy.stats.chisquare(dist_list1)) + '<<BR>><<BR>>')
            #     output.write(str(scipy.stats.chisquare(dist_list2)) + '||')
            #
            #     try:
            #         out = scipy.stats.chi2_contingency([dist_list1,dist_list2])
            #         output.write(str(out[0]) + '<<BR>>' + str(out[1]))
            #     except:
            #         output.write('NULL')
            #
            #     # print('------ together : ----------')
            #     surv = np.array([[pp + pf,tpp-pp + tpf-pf], [pp2 + pf2,tpp2-pp2 + tpf2-pf2]])
            #     out = scipy.stats.chi2_contingency(surv)
            #     output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]))
            #
            #     # # print('------ True claims : ----------')
            #     # surv = np.array([[pp,tpp-pp], [pp2,tpp2-pp2]])
            #     # out = scipy.stats.chi2_contingency(surv)
            #     # output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]))
            #     #
            #     # # print('------ False claims : ----------')
            #     # surv = np.array([[pf,tpf-pf], [pf2,tpf2-pf2]])
            #     # out = scipy.stats.chi2_contingency(surv)
            #     # output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]) )
            #
            #
            #
            #     # surv = np.array([[pp,tpp-pp,pf,tpf-pf], [pp2,tpp2-pp2,pf2,tpf2-pf2]])
            #     # print(scipy.stats.chi2_contingency(surv))
            #
            #
            #     # comparing TPB in diff surveys
            #     # print('TPB vectors comparing')
            #     # output.write('||' + str(scipy.stats.chisquare(tweet_dev_avg_l)) + '<<BR>>')
            #     # output.write(str(scipy.stats.chisquare(tweet_dev_avg_l2)))
            #
            #     out = scipy.stats.chi2_contingency([tweet_dev_avg_l,tweet_dev_avg_l2])
            #     output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]))
            #     # output.write('||' + str(scipy.stats.mannwhitneyu(tweet_dev_avg_l, tweet_dev_avg_l2)))
            #
            #     # print('TPB vectors comparing correlation pearson')
            #     out = np.corrcoef(tweet_dev_avg_l, tweet_dev_avg_l2)
            #     output.write('||' + str(out[1]))
            #
            #     # print('TPB vectors comparing correlation spearman')
            #     out  = scipy.stats.spearmanr(tweet_dev_avg_l, tweet_dev_avg_l2)
            #     output.write('||' + str(out) + '||\n')
            #
            #
            #
            #     # print('TPB and ')
            #     # print(scipy.stats.mannwhitneyu(tweet_dev_avg_l, tweet_dev_avg_l2))
            # #
            # #
            # #
            # #
            # #

    if args.t == "AMT_dataset_reliable_news_processing_all_dataset_weighted_visualisation_initial_stastistics_analysis_plan_individual_tweet":

        remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
        output = open(remotedir + '/comparing_diff_surveys.txt', 'w')
        # output.write('||Surveys || Distribution of answering|| Chi square of dist of answering survey 1 <<BR>> survey 2'
        #              '|| Chi square test (dependency) of survey 1 and 2 ||Chi square proportion of correct answering|| Chi square of TPB of survey 1<<BR>> survey 2'
        #              '|| Chi square test(dependency) of survey 1 and 2||MWU test to compare TPB|| Pearson Corr of TPB|| Spearman Corr of TPB||\n')



        output.write('||claims || Ground truth ||Surveys || Distribution of answering|| Relative distribution of answering|| \n')

        # output.write('|| Survey || pearson corr between TPB and Disp|| pearson corr between TPB and Disp||\n')

        data_n = 'sp';
        ind = 1
        inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
        inp2 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
        inp3 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
        inp4 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'

        df_gt = pd.read_csv(inp1, sep='\t')
        df_gt_exp = pd.read_csv(inp2, sep='\t')
        df_inc_gt = pd.read_csv(inp3, sep='\t')
        df_inc_gt_exp = pd.read_csv(inp4, sep='\t')

        inp1 = remotedir + 'gt.csv'
        inp2 = remotedir + 'gt_experts.csv'
        inp3 = remotedir + 'incentivize_gt.csv'
        inp4 = remotedir + 'incentivize_gt_experts.csv'

        df_gt_amt = pd.read_csv(inp1, sep=',')
        df_gt_exp_amt = pd.read_csv(inp2, sep=',')
        df_inc_gt_amt = pd.read_csv(inp3, sep=',')
        df_inc_gt_exp_amt = pd.read_csv(inp4, sep=',')

        w_id_gt = df_gt_amt['WorkerId']
        w_id_gt_exp = df_gt_exp_amt['WorkerId']
        w_id_inc_gt = df_inc_gt_amt['WorkerId']
        w_id_inc_gt_exp = df_inc_gt_exp_amt['WorkerId']

        w1_gt_l = list(set(w_id_gt) & set(w_id_gt_exp))
        w2_gt_l = list(set(w_id_gt) & set(w_id_inc_gt_exp))

        w1_gt_exp_l = list(set(w_id_gt_exp) & set(w_id_gt))
        w2_gt_exp_l = list(set(w_id_gt_exp) & set(w_id_inc_gt))

        w1_inc_gt_l = list(set(w_id_inc_gt) & set(w_id_gt_exp))
        w2_inc_gt_l = list(set(w_id_inc_gt) & set(w_id_inc_gt_exp))

        w1_inc_gt_exp_l = list(set(w_id_inc_gt_exp) & set(w_id_gt))
        w2_inc_gt_exp_l = list(set(w_id_inc_gt_exp) & set(w_id_inc_gt))

        inters_gt_l = []
        inters_gt_exp_l = []
        inters_inc_gt_l = []
        inters_inc_gt_exp_l = []

        inters_gt_l += w1_gt_l
        inters_gt_l += w2_gt_l

        inters_gt_exp_l += w1_gt_exp_l
        inters_gt_exp_l += w2_gt_exp_l

        inters_inc_gt_l += w1_inc_gt_l
        inters_inc_gt_l += w2_inc_gt_l

        inters_inc_gt_exp_l += w1_inc_gt_exp_l
        inters_inc_gt_exp_l += w2_inc_gt_exp_l

        import hashlib

        cod_id_gt_list = []
        for w_id in inters_gt_l:
            try:
                ind_tmp = df_gt_amt[df_gt_amt['WorkerId'] == w_id].index.tolist()[0]
                cod_id = df_gt_amt['Answer.Q2age'][ind_tmp]
                cod_id_gt_list.append(cod_id)
            except:
                continue

        w_id_gt_filter = []
        for w_id in range(0, 1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_gt_list:
                w_id_gt_filter.append(w_id + 1000)

        cod_id_gt_exp_list = []
        for w_id in inters_gt_exp_l:
            try:
                ind_tmp = df_gt_exp_amt[df_gt_exp_amt['WorkerId'] == w_id].index.tolist()[0]
                cod_id = df_gt_exp_amt['Answer.Q2age'][ind_tmp]
                cod_id_gt_exp_list.append(cod_id)
            except:
                continue

        w_id_gt_exp_filter = []
        for w_id in range(0, 1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_gt_exp_list:
                w_id_gt_exp_filter.append(w_id + 1000)

        cod_id_inc_gt_list = []
        for w_id in inters_inc_gt_l:
            try:
                ind_tmp = df_inc_gt_amt[df_inc_gt_amt['WorkerId'] == w_id].index.tolist()[0]
                cod_id = df_inc_gt_amt['Answer.Q2age'][ind_tmp]
                cod_id_inc_gt_list.append(cod_id)
            except:
                continue

        w_id_inc_gt_filter = []
        for w_id in range(0, 1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_inc_gt_list:
                w_id_inc_gt_filter.append(w_id + 1000)

        cod_id_inc_gt_exp_list = []
        for w_id in inters_inc_gt_exp_l:
            try:
                ind_tmp = df_inc_gt_exp_amt[df_inc_gt_exp_amt['WorkerId'] == w_id].index.tolist()[0]
                cod_id = df_inc_gt_exp_amt['Answer.Q2age'][ind_tmp]
                cod_id_inc_gt_exp_list.append(cod_id)
            except:
                continue

        w_id_inc_gt_exp_filter = []
        for w_id in range(0, 1000):
            algo = 'sha1'
            data = "twitter app survey crowd signal";
            h = hashlib.new(algo)
            h.update(data + str(w_id))

            result = h.hexdigest()
            if result in cod_id_inc_gt_exp_list:
                w_id_inc_gt_exp_filter.append(w_id + 1000)

        publisher_leaning = 1
        source_dict = {}
        text_dict = {}
        date_dict = {}
        c_ind_news = 0
        c_ind_source = 0
        c_ind_date = 0
        news_txt = {}
        news_source = {}
        news_cat = {}
        news_date = {}
        news_topic = {}
        topic_count_dict = collections.defaultdict(int)

        line_count = 0
        tmp_dict = {}
        claims_list = []

        # run = 'plot'
        run = 'analysis'
        # run = 'second-analysis'
        # exp1_list = sample_tweets_exp1
        df = collections.defaultdict()
        df_w = collections.defaultdict()
        tweet_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg_med_var = collections.defaultdict(list)
        tweet_dev_avg = {}
        tweet_dev_med = {}
        tweet_dev_var = {}
        tweet_avg = {}
        tweet_med = {}
        tweet_var = {}
        tweet_gt_var = {}

        tweet_dev_avg_l = []
        tweet_dev_med_l = []
        tweet_dev_var_l = []
        tweet_avg_l = []
        tweet_med_l = []
        tweet_var_l = []
        tweet_gt_var_l = []
        avg_susc = 0
        avg_gull = 0
        avg_cyn = 0

        tweet_abs_dev_avg = {}
        tweet_abs_dev_med = {}
        tweet_abs_dev_var = {}

        tweet_abs_dev_avg_l = []
        tweet_abs_dev_med_l = []
        tweet_abs_dev_var_l = []

        # for ind in [1,2,3]:
        all_acc = []

        balance_f = 'un_balanced'
        # balance_f = 'balanced'

        # fig_f = True
        fig_f = False

        gt_l_dict = collections.defaultdict(list)
        perc_l_dict = collections.defaultdict(list)
        abs_perc_l_dict = collections.defaultdict(list)
        gt_set_dict = collections.defaultdict(list)
        perc_mean_dict = collections.defaultdict(list)
        abs_perc_mean_dict = collections.defaultdict(list)
        remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'

        # inp10 = remotedir + 'amt_answers_' + 'sp_gt_10' + '_claims_exp' + str(1) + '_final_weighted_gt.csv'
        inp = remotedir + 'amt_answers_' + 'sp' + '_claims_exp' + str(1) + '_final_weighted.csv'
        df = pd.read_csv(inp, sep="\t")

        claims_list_all = set(df['tweet_id'])

        data_c = 0
        # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt_1', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2', 'snopes_incentive_notimer',
        #              'snopes_noincentive_timer']


        dist_list = collections.defaultdict()
        # data_list = ['snopes', 'snopes_gt_1', 'snopes_2']
        # data_list = ['snopes_2', 'snopes_incentive']

        # data_list = ['snopes',  'snopes_ssi', 'snopes_incentive', 'snopes_gt', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2', 'snopes_incentive_notimer', 'snopes_noincentive_timer']


        data_list = ['snopes_gt', 'snopes_gt_10']


        # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt_1','snopes_gt_2','snopes_gt_3', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer','snopes_gt_10']
        # data_list1 = ['snopes',  'snopes_ssi', 'snopes_incentive', 'snopes_gt','snopes_gt_2','snopes_gt_3',  'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2', 'snopes_incentive_notimer', 'snopes_noincentive_timer','snopes_gt_10']
        # #

        # data_list = ['snopes_gt', 'snopes_gt_2', 'snopes_gt_3' ]
        # data_list1 = ['snopes_gt',  'snopes_gt_2', 'snopes_gt_3' ]


        TPB_dict = collections.defaultdict(list)

        Disp_dict = collections.defaultdict(list)
        tweet_dev_avg_dict = collections.defaultdict()
        for dataset1 in data_list:  # , 'snopes_incentive_10', 'snopes_2', 'snopes_ssi', 'snopes_nonpol', 'snopes','snopes_gt','snope_gt_experts', 'snopes_gt_incentive', 'snopes_gt_incentive_experts']:
            # dataset1 = dataset
            data_c += 1
            dist_list[dataset1] = collections.defaultdict(list)
            tweet_dev_avg_dict[dataset1] = collections.defaultdict()
            if dataset1 == 'snopes' or dataset1 == 'snopes_ssi' or dataset1 == 'snopes_incentive' \
                    or dataset1 == 'snopes_gt' or dataset1 == 'snopes_gt_1' or dataset1 == 'snopes_gt_experts' or dataset1 == 'snopes_gt_incentive' \
                    or dataset1 == 'snopes_gt_incentive_experts' or dataset1 == 'snopes_2' or dataset1 == 'snopes_incentive_notimer' \
                    or dataset1 == 'snopes_noincentive_timer' or dataset1 == 'snopes_gt_10' or dataset1 == 'snopes_gt_2' or dataset1 == 'snopes_gt_3':
                claims_list = []
                remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
                inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
                news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']

                sample_tweets_exp1 = []

                tweet_txt_dict = {}
                tweet_date_dict = {}
                tweet_lable_dict = {}
                tweet_publisher_dict = {}
                print(inp_all)

                for i in range(0, 5):
                    df_cat = news_cat_list[i]
                    df_cat_f = news_cat_list_f[i]
                    inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
                    cat_count = 0
                    for line in inF:
                        claims_list.append(line)
                        cat_count += 1

                for line in claims_list:
                    line_splt = line.split('<<||>>')
                    publisher_name = int(line_splt[2])
                    tweet_txt = line_splt[3]
                    tweet_id = publisher_name
                    cat_lable = line_splt[4]
                    dat = line_splt[5]
                    dt_splt = dat.split(' ')[0].split('-')
                    m_day = int(dt_splt[2])
                    m_month = int(dt_splt[1])
                    m_year = int(dt_splt[0])
                    m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
                    tweet_txt_dict[tweet_id] = tweet_txt
                    tweet_date_dict[tweet_id] = m_date
                    tweet_lable_dict[tweet_id] = cat_lable
                    # outF = open(remotedir + 'table_out.txt', 'w')

            print('-------------------' + dataset1 + '------------------')

            if dataset1 == 'snopes':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes'
            elif dataset1 == 'snopes_2':
                data_n = 'sp_2'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_2'

            elif dataset1 == 'snopes_gt':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt'
            elif dataset1 == 'snopes_gt_1':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_1'
            elif dataset1 == 'snopes_gt_2':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_2'
            elif dataset1 == 'snopes_gt_3':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_3'
            elif dataset1 == 'snopes_gt_10':
                data_n = 'sp_gt_10'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_10'

            elif dataset1 == 'snopes_gt_experts':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_exp'

            elif dataset1 == 'snopes_gt_incentive':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_inc'


            elif dataset1 == 'snopes_gt_incentive_experts':
                data_n = 'sp'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_gt_inc_exp'


            elif dataset1 == 'snopes_incentive_notimer':
                data_n = 'sp_incentive_notimer'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive_notimer'


            elif dataset1 == 'snopes_noincentive_timer':

                data_n = 'sp_noincentive_timer'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_noincentive_timer'


            elif dataset1 == 'snopes_incentive':
                data_n = 'sp_incentive'
                data_addr = 'snopes'
                ind_l = [1, 2, 3]
                ind_l = [1]
                data_name = 'Snopes_incentive'

            # elif dataset == 'snopes_incentive_10':
            #     data_n = 'sp_incentive_10'
            #     data_addr = 'snopes'
            #     ind_l = [1, 2, 3]
            #     ind_l = [1]
            #     data_name = 'Snopes_incentive_10'

            elif dataset1 == 'snopes_ssi':
                data_n = 'sp_ssi'
                data_addr = 'snopes'
                ind_l = [1]
                data_name = 'Snopes_ssi'

            df = collections.defaultdict()
            df_w = collections.defaultdict()
            tweet_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg_med_var = collections.defaultdict(list)
            tweet_dev_avg = {}
            tweet_dev_med = {}
            tweet_dev_var = {}
            tweet_avg = {}
            tweet_med = {}
            tweet_var = {}
            tweet_gt_var = {}

            tweet_dev_avg_l = []
            tweet_dev_med_l = []
            tweet_dev_var_l = []
            tweet_avg_l = []
            tweet_med_l = []
            tweet_var_l = []
            tweet_gt_var_l = []
            avg_susc = 0
            avg_gull = 0
            avg_cyn = 0

            tweet_abs_dev_avg = {}
            tweet_abs_dev_med = {}
            tweet_abs_dev_var = {}

            tweet_abs_dev_avg_l = []
            tweet_abs_dev_med_l = []
            tweet_abs_dev_var_l = []

            tweet_abs_dev_avg_rnd = {}
            tweet_dev_avg_rnd = {}

            tweet_skew = {}
            tweet_skew_l = []

            tweet_vote_avg_med_var = collections.defaultdict(list)
            tweet_vote_avg = collections.defaultdict()
            tweet_vote_med = collections.defaultdict()
            tweet_vote_var = collections.defaultdict()

            tweet_avg_group = collections.defaultdict()
            tweet_med_group = collections.defaultdict()
            tweet_var_group = collections.defaultdict()
            tweet_var_diff_group = collections.defaultdict()

            tweet_kldiv_group = collections.defaultdict()

            tweet_vote_avg_l = []
            tweet_vote_med_l = []
            tweet_vote_var_l = []
            tweet_chi_group = {}
            tweet_chi_group_1 = {}
            tweet_chi_group_2 = {}
            tweet_skew = {}
            news_cat_list_tf = [4, 2, 3, 1]
            t_f_dict_len = collections.defaultdict(int)
            t_f_dict = {}
            tid_time_avg = {}
            tid_time_std = {}
            if dataset1 == 'snopes' or dataset1 == 'snopes_ssi' or dataset1 == 'snopes_incentive' \
                    or dataset1 == 'snopes_gt' or dataset1 == 'snopes_gt_1' or dataset1 == 'snopes_gt_experts' or dataset1 == 'snopes_gt_incentive' \
                    or dataset1 == 'snopes_gt_incentive_experts' or dataset1 == 'snopes_2' or dataset1 == 'snopes_incentive_notimer' \
                    or dataset1 == 'snopes_noincentive_timer' or dataset1 == 'snopes_gt_10' or dataset1 == 'snopes_gt_2' or dataset1 == 'snopes_gt_3':
                news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
                news_cat_list_v = [-1, -.5, 0, 0.5, 1]


            w_fnb_dict = collections.defaultdict()
            w_fpb_dict = collections.defaultdict()
            w_apb_dict = collections.defaultdict()
            gt_acc = collections.defaultdict()
            # for cat in news_cat_list_v:
            #     gt_acc[cat] = [0] * (len(news_cat_list_t_f))
            weight_list = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
            # weight_list = [-2.5, -0.84, 0.25, 1, -1.127, 1.1, 1.05]
            # weight_list = [-2.36, -0.73, 0.53, 0.87, -0.87, 0.93, 1.53]
            pt_list = []
            gt_list = []
            pp = 0
            pf = 0
            tpp = 0
            tpf = 0
            chi_sq_l = []
            f_filter = []
            ind_l = [1]
            dist_list1 = []

            for ind in ind_l:

                inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
                inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted.csv'
                # df[ind] = pd.read_csv(inp1, sep="\t")


                if dataset1 == 'snopes_gt':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted_gt.csv'
                    f_filter = w_id_gt_filter
                elif dataset1 == 'snopes_gt_1':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp1_2_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp1_2_weighted_gt.csv'
                    # f_filter = w_id_gt_filter
                elif dataset1 == 'snopes_gt_2':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp2_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp2_weighted_gt.csv'
                elif dataset1 == 'snopes_gt_3':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp3_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp3_weighted_gt.csv'
                    # f_filter = w_id_gt_filter
                elif dataset1 == 'snopes_gt_10':
                    inp1 = remotedir + 'amt_answers_sp_gt_10_claims_exp' + str(ind) + '_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_sp_gt_10_claims_exp' + str(
                        ind) + '_weighted_gt.csv'
                elif dataset1 == 'snopes_gt_experts':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(
                        ind) + '_weighted_gt_experts.csv'
                    f_filter = w_id_gt_exp_filter

                elif dataset1 == 'snopes_gt_incentive':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(
                        ind) + '_final_weighted_gt.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(
                        ind) + '_weighted_gt.csv'

                    f_filter = w_id_inc_gt_filter

                elif dataset1 == 'snopes_gt_incentive_experts':
                    inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(
                        ind) + '_final_weighted_gt_experts.csv'
                    inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(
                        ind) + '_weighted_gt_experts.csv'
                    f_filter = w_id_inc_gt_exp_filter

                df[ind] = pd.read_csv(inp1, sep="\t")
                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                df_m = df[ind].copy()

                worker_id_list = set(df_m['worker_id'])
                filter_l = list(worker_id_list - set(f_filter))
                df_m = df_m[df_m['worker_id'].isin(filter_l)]

                df_tt = df_m[df_m['tweet_id'] == 2001]
                df_t2 = df_tt[df_tt['ra'] < 4]
                stp_workers1 = df_t2['worker_id']

                df_tt = df_m[df_m['tweet_id'] == 2002]
                df_t2 = df_tt[df_tt['ra'] > 2]
                stp_workers2 = df_t2['worker_id']

                worker_id_list = set(df_m['worker_id'])
                filter_l = list(worker_id_list - set(list(stp_workers1) + list(stp_workers2)))
                df_m = df_m[df_m['worker_id'].isin(filter_l)]

                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                # df_w[ind] = pd.read_csv(inp1_w, sep="\t")

                # df_m = df[ind].copy()

                groupby_ftr = 'tweet_id'
                grouped = df_m.groupby(groupby_ftr, sort=False)
                grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()

                # tmp_list = df_m[df_m['ra'] == 1].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[0]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 2].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[1]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 3].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[2]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 4].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[3]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 5].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[4]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 6].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[5]] * len(tmp_list)
                # tmp_list = df_m[df_m['ra'] == 7].index.tolist()
                # df_m['rel_v'][tmp_list] = [weight_list[6]] * len(tmp_list)

                # df_mm = df_m[df_m['tweet_id'].isin(claims_list_all)]
                df_mm = df_m[df_m['tweet_id'].isin(grouped.groups.keys())]

                if 'gt' in dataset1:
                    dist_list1.append(len(df_mm[df_mm['ra'] == 1]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 2]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 3]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 4]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 5]))

                else:
                    dist_list1.append(len(df_mm[df_mm['ra'] == 1]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 2]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 3]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 4]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 5]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 6]))
                    dist_list1.append(len(df_mm[df_mm['ra'] == 7]))

                # for t_id in grouped.groups.keys():
                for t_id in list(claims_list_all):
                    # if t_id == 1027:
                    print (t_id)
                    # if t_id==2001:

                    if t_id >= 2000:
                        continue
                    pp_tmp = 0
                    pf_tmp = 0
                    tpp_tmp = 0
                    tpf_tmp = 0

                    df_tmp = df_m[df_m['tweet_id'] == t_id]

                    if 'gt' in dataset1:
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 1])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 2])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 3])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 4])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 5])))

                    else:
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 1])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 2])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 3])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 4])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 5])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 6])))
                        dist_list[dataset1][t_id].append(float(len(df_tmp[df_tmp['ra'] == 7])))

                    time_ans = df_tmp['delta_time']
                    tid_time_avg[t_id]=np.mean(list(time_ans))
                    tid_time_std[t_id]=np.std(list(time_ans))
                    pt = np.mean(df_tmp['rel_v_b'])
                    # pt = np.mean(df_tmp['acc'])
                    gt = list(df_tmp['rel_gt_v'])[0]
                    if gt != 0:
                        #     continue
                        if gt > 0:
                            if pt > 0:
                                pp += 1
                            tpp += 1
                        if gt < 0:
                            if pt < 0:
                                pf += 1
                            tpf += 1

                        for el_val in list(df_tmp['rel_v_b']):
                            if gt > 0:
                                if el_val > 0:
                                    pp_tmp += 1
                                tpp_tmp += 1
                            if gt < 0:
                                if el_val < 0:
                                    pf_tmp += 1
                                tpf_tmp += 1

                        if gt > 0:
                            chi_sq_l.append(pp_tmp)
                            chi_sq_l.append(tpp_tmp - pp_tmp)
                        elif gt < 0:
                            chi_sq_l.append(pf_tmp)
                            chi_sq_l.append(tpf_tmp - pf_tmp)

                    val_list = list(df_tmp['rel_v'])
                    val_list_ra = list(df_tmp['ra'])
                    tweet_avg[t_id] = np.mean(val_list)
                    tweet_med[t_id] = np.median(val_list)
                    tweet_var[t_id] = np.var(val_list)
                    tweet_avg_l.append(np.var(val_list))

                    val_list1 = list(df_tmp['err'])
                    abs_var_err = [np.abs(x) for x in val_list1]

                    tweet_dev_avg[t_id] = np.mean(abs_var_err)
                    tweet_dev_med[t_id] = np.median(abs_var_err)
                    tweet_dev_var[t_id] = np.var(abs_var_err)
                    tweet_dev_avg_l.append(np.mean(abs_var_err))

                tweet_dev_avg_dict[dataset1] = tweet_dev_avg
                TPB_dict[dataset1] = tweet_dev_avg_l
                Disp_dict[dataset1] = tweet_avg_l
                print(np.corrcoef(tweet_dev_avg_l, tweet_avg_l))
                print(scipy.stats.spearmanr(tweet_dev_avg_l, tweet_avg_l))
                print(pp / float(tpp))
                print(pp)
                print(pf / float(tpf))
                print(pf)
                # print()

                time_avg_list = []
                time_std_list = []
                tid_sorted = sorted(tweet_dev_avg_dict[dataset1], key=tweet_dev_avg_dict[dataset1].get, reverse=True)
                time_avg_list_cat = collections.defaultdict(list)
                time_std_list_cat = collections.defaultdict(list)
                # for cat in ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']:
                #     time_avg_list_cat[cat] = collections.defaultdict()
                #     time_std_list_cat[cat] = collections.defaultdict()

                for t_id in claims_list_all:
                # for t_id in tid_sorted:
                    time_avg_list.append(tid_time_avg[t_id])
                    time_std_list.append(tid_time_std[t_id])
                    time_avg_list_cat[tweet_lable_dict[t_id]].append(tid_time_avg[t_id])
                    time_std_list_cat[tweet_lable_dict[t_id]].append(tid_time_std[t_id])

                for cat in ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']:
                    mplpl.figure()
                    mplpl.errorbar(range(len(time_avg_list_cat[cat])), time_avg_list_cat[cat], yerr=time_std_list_cat[cat])
                    mplpl.title(dataset1 + '_' + cat + ',  Avg : ' + str(np.mean(time_avg_list_cat[cat])))
                    mplpl.xlabel('Claims')
                    mplpl.ylabel('Time')
                    mplpl.ylim([-50,50])
                    pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/timing/' + dataset1 + '_time_sorted_diff_cat_' + cat
                    # pp = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/fig/timing/' + dataset1 + '_time'
                    mplpl.savefig(pp + '.pdf', format='pdf')
                    mplpl.savefig(pp + '.png', format='png')

        # for t_id in claims_list_all:
        #
        #     output.write('||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id])
        #     output.write('||' + data_list[0]+'<<BR>>'+ data_list[1]+'<<BR>>'+ data_list[2])
        #     output.write('||' + str(dist_list[data_list[0]][t_id])+'<<BR>>'+
        #                  str(dist_list[data_list[1]][t_id])+'<<BR>>'
        #                  + str(dist_list[data_list[2]][t_id]))
        #     output.write(
        #         '||' + str(np.round(dist_list[data_list[0]][t_id] / np.sum(dist_list[data_list[0]][t_id]),3)) + '<<BR>>' +
        #         str(np.round(dist_list[data_list[1]][t_id] / np.sum(dist_list[data_list[1]][t_id]),3))+ '<<BR>>'
        #         + str(np.round(dist_list[data_list[2]][t_id]/np.sum(dist_list[data_list[2]][t_id]),3)) + '||\n')

        # for t_id in claims_list_all:
        #     output.write('||' + tweet_txt_dict[t_id] + '||' + tweet_lable_dict[t_id])
        #     output.write('||' + data_list[0] + '<<BR>>' + data_list[1])
        #     output.write('||' + str(dist_list[data_list[0]][t_id]) + '<<BR>>' +
        #                  str(dist_list[data_list[1]][t_id]) )
        #     output.write(
        #         '||' + str(np.round(dist_list[data_list[0]][t_id] / np.sum(dist_list[data_list[0]][t_id]),3)) + '<<BR>>' +
        #         str(np.round(dist_list[data_list[1]][t_id] / np.sum(dist_list[data_list[1]][t_id]),3))  + '||\n')

                # output.write('|| ' + str(scipy.stats.spearmanr(tweet_avg_l, tweet_dev_avg_l))+ '||\n')



        # TPB_dict[dataset1] = tweet_dev_avg_l
        #
        #
        #
        # index_list_m = range(len(TPB_dict[data_list[0]]+TPB_dict[data_list[1]]+TPB_dict[data_list[2]]))
        # data_list = ['snopes_gt','snopes_gt_2','snopes_gt_3']
        #
        # df_w = pd.DataFrame({'sp_tpb': Series(TPB_dict[data_list[0]]+TPB_dict[data_list[1]]+TPB_dict[data_list[2]], index=index_list_m),
        #                      'sp_disp': Series(Disp_dict[data_list[0]]+Disp_dict[data_list[1]]+Disp_dict[data_list[2]], index=index_list_m),})
        #
        # print(remotedir)
        # df_w.to_csv(remotedir + 'TPB_Disp_diff_surveys_sp150.csv',
        #             columns=df_w.columns, sep=",", index=False)



        # index_list_m = range(len(TPB_dict[data_list[0]]))
        # data_list = ['snopes_gt','snopes_gt_2','snopes_gt_3']
        #
        # df_w = pd.DataFrame({data_list[0]+'_tpb': Series(TPB_dict[data_list[0]], index=index_list_m),
        #                      data_list[0]+ '_disp': Series(Disp_dict[data_list[0]], index=index_list_m),
        #                      data_list[1] + '_tpb': Series(TPB_dict[data_list[1]], index=index_list_m),
        #                      data_list[1] + '_disp': Series(Disp_dict[data_list[1]], index=index_list_m),
        #                      data_list[2]+'_tpb': Series(TPB_dict[data_list[2]], index=index_list_m),
        #                      data_list[2]+ '_disp': Series(Disp_dict[data_list[2]], index=index_list_m),})
        #
        # print(remotedir)
        # df_w.to_csv(remotedir + 'TPB_Disp_diff_surveys_sp1_2_3.csv',
        #             columns=df_w.columns, sep=",", index=False)

        #
        #
        # index_list_m = range(len(TPB_dict[data_list[0]]))
        # # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt','snopes_gt_1','snopes_gt_1','snopes_gt_2','snopes_gt_3', 'snopes_gt_experts'
        # #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer', 'snopes_gt_10']
        #
        # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt_1', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2', 'snopes_incentive_notimer',
        #              'snopes_noincentive_timer']
        # df_w = pd.DataFrame({data_list[0] + '_tpb': Series(TPB_dict[data_list[0]], index=index_list_m),
        #                      data_list[0] + '_disp': Series(Disp_dict[data_list[0]], index=index_list_m),
        #                      data_list[1] + '_tpb': Series(TPB_dict[data_list[1]], index=index_list_m),
        #                      data_list[1] + '_disp': Series(Disp_dict[data_list[1]], index=index_list_m),
        #                      data_list[2] + '_tpb': Series(TPB_dict[data_list[2]], index=index_list_m),
        #                      data_list[2] + '_disp': Series(Disp_dict[data_list[2]], index=index_list_m),
        #                      data_list[3] + '_tpb': Series(TPB_dict[data_list[3]], index=index_list_m),
        #                      data_list[3] + '_disp': Series(Disp_dict[data_list[3]], index=index_list_m),
        #                      data_list[4] + '_tpb': Series(TPB_dict[data_list[4]], index=index_list_m),
        #                      data_list[4] + '_disp': Series(Disp_dict[data_list[4]], index=index_list_m),
        #                      data_list[5] + '_tpb': Series(TPB_dict[data_list[5]], index=index_list_m),
        #                      data_list[5] + '_disp': Series(Disp_dict[data_list[5]], index=index_list_m),
        #                      data_list[6] + '_tpb': Series(TPB_dict[data_list[6]], index=index_list_m),
        #                      data_list[6] + '_disp': Series(Disp_dict[data_list[6]], index=index_list_m),
        #                      data_list[7] + '_tpb': Series(TPB_dict[data_list[7]], index=index_list_m),
        #                      data_list[7] + '_disp': Series(Disp_dict[data_list[7]], index=index_list_m),
        #                      data_list[8] + '_tpb': Series(TPB_dict[data_list[8]], index=index_list_m),
        #                      data_list[8] + '_disp': Series(Disp_dict[data_list[8]], index=index_list_m),
        #                      data_list[9] + '_tpb': Series(TPB_dict[data_list[9]], index=index_list_m),
        #                      data_list[9] + '_disp': Series(Disp_dict[data_list[9]], index=index_list_m), })
        #
        # print(remotedir)
        # df_w.to_csv(remotedir + 'TPB_Disp_diff_surveys_50claims.csv',
        #             columns=df_w.columns, sep=",", index=False)
        #


        exit()
        # index_list_m = range(len(TPB_dict[data_list[0]]))
        # data_list = ['snopes', 'snopes_ssi', 'snopes_incentive', 'snopes_gt', 'snopes_gt_experts'
        #     , 'snopes_gt_incentive', 'snopes_gt_incentive_experts', 'snopes_2','snopes_incentive_notimer', 'snopes_noincentive_timer']
        #
        # df_w = pd.DataFrame({data_list[0]: Series(TPB_dict[data_list[0]], index=index_list_m),
        #                      data_list[1]: Series(TPB_dict[data_list[1]], index=index_list_m),
        #                      data_list[2]: Series(TPB_dict[data_list[2]], index=index_list_m),
        #                      data_list[3]: Series(TPB_dict[data_list[3]], index=index_list_m),
        #                      data_list[4]: Series(TPB_dict[data_list[4]], index=index_list_m),
        #                      data_list[5]: Series(TPB_dict[data_list[5]], index=index_list_m),
        #                      data_list[6]: Series(TPB_dict[data_list[6]], index=index_list_m),
        #                      data_list[7]: Series(TPB_dict[data_list[7]], index=index_list_m),
        #                      data_list[8]: Series(TPB_dict[data_list[8]], index=index_list_m),
        #                      data_list[9]: Series(TPB_dict[data_list[9]], index=index_list_m), })
        #
        # print(remotedir)
        # df_w.to_csv(remotedir + 'TPB_diff_surveys_1.csv',
        #             columns=df_w.columns, sep=",", index=False)
        ##################################################
        # print(pp / float(tpp))
        # print(pp)
        # print(pf / float(tpf))
        # print(pf)
        # output.write('||' + dataset1 )
        # output.write('|| ' + str(np.corrcoef(tweet_avg_l, tweet_dev_avg_l)[1]))
        # output.write('|| ' + str(scipy.stats.spearmanr(tweet_avg_l, tweet_dev_avg_l))+ '||\n')
        #
        # news_cat_list = ['pants-fire', 'false', 'mostly_false', 'half-true', 'mostly-true', 'true']
        # news_cat_list_f = ['mostly-false', 'false', 'half-true', 'true', 'mostly-true', 'pants-fire']
        #
        #
        # for dataset2 in data_list1[data_c:]:#, 'snopes_incentive_10', 'snopes_2', 'snopes_ssi', 'snopes_nonpol', 'snopes','mia', 'mia', 'politifact']:
        #
        #     # dataset2 = dataset
        #     print('##############' + dataset2+'##############')
        #     if dataset2 == 'snopes' or dataset2 == 'snopes_ssi' or dataset2 == 'snopes_incentive' or dataset2 == 'snopes_incentive_10' \
        #             or dataset2 == 'snopes_gt' or dataset2 == 'snopes_gt_experts' or dataset2 == 'snopes_gt_incentive' \
        #             or dataset2 == 'snopes_gt_incentive_experts' or dataset2 == 'snopes_2' or dataset2 == 'snopes_incentive_notimer' \
        #             or dataset2 == 'snopes_noincentive_timer' or dataset2 == 'snopes_gt_10':
        #         claims_list = []
        #         remotedir = '/NS/twitter-8/work/Reza/reliable_news/new_collected_data/snopes/'
        #         inp_all = glob.glob(remotedir + '/all_snopes_news.txt')
        #         news_cat_list = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
        #         news_cat_list_f = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']
        #
        #         sample_tweets_exp1 = []
        #
        #         tweet_txt_dict = {}
        #         tweet_date_dict = {}
        #         tweet_lable_dict = {}
        #         tweet_publisher_dict = {}
        #         print(inp_all)
        #
        #         for i in range(0, 5):
        #             df_cat = news_cat_list[i]
        #             df_cat_f = news_cat_list_f[i]
        #             inF = open(remotedir + 'politic_claims/' + df_cat_f + '/politic_claims.txt', 'r')
        #             cat_count = 0
        #             for line in inF:
        #                 claims_list.append(line)
        #                 cat_count += 1
        #
        #         for line in claims_list:
        #             line_splt = line.split('<<||>>')
        #             publisher_name = int(line_splt[2])
        #             tweet_txt = line_splt[3]
        #             tweet_id = publisher_name
        #             cat_lable = line_splt[4]
        #             dat = line_splt[5]
        #             dt_splt = dat.split(' ')[0].split('-')
        #             m_day = int(dt_splt[2])
        #             m_month = int(dt_splt[1])
        #             m_year = int(dt_splt[0])
        #             m_date = datetime.datetime(year=m_year, month=m_month, day=m_day)
        #             tweet_txt_dict[tweet_id] = tweet_txt
        #             tweet_date_dict[tweet_id] = m_date
        #             tweet_lable_dict[tweet_id] = cat_lable
        #             # outF = open(remotedir + 'table_out.txt', 'w')
        #
        #     if dataset2 == 'snopes':
        #         data_n = 'sp'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes'
        #     elif dataset2 == 'snopes_2':
        #         data_n = 'sp_2'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_2'
        #
        #     elif dataset2 == 'snopes_gt':
        #         data_n = 'sp'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_gt'
        #     elif dataset2 == 'snopes_gt_10':
        #         data_n = 'sp_gt_10'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_gt_10'
        #
        #     elif dataset2 == 'snopes_gt_experts':
        #         data_n = 'sp'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_gt_exp'
        #
        #     elif dataset2 == 'snopes_gt_inc':
        #         data_n = 'sp'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_gt_incentive'
        #
        #
        #     elif dataset2 == 'snopes_gt_incentive_experts':
        #         data_n = 'sp'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_gt_inc_exp'
        #
        #
        #     elif dataset2 == 'snopes_incentive_notimer':
        #         data_n = 'sp_incentive_notimer'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_incentive_notimer'
        #
        #
        #     elif dataset2 == 'snopes_noincentive_timer':
        #
        #         data_n = 'sp_noincentive_timer'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_noincentive_timer'
        #
        #
        #     elif dataset2 == 'snopes_incentive':
        #         data_n = 'sp_incentive'
        #         data_addr = 'snopes'
        #         ind_l = [1, 2, 3]
        #         ind_l = [1]
        #         data_name = 'Snopes_incentive'
        #
        #     # elif dataset == 'snopes_incentive_10':
        #     #     data_n = 'sp_incentive_10'
        #     #     data_addr = 'snopes'
        #     #     ind_l = [1, 2, 3]
        #     #     ind_l = [1]
        #     #     data_name = 'Snopes_incentive_10'
        #
        #     elif dataset2 == 'snopes_ssi':
        #         data_n = 'sp_ssi'
        #         data_addr = 'snopes'
        #         ind_l = [1]
        #         data_name = 'Snopes_ssi'
        #
        #     df = collections.defaultdict()
        #     df_w = collections.defaultdict()
        #     tweet_avg_med_var = collections.defaultdict(list)
        #     tweet_dev_avg_med_var = collections.defaultdict(list)
        #     tweet_dev_avg2 = {}
        #     tweet_dev_med2 = {}
        #     tweet_dev_var2 = {}
        #     tweet_avg2 = {}
        #     tweet_med2 = {}
        #     tweet_var2 = {}
        #     tweet_gt_var = {}
        #
        #     tweet_dev_avg_l2 = []
        #     tweet_dev_med_l2 = []
        #     tweet_dev_var_l2 = []
        #     tweet_avg_l2 = []
        #     tweet_med_l2 = []
        #     tweet_var_l2 = []
        #     tweet_gt_var_l = []
        #     avg_susc = 0
        #     avg_gull = 0
        #     avg_cyn = 0
        #
        #     tweet_abs_dev_avg2 = {}
        #     tweet_abs_dev_med2 = {}
        #     tweet_abs_dev_var2 = {}
        #
        #
        #     tweet_abs_dev_avg_rnd2 = {}
        #     tweet_dev_avg_rnd2 = {}
        #
        #     tweet_skew = {}
        #     tweet_skew_l = []
        #
        #
        #
        #     tweet_kldiv_group = collections.defaultdict()
        #
        #     news_cat_list_tf = [4, 2, 3, 1]
        #     t_f_dict_len = collections.defaultdict(int)
        #     t_f_dict = {}
        #
        #     # if dataset == 'snopes' or dataset == 'snopes_nonpol' or dataset == 'snopes_ssi' or dataset == 'snopes_incentive' or dataset == 'snopes_incentive_10' or dataset == 'snopes_2':
        #     #     news_cat_list_t_f = ['FALSE', 'MOSTLY FALSE', 'MIXTURE', 'MOSTLY TRUE', 'TRUE']
        #     #     news_cat_list_v = [-1, -.5, 0, 0.5, 1]
        #
        #     w_fnb_dict = collections.defaultdict()
        #     w_fpb_dict = collections.defaultdict()
        #     w_apb_dict = collections.defaultdict()
        #     gt_acc = collections.defaultdict()
        #     # for cat in news_cat_list_v:
        #     #     gt_acc[cat] = [0] * (len(news_cat_list_t_f))
        #     # weight_list = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
        #     # weight_list = [-2.5, -0.84, 0.25, 1, -1.127, 1.1, 1.05]
        #     # weight_list = [-2.36, -0.73, 0.53, 0.87, -0.87, 0.93, 1.53]
        #     pt_list = []
        #     gt_list = []
        #     pp2 = 0
        #     pf2 = 0
        #     tpp2 = 0
        #     tpf2 = 0
        #     chi_sq_l2=[]
        #     f_filter = []
        #     ind_l = [1]
        #     dist_list2 = []
        #
        #
        #     for ind in ind_l:
        #
        #         inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted.csv'
        #         inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted.csv'
        #         # df[ind] = pd.read_csv(inp1, sep="\t")
        #
        #
        #         if dataset2=='snopes_gt':
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt.csv'
        #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted_gt.csv'
        #             f_filter = w_id_gt_filter
        #         elif dataset2 == 'snopes_gt_10':
        #             inp1 = remotedir + 'amt_answers_sp_gt_10_claims_exp' + str(
        #                 ind) + '_final_weighted_gt.csv'
        #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(
        #                 ind) + '_weighted_gt.csv'
        #             f_filter = w_id_gt_filter
        #         elif dataset2=='snopes_gt_experts':
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
        #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_claims_exp' + str(ind) + '_weighted_gt_experts.csv'
        #             f_filter = w_id_gt_exp_filter
        #
        #         elif dataset2=='snopes_gt_incentive':
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt.csv'
        #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_weighted_gt.csv'
        #
        #             f_filter = w_id_inc_gt_filter
        #
        #         elif dataset2=='snopes_gt_incentive_experts':
        #             inp1 = remotedir + 'amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_final_weighted_gt_experts.csv'
        #             inp1_w = remotedir + 'worker_amt_answers_' + data_n + '_incentive_claims_exp' + str(ind) + '_weighted_gt_experts.csv'
        #             f_filter = w_id_inc_gt_exp_filter
        #
        #         df[ind] = pd.read_csv(inp1, sep="\t")
        #         # df_w[ind] = pd.read_csv(inp1_w, sep="\t")
        #
        #         df_m = df[ind].copy()
        #
        #         worker_id_list = set(df_m['worker_id'])
        #         filter_l = list(worker_id_list - set(f_filter))
        #         df_m=df_m[df_m['worker_id'].isin(filter_l)]
        #
        #         df_mm=df_m[df_m['tweet_id'].isin(claims_list_all)]
        #
        #
        #         if 'gt' in dataset2:
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 1]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 2]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 3]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 4]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 5]))
        #
        #         else:
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 1]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 2]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 3]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 4]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 5]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 6]))
        #             dist_list2.append(len(df_mm[df_mm['ra'] == 7]))
        #
        #
        #
        #         groupby_ftr = 'tweet_id'
        #         grouped = df_m.groupby(groupby_ftr, sort=False)
        #         grouped_sum = df_m.groupby(groupby_ftr, sort=False).sum()
        #
        #         # tmp_list = df_m[df_m['ra'] == 1].index.tolist()
        #         # df_m['rel_v'][tmp_list] = [weight_list[0]] * len(tmp_list)
        #         # tmp_list = df_m[df_m['ra'] == 2].index.tolist()
        #         # df_m['rel_v'][tmp_list] = [weight_list[1]] * len(tmp_list)
        #         # tmp_list = df_m[df_m['ra'] == 3].index.tolist()
        #         # df_m['rel_v'][tmp_list] = [weight_list[2]] * len(tmp_list)
        #         # tmp_list = df_m[df_m['ra'] == 4].index.tolist()
        #         # df_m['rel_v'][tmp_list] = [weight_list[3]] * len(tmp_list)
        #         # tmp_list = df_m[df_m['ra'] == 5].index.tolist()
        #         # df_m['rel_v'][tmp_list] = [weight_list[4]] * len(tmp_list)
        #         # tmp_list = df_m[df_m['ra'] == 6].index.tolist()
        #         # df_m['rel_v'][tmp_list] = [weight_list[5]] * len(tmp_list)
        #         # tmp_list = df_m[df_m['ra'] == 7].index.tolist()
        #         # df_m['rel_v'][tmp_list] = [weight_list[6]] * len(tmp_list)
        #
        #         # for t_id in grouped.groups.keys():
        #         for t_id in claims_list_all:
        #
        #             if t_id >= 2000:
        #                 continue
        #             pp_tmp2 = 0
        #             pf_tmp2 = 0
        #             tpp_tmp2 = 0
        #             tpf_tmp2 = 0
        #
        #             df_tmp = df_m[df_m['tweet_id'] == t_id]
        #             pt = np.mean(df_tmp['rel_v_b'])
        #             # pt = np.mean(df_tmp['acc'])
        #             gt = list(df_tmp['rel_gt_v'])[0]
        #             if gt!=0:
        #                 # continue
        #                 if gt > 0:
        #                     if pt > 0:
        #                         pp2 += 1
        #                     tpp2 += 1
        #                 if gt < 0:
        #                     if pt < 0:
        #                         pf2 += 1
        #                     tpf2 += 1
        #
        #                 for el_val in list(df_tmp['rel_v_b']):
        #                     if gt > 0:
        #                         if el_val > 0:
        #                             pp_tmp2 += 1
        #                         tpp_tmp2 += 1
        #                     if gt < 0:
        #                         if el_val < 0:
        #                             pf_tmp2 += 1
        #                         tpf_tmp2 += 1
        #
        #
        #                 if gt>0:
        #                     chi_sq_l2.append(pp_tmp2)
        #                     chi_sq_l2.append(tpp_tmp2 - pp_tmp2)
        #                 elif gt<0:
        #                     chi_sq_l2.append(pf_tmp2)
        #                     chi_sq_l2.append(tpf_tmp2 - pf_tmp2)
        #
        #
        #             val_list = list(df_tmp['rel_v'])
        #             val_list_ra = list(df_tmp['ra'])
        #             tweet_avg2[t_id] = np.mean(val_list)
        #             tweet_med2[t_id] = np.median(val_list)
        #             tweet_var2[t_id] = np.var(val_list)
        #
        #             tweet_avg_l2.append(np.var(val_list))
        #
        #             val_list = list(df_tmp['err'])
        #             abs_var_err = [np.abs(x) for x in val_list]
        #
        #             tweet_dev_avg2[t_id] = np.mean(abs_var_err)
        #             tweet_dev_med2[t_id] = np.median(abs_var_err)
        #             tweet_dev_var2[t_id] = np.var(abs_var_err)
        #             tweet_dev_avg_l2.append(np.mean(abs_var_err))
        #
        #     ##################################################
        #     print(pp2 / float(tpp2))
        #     print(pp)
        #     print(pf2 / float(tpf2))
        #     print(pf2)
        #     # print(np.corrcoef(pt_list, gt_list))
        #     # print(scipy.stats.spearmanr(pt_list, gt_list))
        #     # print(remotedir)
        #     # from scipy.stats import chisquare
        #     # Pearson Corr of TPB and Disp|| Spearman corr of TPB and Disp')
        #     print([pp + pf,tpp-pp + tpf-pf], [pp2 + pf2,tpp2-pp2 + tpf2-pf2])
        #     output.write('||' + dataset1 + ' and ' + dataset2 + '||')
        #
        #     if 'gt' in dataset1:
        #         output.write(str(dist_list1[0]) + ', ' + str(dist_list1[1]) + ', ' + str(dist_list1[2]) + ', ' +
        #                      str(dist_list1[3]) + ', ' + str(dist_list1[4]) + '<<BR>> <<BR>>')
        #     else:
        #         output.write(str(dist_list1[0])+ ', '+ str(dist_list1[1])+ ', '+str(dist_list1[2])+ ', '+
        #                      str(dist_list1[3])+ ', '+str(dist_list1[4])+ ', '+str(dist_list1[5])+ ', '+str(dist_list1[6]) + '<<BR>><<BR>>')
        #
        #
        #     if 'gt' in dataset2:
        #         output.write(str(dist_list2[0]) + ', ' + str(dist_list2[1]) + ', ' + str(dist_list2[2]) + ', ' +
        #                      str(dist_list2[3]) + ', ' + str(dist_list2[4]) + '||')
        #
        #     else:
        #         output.write(str(dist_list2[0])+ ', '+ str(dist_list2[1])+ ', '+str(dist_list2[2])+ ', '+
        #                      str(dist_list2[3])+ ', '+str(dist_list2[4])+ ', '+str(dist_list2[5])+ ', '+str(dist_list2[6]) + '||')
        #
        #     # scipy.stats.chisquare(dist_list1)
        #     output.write(str(scipy.stats.chisquare(dist_list1)) + '<<BR>><<BR>>')
        #     output.write(str(scipy.stats.chisquare(dist_list2)) + '||')
        #
        #     try:
        #         out = scipy.stats.chi2_contingency([dist_list1,dist_list2])
        #         output.write(str(out[0]) + '<<BR>>' + str(out[1]))
        #     except:
        #         output.write('NULL')
        #
        #     # print('------ together : ----------')
        #     surv = np.array([[pp + pf,tpp-pp + tpf-pf], [pp2 + pf2,tpp2-pp2 + tpf2-pf2]])
        #     out = scipy.stats.chi2_contingency(surv)
        #     output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]))
        #
        #     # # print('------ True claims : ----------')
        #     # surv = np.array([[pp,tpp-pp], [pp2,tpp2-pp2]])
        #     # out = scipy.stats.chi2_contingency(surv)
        #     # output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]))
        #     #
        #     # # print('------ False claims : ----------')
        #     # surv = np.array([[pf,tpf-pf], [pf2,tpf2-pf2]])
        #     # out = scipy.stats.chi2_contingency(surv)
        #     # output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]) )
        #
        #
        #
        #     # surv = np.array([[pp,tpp-pp,pf,tpf-pf], [pp2,tpp2-pp2,pf2,tpf2-pf2]])
        #     # print(scipy.stats.chi2_contingency(surv))
        #
        #
        #     # comparing TPB in diff surveys
        #     # print('TPB vectors comparing')
        #     # output.write('||' + str(scipy.stats.chisquare(tweet_dev_avg_l)) + '<<BR>>')
        #     # output.write(str(scipy.stats.chisquare(tweet_dev_avg_l2)))
        #
        #     out = scipy.stats.chi2_contingency([tweet_dev_avg_l,tweet_dev_avg_l2])
        #     output.write('||' + str(out[0]) + '<<BR>>' + str(out[1]))
        #     # output.write('||' + str(scipy.stats.mannwhitneyu(tweet_dev_avg_l, tweet_dev_avg_l2)))
        #
        #     # print('TPB vectors comparing correlation pearson')
        #     out = np.corrcoef(tweet_dev_avg_l, tweet_dev_avg_l2)
        #     output.write('||' + str(out[1]))
        #
        #     # print('TPB vectors comparing correlation spearman')
        #     out  = scipy.stats.spearmanr(tweet_dev_avg_l, tweet_dev_avg_l2)
        #     output.write('||' + str(out) + '||\n')
        #
        #
        #
        #     # print('TPB and ')
        #     # print(scipy.stats.mannwhitneyu(tweet_dev_avg_l, tweet_dev_avg_l2))
        # #
        # #
        # #
        # #
        # #



