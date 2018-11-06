from rdflib import Graph
import random
import rdflib
from nltk.tokenize import sent_tokenize
import pandas as pd
import io
import fastText
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import gzip
import json
import re
import time
from pathlib import Path
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import scipy.sparse


"""
Print System Info
"""
def print_info():
	intro = PrettyTable(['Causality Search Engine v0 for M11 corpus (Author: Shixiang Zhu, THOR Group)'])
	intro_list = []
	intro_list.append("Causality Search Engine")
	intro.add_row(intro_list)

	intro_list = []
	intro_list.append("Usage: \nStep 1: You need to specify several keywords regarding what you want \nStep 2: Based on the keywords you provided, the system will retrive relevant or relevant and causal sentences")
	intro.add_row(intro_list)
	intro.align['Causality Search Engine system v0 for harvey corpus (Author: Shixiang Zhu, THOR Group)'] = 'l'
	print(intro)
	print(" ")
	
"""
Readin m11 pre-stored data
parameter: roots
return: causal and no_causal data
"""
def readinKWresult(rootC, rootN):
  causal_kw = open(rootC, 'r')
  non_causal_kw = open(rootN, 'r')
  causal_kw = causal_kw.readlines()
  non_causal_kw = non_causal_kw.readlines()
  return causal_kw, non_causal_kw

"""
Get users' input
parameter: 
return: string (users' input)
"""
def readinUser():
	userIn = input("Enter your keywords (separate them by space): ")
	print("Your input is: ", userIn, " Type: ", userIn.__class__)
	print(" ")
	return userIn


"""
Get tfidf vectors
parameter: user's input string, causal list, no_causal list
return: causal, nocausal vec list, user's input vector list
"""
def getTfidf(userIn, causal, no_causal):
	# Merge all sentences
	all_list = []
	all_list.append(userIn)
	all_list = all_list + causal + no_causal

	# TFIDF
	feature_extraction = TfidfVectorizer(lowercase=False, norm = 'l2') # l2 normalization makes sqrt(a^2 + b^2 + c^2) = 1
	featureVec = feature_extraction.fit_transform(all_list)

	# Get vectors for similarity score
	causal_len = len(causal)
	no_causal_len = len(no_causal)
	total_len = featureVec.shape[0]
	user_vec = featureVec[0]
	causal_vec = featureVec[1:causal_len+1]
	no_causal_vec = featureVec[causal_len+1:total_len]
	
	return user_vec, causal_vec, no_causal_vec




"""
Get simlarity score and print result 
parameter: user_vec, causal_vec, no_causal_vec, causal_kw, non_causal_kw
return: 
"""
def getSimScore_and_print(user_vec, causal_vec, no_causal_vec, causal_kw, non_causal_kw):
	user_vec = user_vec.transpose()  
	dict_causal = {} 
	dict_no_causal = {}
	causal_sim = np.dot(causal_vec, user_vec)
	causal_sim = scipy.sparse.coo_matrix(causal_sim)
	no_causal_sim = np.dot(no_causal_vec, user_vec)
	no_causal_sim = scipy.sparse.coo_matrix(no_causal_sim)

	for i,j,v in zip(causal_sim.row, causal_sim.col, causal_sim.data):
		dict_causal[i] = v
	for i,j,v in zip(no_causal_sim.row, no_causal_sim.col, no_causal_sim.data):
		dict_no_causal[i] = v
	causal_sort = [(k, dict_causal[k]) for k in sorted(dict_causal, key=dict_causal.get, reverse=True)]
	no_causal_sort = [(k, dict_no_causal[k]) for k in sorted(dict_no_causal, key=dict_no_causal.get, reverse=True)]
	
	all_dict = {} # Store all the result causal + no_causal, noncausal index = i - len(causal)
	top10 = []
	for k,v in causal_sort:
		all_dict[k] = v
		if len(top10) <= 10 and v > 0.0:
			top10.append((causal_kw[k], v))
		else:
			break
	for k,v in no_causal_sort:
		all_dict[k + len(causal_kw)] = v
		if len(top10) <= 10 and v > 0.0:
			top10.append((non_causal_kw[k], v))
		else:
			break

	# Normal Result:
	print('Causality Search Engine v0 Search Result (Normal): ')
	print(' ')
	all_sort = [(k, all_dict[k]) for k in sorted(all_dict, key=all_dict.get, reverse=True)]
	count = 0
	for k,v in all_sort:
		if count < 5:
			if k < len(causal_kw):
				print(causal_kw[k], v)
			else:
				print(non_causal_kw[k - len(causal_kw)], v)
		count += count

	# Causal Result:
	print(" ")
	print('Causality Search Engine v0 Search Result (Causal): ')
	print(' ')
	if len(top10) == 0:
		print("No Search Result")
	else:
		for k,v in top10[0:5]:
			print(k, v)
			print(" ")
	
def main():
  # PRINT SYSTEM INFO
  print_info()
  # READ IN M11 DATA
  causal_kw, non_causal_kw = readinKWresult('/Users/zhengshuangjing/desktop/work/projects/Causal50/intermedia_files/causal_keyword.txt', '/Users/zhengshuangjing/desktop/work/projects/Causal50/intermedia_files/nocausal_keyword.txt')
  # GET USER'S INPUT:
  user_In = readinUser()
  # GET TFIDF VECTOR
  user_vec, causal_vec, no_causal_vec = getTfidf(user_In, causal_kw, non_causal_kw)
  # GET SIMILARITY SCORE AND PRINT SENTENCES
  getSimScore_and_print(user_vec, causal_vec, no_causal_vec, causal_kw, non_causal_kw)



if __name__ == '__main__':
  main()
