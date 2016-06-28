#!/usr/bin/python
# -*- coding: utf-8 -*-

DatabeseName = "filtered2.db"
write_file = "6_28debug.txt"

import sqlite3
import random
import gensim
import MeCab
import numpy as np
import copy
from gensim import corpora, models, similarities
import time
import re
import math
from progressbar import ProgressBar
#stoplist = set('/ // @ # ( ) : ! , . 1 2 3 4 5 6 7 8 9 follow'.split())
stoplist = set([])

def preprocess_id_list(id_list):
	id_list = str(id_list)
	id_list = id_list.replace("(","")
	id_list = id_list.replace(")","")
	id_list = id_list.replace("u'","")
	id_list = id_list.replace("'","")
	if(id_list[-1] == ","):
		id_list = id_list.rstrip(",")
	id_list = id_list.split(",")
	id_list.reverse()
	return id_list

def extractKeyword(text):
	#print text
	tagger = MeCab.Tagger("-d /var/lib/mecab/dic/ipadic-utf8 -Owakati")
	encoded_text = text.encode('utf-8')
	node = tagger.parseToNode(encoded_text).next
	keywords = []
	while node:
		if (node.feature.split(",")[0] == "名詞" or node.feature.split(",")[0] == "動詞"): 
			keywords.append(node.surface)
		node = node.next
	return keywords

def splitDocument(documents):
	splitted_documents = []
	for d in documents:
		keywords = extractKeyword(d)
		splitted_documents.append(' '.join(keywords))
	return splitted_documents

def removeStoplist(documents, stoplist):
	stoplist_removed_documents = []
	for document in documents:
		words = []
		for word in document.lower().split():
			if word not in stoplist:
				words.append(word)
		stoplist_removed_documents.append(words)
	return stoplist_removed_documents

def removeTokensOnce(documents, tokens_once):
	token_removed_documents = []
	for document in documents:
		#print document
		words = []
		for word in document:
			#print word
			if word not in tokens_once:
				words.append(word)
		if len(words) != 0 and ' '.join(words) != '':
			token_removed_documents.append(words)
	return token_removed_documents

def read_DB(cur,tweet_id):
		return cur.execute( "SELECT sentence FROM c_short WHERE tweet_id=" + str(tweet_id)).fetchone()

def make_dialogue(cur,id_list):
	dialogue = []
	for tweet_id in id_list:
		dialogue.append(read_DB(cur,tweet_id))
	return dialogue

def split_data(dialogue,pop_num):
	for i in range(0,pop_num):
		dialogue.pop()
	#print dialogue
	return (dialogue)

def vecNormalization(vector):
	summation = 0
	summation = sum([x[1] for x in vector])
	if(summation == 0):
		return vector
	map(lambda x:x[1]/summation, vector)
	return vector

def cosSimilarity(v1,v2):
	v1 = vecNormalization(v1)
	v2 = vecNormalization(v2)
	sumMolecule = 0
	for element1 in v1:
		element2 = filter(lambda x: x[0] == element1[0],v2)
		if(element2 != []):
			sumMolecule = sumMolecule + (element1[1] * element2[0][1])
	#print sumMolecule
	return sumMolecule

#what number is answer utterance? 
def ranking_data(dia_his,co_ID,utterance_list,dictionary,lda,utt_vec_lda_list):
	print ("co_ID:" + str(co_ID))
	#dia_his assign lda
	utterance_ranking = []
	ansNumber = 0
	for dia_tuple in dia_his:
		if(isinstance(dia_tuple,tuple)):
			for dia_str in dia_tuple:
				his_vec_bow = dictionary.doc2bow(extractKeyword(atmarkDelete(dia_str)))
				#print his_vec_bow
		else:
			print("############################")
	his_vec_lda = lda[his_vec_bow]
	for utterance,utt_vec_lda in zip(utterance_list,utt_vec_lda_list):
		if(isinstance(his_vec_lda,list) and isinstance(utt_vec_lda,list)):
			utterance_ranking.append([utterance[0],utterance[2],cosSimilarity(his_vec_lda,utt_vec_lda)])
	# memory problem
	#print ("utterance_ranking\n\n") 
	utterance_ranking = sorted(utterance_ranking, key=lambda x: x[2])[::-1]
	#print utterance_ranking
	answer = filter(lambda x: x[1] == co_ID,utterance_ranking)
	if(answer != []):
		ansNumber = utterance_ranking.index(answer[0])
		#print("utterance_ranking[ansNumber][0]" + str(utterance_ranking[ansNumber][0].encode("utf_8")) + "\n")
	else:
		print("co_ID:" + str(co_ID))
		print("データに不備があるようです。")
	return ansNumber

def utt_vec_lda_listMake(utterance_list,dictionary,pop_num,cur):
	print ("lda_list making...!")
	utt_vec_lda_list = []
	j = 0
	sums = 0.0
	#print len(utterance_list)
	for utterance in utterance_list:
		sucUtterance = utterance[0]
		for i in range(1,pop_num):
			dbTuple = cur.execute( "SELECT sentence FROM c_short WHERE tweet_id=" + str(utterance[1])).fetchone()
			#print dbTuple[0]
			sucUtterance = sucUtterance + "  " + dbTuple[0]
		#print sucUtterance
		utt_vec_bow = dictionary.doc2bow(extractKeyword(atmarkDelete(sucUtterance)))
		#print utt_vec_bow
		utt_vec_lda = lda[utt_vec_bow]
		utt_vec_lda_list.append(utt_vec_lda)
		if (j % 1000 == 0):
			print(str(float(j) / len (utterance_list) * 100) + " % finished ")
			# sums = sums + (float(j) / len (utterance_list)) * 100
			# p.update(sums)
		j = j + 1
	print ("utt_vec_lda_list making finished!!!")
	return utt_vec_lda_list	

def atmarkDelete(sentence):
	sentence = re.sub(r'@[0-9a-zA-Z_]+', " ", sentence)
	return sentence

def output_result(sentence,result,f):
	print(sentence + result)
	f.write(sentence + result + "\n")

if __name__ == '__main__':
	starttime = time.clock()
	pop_num = 2
	print("process start!!")
	# splitted_documents = splitDocument([u"今回は、一行ごとにツイッターユーザーのプロフィールが書かれたテキストファイルを入力データとして使いました。",u"今日"])
	# tokens_once = set([])
	# stoplist_removed_documents = removeStoplist(splitted_documents, stoplist)
	# preprocessed_documents = removeTokensOnce(stoplist_removed_documents, tokens_once)
	dictionary = gensim.corpora.Dictionary.load_from_text('Topic_Model/hatena2_wordids.txt')
	#bow_corpus = [dictionary.doc2bow(d) for d in preprocessed_documents]
	lda = gensim.models.LdaModel.load('Topic_Model/hatena2_lda.model')
	conn = sqlite3.connect(DatabeseName)
	cur = conn.cursor()
	id_lists = cur.execute( "select idstr,dialogue_id from c_short WHERE repto = 0").fetchall()
	utterance_list = cur.execute( "select sentence,repto,dialogue_id from c_short WHERE repfrom = 0").fetchall()
	rank_array = []
	rank_text = []
	utt_vec_lda_list = utt_vec_lda_listMake(utterance_list,dictionary,pop_num,cur)
	for id_list,label in id_lists:
		id_list = preprocess_id_list(id_list)
		dialogue = make_dialogue(cur,id_list) #for 文の中でデータベースを読み込むのか…
		history = copy.copy(dialogue)
		dia_his = split_data(history,pop_num)
		rank = ranking_data(dia_his,label,utterance_list,dictionary,lda,utt_vec_lda_list)
		print ("ranking:" + str(rank))
		if(rank != 0):
			rank_array.append(rank)
			rank_text.append((rank,dialogue))
		print("process " + str(((label) * 100 / len(id_lists)) ) + "% finished\n")
	mean = sum(rank_array)/len(rank_array)
	output_result("rank_array_mean:",str(mean),f)
	# print("rank_array_mean:" + str(mean))
	var = 0
	under1000 = 0
	under10000 = 0
	over30000 = 0
	for x in rank_array:
		var += (x**2) - (mean**2) 
		if(x < 1000):
			under1000 = under1000 + 1
		if(x < 10000):
			under10000 = under10000 + 1
		if(x > 30000):
			over30000 = over30000 + 1	
	output_result("ranking_under 1000:",str(under1000),f)
	output_result("ranking_under 10000:",str(under10000),f)
	output_result("ranking_over 30000:",str(over30000),f)
	output_result("rank_array_variable:",str(var),f)
	output_result("rank_array_standard_deviation:",str(var**(1/2)),f)
	output_result("time is ",str(time.clock() - starttime),f)
	# print("ranking_under 10000:" + str(under10000))
	# print("ranking_over 30000:" + str(over30000))
	# print("rank_array_variable:" + str(var))
	# print("rank_array_variable:" + str(var**(1/2)))
	# print("time is " + str(time.clock() - starttime) + " seconds")
	rank_text = sorted(rank_text, key=lambda x: x[0])[::-1]
	f = open(write_file,"w")
	for x in rank_text:
		f.write("ranking:" + str(x[0]) + "\n")
		for y in x[1]:
			if(y != None):
				f.write(y[0].encode("utf_8") + "\n")
		f.write("\n###################\n")
	f.close()