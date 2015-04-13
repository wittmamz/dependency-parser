import codecs
import time

class Token(object):
	def __init__( self, line ):
		entries = line.split('\t')
		self.form = entries[0]
		self.pos = entries[1]


def sentences( filestream ):

	sentence = []
	for line in filestream:
		line = line.rstrip()
		if line:
			sentence.append(Token(line))
		elif sentence:
			yield sentence
			sentence = []
	if sentence:
		yield sentence

def evaluate(posDict):
	goldPosSet = set()
	predPosSet = set()
	uniqueTags = {}
	uniqueTagsScores = {}
	correctPredictions = 0

	for key in posDict:

		# creates sets of gold and prediction tags

		goldPosSet.add(posDict[key][0])
		predPosSet.add(posDict[key][1])
	

	for key in posDict:

		# creates a dictionary with every existing tag (either existing only in gold, only in 
		# prediction, or both) the key and a dictionary each as the value containing TP, FN, FP
		# as the key and zero as the value.

		if posDict[key][0] not in uniqueTags.keys():
			uniqueTags[posDict[key][0]] = {'TP':0,'FN':0,'FP':0}
		if posDict[key][1] not in uniqueTags.keys():
			uniqueTags[posDict[key][1]] = {'TP':0,'FN':0,'FP':0}

	for key in posDict:

		# Computes TP, FN and FP for each unique tag.

		if posDict[key][0] == posDict[key][1]:
			correctPredictions+=1
			uniqueTags[posDict[key][0]]['TP']+=1
		else:
			uniqueTags[posDict[key][0]]['FN']+=1
			uniqueTags[posDict[key][1]]['FP']+=1

	for pos in uniqueTags.keys():

		# Computes Precision, Recall, Accuracy and F-Score for each Tag based on TP, FN, FP.

		uniqueTagsScores[pos]={'Precision':0.00, 'Recall':0.00, 'Accuracy':0.00, 'F-Score':0.00}
		if uniqueTags[pos]['TP']+uniqueTags[pos]['FP'] == 0:
			uniqueTagsScores[pos]['Precision'] = 0.00
		else:
			uniqueTagsScores[pos]['Precision']=round((float(uniqueTags[pos]['TP']))/(float(uniqueTags[pos]['TP'])+float(uniqueTags[pos]['FP']))*100.00, 2)

		if uniqueTags[pos]['TP']+uniqueTags[pos]['FN'] == 0:
			uniqueTagsScores[pos]['Recall'] = 0.00
		else: uniqueTagsScores[pos]['Recall']=round((float(uniqueTags[pos]['TP']))/(float(uniqueTags[pos]['TP'])+float(uniqueTags[pos]['FN']))*100.00, 2)

		if uniqueTags[pos]['TP']+uniqueTags[pos]['FP']+uniqueTags[pos]['FN'] == 0:
			uniqueTagsScores[pos]['Accuracy'] = 0.00
		else: uniqueTagsScores[pos]['Accuracy']=round(float(uniqueTags[pos]['TP'])/(float(uniqueTags[pos]['TP'])+float(uniqueTags[pos]['FN'])+float(uniqueTags[pos]['FP']))*100.00, 2)

		if uniqueTagsScores[pos]['Precision']+uniqueTagsScores[pos]['Recall'] == 0.00:
			uniqueTagsScores[pos]['F-Score'] = 0.00
		else: uniqueTagsScores[pos]['F-Score'] = round((2*float(uniqueTagsScores[pos]['Precision'])*float(uniqueTagsScores[pos]['Recall']))/(float(uniqueTagsScores[pos]['Precision'])+float(uniqueTagsScores[pos]['Recall'])), 2)
	

	precisionSum = 0.0
	recallSum = 0.0
	fscoreSum = 0.0
	
	for pos in uniqueTagsScores.keys():
		precisionSum += uniqueTagsScores[pos]['Precision']
		recallSum += uniqueTagsScores[pos]['Recall']
		fscoreSum += uniqueTagsScores[pos]['F-Score']

	overallPrecision = precisionSum/float(len(uniqueTagsScores))
	overallRecall = recallSum/float(len(uniqueTagsScores))
	overallFscore = fscoreSum/float(len(uniqueTagsScores))

	overallAccuracy = (float(correctPredictions)/float(len(posDict)))*100
	falseTags = len(posDict)-correctPredictions

	print "Total Predictions:\t"+str(len(posDict))
	print "Correct Predictions:\t"+str(correctPredictions)
	print "False Predictions:\t"+str(falseTags)
	print ""
	print "Overall Accuracy:\t"+str(overallAccuracy)
	print "Overall Precision:\t"+str(overallPrecision)
	print "Overall Recall:\t"+str(overallRecall)
	print "Overall F-Score:\t"+str(overallFscore)
	print ""

	print "Tagwise Accuracy, Precision, Recall and F-Score:\n"
	for pos in uniqueTagsScores.keys():
		print pos+"\tAccuracy: "+str(uniqueTagsScores[pos]['Accuracy'])+"\tPrecision: "+str(uniqueTagsScores[pos]['Precision'])+"\tRecall: "+str(uniqueTagsScores[pos]['Recall'])+"\tF-Score: "+str(uniqueTagsScores[pos]['F-Score'])


if __name__=='__main__':
	t0 = time.time()
	import argparse
	argpar = argparse.ArgumentParser(description='Creates a feature representation for each word in a given file in CoNLL09 format')
	
	argpar.add_argument('-g','--gold',dest='gold',help='gold',required=True)
	argpar.add_argument('-p','--prediction',dest='prediction',help='prediction',required=True)
	#argpar.add_argument('-o','--output',dest='outputfile',help='output file',required=True)
	args = argpar.parse_args()

	posDict = {}

	#outstream = open(args.outputfile,'w')

	counter=0
	for sid, sentence in enumerate(sentences(codecs.open(args.gold,encoding='utf-8'))):
		for tid,token in enumerate(sentence):
			posDict[counter] = []
			posDict[counter].append(token.pos)
			counter+=1
	counter=0
	for sid, sentence in enumerate(sentences(codecs.open(args.prediction,encoding='utf-8'))):
		for tid,token in enumerate(sentence):
			posDict[counter].append(token.pos)
			counter+=1

	evaluate(posDict)
	t1 = time.time()
	print "\n"
	print "Time: "+ str(t1-t0) + "sec."
	#print >> outstream, " "
	#outstream.close()

