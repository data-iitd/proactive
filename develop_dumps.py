import pickle
import numpy as np
import sys, pdb, os

folder = sys.argv[1]
scale = 10.0
thresh = 0
max_t = 0.0
min_t = 10000.0

with open('Datasets/'+folder+'/train_ev.txt', 'r') as in_file:
	eventTrain = [[int(y) for y in x.strip().split()] for x in in_file]

with open('Datasets/'+folder+'/test_ev.txt', 'r') as in_file:
	eventTest = [[int(y) for y in x.strip().split()] for x in in_file]

with open('Datasets/'+folder+'/train_ti.txt', 'r') as in_file:
	timeTrain = [[float(y) for y in x.strip().split()] for x in in_file]

with open('Datasets/'+folder+'/test_ti.txt', 'r') as in_file:
	timeTest = [[float(y) for y in x.strip().split()] for x in in_file]

with open('Datasets/'+folder+'/train_go.txt', 'r') as in_file:
	goalTrain = [int(x) for x in in_file]

with open('Datasets/'+folder+'/test_go.txt', 'r') as in_file:
	goalTest = [int(x) for x in in_file]

# pdb.set_trace()

cat_list = []

# Getting max and min for times
for i in range(len(timeTrain)):
	for j in range(len(timeTrain[i])):
		cat_list.append(eventTrain[i][j])
		val = float(timeTrain[i][j])
		if(val > max_t):
			max_t = val

		if(val < min_t):
			min_t = val

unique = np.unique(np.asarray(cat_list))
len_cats = len(unique)
len_goals = len(np.unique(np.asarray(goalTrain)))

if 0 in unique:
	incr = 1
else:
	incr = 0

# Updating Train
for i in range(len(timeTrain)):
	for j in range(len(timeTrain[i])):
		val = float(timeTrain[i][j])
		new_val = float(scale * ((val - min_t)/(max_t - min_t)))
		timeTrain[i][j] = new_val
		eventTrain[i][j] += incr 

# Updating Test
for i in range(len(timeTest)):
	for j in range(len(timeTest[i])):
		val = float(timeTest[i][j])
		new_val = float(scale * ((val - min_t)/(max_t - min_t)))
		timeTest[i][j] = new_val
		eventTest[i][j] += incr 

# try:
# 	os.mkdir('Datasets/RMTPP_Scaled/'+folder)
# except:
# 	print(folder +" already there")

# event_train = open('Datasets/RMTPP_Scaled/'+folder+'/event_train.txt', "w")
# time_train = open('Datasets/RMTPP_Scaled/'+folder+'/time_train.txt', "w")
# event_test = open('Datasets/RMTPP_Scaled/'+folder+'/event_test.txt', "w")
# time_test = open('Datasets/RMTPP_Scaled/'+folder+'/time_test.txt', "w")

# # Writing Train: RMTPP
# for i in range(len(timeTrain)):
# 	for j in range(len(timeTrain[i]) - 1):
# 		event_train.write(str(eventTrain[i][j]))
# 		event_train.write(" ")

# 		time_train.write(str(timeTrain[i][j]))
# 		time_train.write(" ")

# 	event_train.write(str(eventTrain[i][j+1]))
# 	time_train.write(str(timeTrain[i][j+1]))
	
# 	event_train.write("\n")
# 	time_train.write("\n")

# # Writing Test: RMTPP
# for i in range(len(timeTest)):
# 	for j in range(len(timeTest[i]) - 1):
# 		event_test.write(str(eventTest[i][j]))
# 		event_test.write(" ")

# 		time_test.write(str(timeTest[i][j]))
# 		time_test.write(" ")

# 	event_test.write(str(eventTest[i][j+1]))
# 	time_test.write(str(timeTest[i][j+1]))
	
# 	event_test.write("\n")
# 	time_test.write("\n")

# event_train.close()
# time_train.close()
# event_test.close()
# time_test.close()

def mk_dict(len_cats):
	temp = {}
	temp['test'] = []
	temp['dev'] = []
	temp['devtest'] = []
	temp['dim_process'] = len_cats
	temp['dim_goals'] = len_goals
	temp['train'] = []
	temp['args'] = []
	# temp['goals'] = []
	return temp

train_dict = mk_dict(len_cats)
test_dict = mk_dict(len_cats)
dev_dict = mk_dict(len_cats)

# Writing Train
train_dict['train'] = []
for i in range(len(timeTrain)):
	prev = 0
	temp_list = []
	for j in range(len(timeTrain[i])):
		temp_dict = {}
		val = float(timeTrain[i][j])		
		temp_dict['time_since_start'] = val
		diff = val - prev
		if diff <= 0.0:
			diff = 0.00001
		temp_dict['time_since_last_event'] = diff
		temp_dict['type_event'] = eventTrain[i][j] - 1
		temp_dict['type_goal'] = goalTrain[i]
		prev = val
		temp_list.append(temp_dict)
	train_dict['train'].append(temp_list)
	# train_dict['goals'].append(goalTrain[i])

# Writing Test
test_dict['test'] = []
for i in range(len(timeTest)):
	prev = 0
	temp_list = []
	for j in range(len(timeTest[i])):
		temp_dict = {}
		val = float(timeTest[i][j])
		temp_dict['time_since_start'] = val
		diff = val - prev
		if diff <= 0.0:
			diff = 0.00001
		temp_dict['time_since_last_event'] = diff
		temp_dict['type_event'] = eventTest[i][j] - 1
		temp_dict['type_goal'] = goalTest[i]
		prev = val
		temp_list.append(temp_dict)
	test_dict['test'].append(temp_list)
	# test_dict['goals'].append(goalTest[i])

# # Writing Dev: Neural Hawkes
# len_dev = int(0.1 * len(timeTrain))
# dev_dict['dev'] = []
# for i in range(len_dev):
# 	prev = 0
# 	temp_list = []
# 	for j in range(len(timeTrain[i])):
# 		temp_dict = {}
# 		val = timeTrain[i][j]
# 		temp_dict['time_since_start'] = val
# 		diff = val - prev
# 		if diff <= 0.0:
# 			diff = 0.00001
# 		temp_dict['time_since_last_event'] = diff
# 		temp_dict['type_event'] = eventTrain[i][j] - 1
# 		prev = val
# 		temp_list.append(temp_dict)
# 	dev_dict['dev'].append(temp_list)
# 	dev_dict['goals'].append(goalTrain[i])

try:
	os.mkdir('Dataset_Dumps/'+folder)
except:
	print(folder +" already there")

pickle.dump(train_dict, open('Dataset_Dumps/'+folder+'/train.pkl', 'wb'))
pickle.dump(test_dict, open('Dataset_Dumps/'+folder+'/test.pkl', 'wb'))
# pickle.dump(dev_dict, open('Dataset_Dumps/'+folder+'/dev.pkl', 'wb'))
# pickle.dump(test_dict, open('Dataset_Dumps/'+folder+'/test1.pkl', 'wb'))

print("Done for "+folder+" with num_cats "+str(len_cats) + " with num_goals "+str(len_goals))