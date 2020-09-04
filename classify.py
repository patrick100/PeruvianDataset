import os
import csv

origin_path = "./OBJs"
target_path = "./Peruvian-DB"


categories = ['animal-bottle',
'animal-head',
'bowl',
'cone-vase',
'cuenco',
'flat-canteen',
'jar',
'lebrillo',
'olla',
'plate',
'statue',
'vase',
'vessel']

for i in range(len(categories)):
	command = " ".join(["mkdir",target_path+"/"+categories[i]])
	os.system(command)
	f_train_command = " ".join(["mkdir",target_path+"/"+categories[i]+"/train"])
	os.system(f_train_command)
	f_test_command = " ".join(["mkdir",target_path+"/"+categories[i]+"/test"])
	os.system(f_test_command)



reader = csv.DictReader(open("classifier.csv"))
for row in reader:
	#print(row['name'],row['category'],row['split'])
	if(row['category']!='trash'):
		command = " ".join(["cp",origin_path+"/"+row['name']+".obj",target_path+"/"+row['category']+"/"+row['split']+"/"+row['name']+".obj"])	
		#print(command)
		os.system(command)

