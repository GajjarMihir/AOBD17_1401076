import csv
from pprint import pprint
import re
new_list=[]
with open('C:\\Users\\Mihir\\Desktop\\aobd final\\Candidate Profile Data\\Database_Admin.csv','rb') as f:
    reader = csv.reader(f,delimiter=',')
    c_id=[]
    skill=[]
    combine=[]
    new_list=[]
    for col in reader:
        c_id.append(col[0])
        skill.append(col[2])
    combine=zip(c_id,skill)
    #print len(combine)
    for i in range(1,len(combine)):
        new_list.append(combine[i][1])
    str1 = ''.join(new_list)
    str2 = re.sub("[\(\[].*?[\)\]]", "", str1)
    #print str2
    str2 = str2.replace('.',',').replace(';',',').replace('\xe2\x80\x93',',')
    list_1 = str2.split(',')
    pprint(list_1)
    str4 = ''.join(list_1)
    
    

    with open("new_skills.csv", "w") as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(list_1)
