import csv  
import json  
  
# Open the CSV  
f = open('train.csv', 'rU')  
# Change each fieldname to the appropriate field name. I know, so difficult.  
reader = csv.DictReader(f, fieldnames = ("id","qid1","qid2","question1", "question2", "is_duplicate"))  
# Parse the CSV into JSON  
next(reader, None)
out = json.dumps([row for row in reader])  
print("JSON parsed!")  
# Save the JSON  
f = open('dataset.json', 'w')  
f.write(out)  
print("JSON saved!")  