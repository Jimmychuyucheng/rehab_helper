import json

# 讀取 JSON 檔案
with open(r'C:\Users\Alex\Desktop\Sign_dataset\asl100_select.json', 'r') as file:
    data = json.load(file)

target = 'year'
print(target)
for g in data:
    if g['gloss'] == target:
        for d in g['instances']:
            if d['variation_id'] == 0:
                d['trajectory'] = 13
            elif d['variation_id'] == 1:
                d['trajectory'] = 0
            elif d['variation_id'] == 2:
                d['trajectory'] = 0
            
#　print("gloss {} : {}".format(d['gloss'],i['video_id']))
# print(varation_gloss)
# print(len(varation_gloss))
# 寫入回 JSON 檔案
with open(r'C:\Users\Alex\Desktop\Sign_dataset\asl100_select.json', 'w') as file:
    json.dump(data, file, indent=4)  # indent=4 用於美化輸出，使其易於閱讀


    
