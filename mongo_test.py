import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["runoobdb"]
dblist = myclient.list_database_names()
mycol = mydb["sites"]

if "runoobdb" in dblist:
    print("数据库已存在！")

# mydict = { "name": "RUNOOB", "alexa": "10000", "url": "https://www.runoob.com" }
 
# x = mycol.insert_one(mydict) 
# print(x)
# mylist = [
#   { "name": "Taobao", "alexa": "100", "url": "https://www.taobao.com" },
#   { "name": "QQ", "alexa": "101", "url": "https://www.qq.com" },
#   { "name": "Facebook", "alexa": "10", "url": "https://www.facebook.com" },
#   { "name": "知乎", "alexa": "103", "url": "https://www.zhihu.com" },
#   { "name": "Github", "alexa": "109", "url": "https://www.github.com" }
# ]
 
# x = mycol.insert_many(mylist)
 
# # 输出插入的所有文档对应的 _id 值
# print(x.inserted_ids)

for x in mycol.find():
  print(x)