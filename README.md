# cnchar-comp-extr
Extraction of Chinese Character Components in Digital Ink based on Recognition Scores

1、下载POT数据
https://nlpr.ia.ac.cn/databases/handwriting/Download.html

2、解压缩放到data/casia目录下
casia/Pot1.0Test
casia/Pot1.0Train

3、CASIA数据转换为jsonl格式
python pot2cjl.py ./data/casia/Pot1.0Test ./data/casia/pot1.0_test.casia.jsonl
python pot2cjl.py ./data/casia/Pot1.0Train ./data/casia/pot1.0_train.casia.jsonl

4、CASIA的jsonl格式数据，样本数据归一化到指定大小，到转换为dink的jsonl格式
python cjl2djl.py ./data/casia/pot1.0_test.casia.jsonl ./data/dink/pot1.0_test.dink_256x256.jsonl
python cjl2djl.py ./data/casia/pot1.0_train.casia.jsonl ./data/dink/pot1.0_train.dink_256x256.jsonl

5、生成训练用样本数据
