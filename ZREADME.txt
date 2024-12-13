主要文件及作用如下(已按顺序排列):

joern: 用于生成代码属性图的软件，主要使用其中的bin/joern-perse和bin/joern命令
get_rawpatch.py: 用于获取补丁文件的脚本
get_ab_file.py: 用于针对补丁获取前后版本文件的脚本
gen_cpg.py+locate_align.py: 用于去除前后版本中修改无关函数的脚本
my_gen_cpg.py: 用于生成代码属性图并处理节点和边关系的脚本(很麻烦，不用看懂，有借鉴其他论文的处理方式)
test.py: 调用preproc下的两个脚本用于解析cpg文件中的节点信息和边信息
- preproc/extract_graph.py: 读取文件并解析成属性，还要解析代码为token序列
- preproc/construct_grpah.py: 节点嵌入和边嵌入工作，生成向量
testdata2-5就是处理后的数据，用于输入模型(分别对应FFmpeg,Linux,QEMU,Wireshark)

dataloader.py: 配合model.py读取数据用的
model.py: 主要模型文件
output-4edges: 保存模型的目录

其他的都是不重要/不必要的文件，不用关心
