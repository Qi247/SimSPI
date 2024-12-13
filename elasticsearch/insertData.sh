# data=(7 6 1 0 339 892 374 593 328 404 293 306 220 276 188 206 154 175 143 178 113 129 126 130 102 127 103 107 95 106 94 89 71 89 76 87 53 85 83 73 51 50 47 45 55 46 55 63 38 45)

# for(( i=0;i<${#data[@]};i++)) do
#   echo $i,${data[i]}
#   curl -XPOST --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/line_numbers_dis/_doc/${i}?pretty -H 'Content-Type: application/json' -d '
#   {
#     "name":'$i',
#     "line_number_count":'$i',
#     "count":'${data[i]}'
#   }'
# done 

# data=(14 3234 1814 1164 846 558 463 319 257 208 139 129 113 78 68 58 48 47 52 44)

# for(( i=0;i<${#data[@]};i++)) do
#   echo $[$i+1],${data[i]}
#   curl -XPOST --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/segment_numbers_dis/_doc/${i}?pretty -H 'Content-Type: application/json' -d '
#   {
#     "name":'$[$i+1]',
#     "segment_number_count":'$[$i+1]',
#     "count":'${data[i]}'
#   }'
# done 

# ks=('c' 'cpp' 'h' 'hpp' 'cc' 'java' 'php' 'go' 'js' 'py' 'pl' 'json' 'rb' 'others')
# vs=(5602 1033 3746 201 124 1096 233 12 498 262 44 363 24 2300)
# for(( i=0;i<${#ks[@]};i++)) do
#   echo ${ks[i]},${vs[i]}
#   curl -XPOST --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/before_file_dis/_doc/${i}?pretty -H 'Content-Type: application/json' -d '
#   {
#     "file_type":"'${ks[i]}'",
#     "count":'${vs[i]}'
#   }'
# done 


# new_ks=(
#     'c' 'cpp' 'h' 'hpp' 'cc' 'java' 'php' 'go' 'js' 'py' 'others'
# )
# new_vs=(5602 1033 3746 201 124 1096 233 12 498 262 123)
# for(( i=0;i<${#new_ks[@]};i++)) do
#   echo ${new_ks[i]},${new_vs[i]}
#   curl -XPOST --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/after_file_dis/_doc/${i}?pretty -H 'Content-Type: application/json' -d '
#   {
#     "file_type":"'${new_ks[i]}'",
#     "count":'${new_vs[i]}'
#   }'
# done 

# 3 9 5 7 37 53
# curl -XPOST --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/single_analyze/_doc/${i}?pretty -H 'Content-Type: application/json' -d '
# {
#   "reduce_line_count": 3,
#   "add_line_count":9,
#   "control_count": 5,
#   "var_count": 7,
#   "nodes_count": 37,
#   "edges_count":53
# }'

# curl -XPOST --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/single_result/_doc/${i}?pretty -H 'Content-Type: application/json' -d '
# {
#   "prob_all": 0.87,
#   "prob_vul": 0.81,
#   "prob_sim": 0.91,
#   "nodes_count": 23,
#   "edges_count": 36,
#   "analyze_text": "分析: 代码修改前后相似度较高，代码中控制单元和逻辑运算符数量较高，可判定为安全补丁"
# }'