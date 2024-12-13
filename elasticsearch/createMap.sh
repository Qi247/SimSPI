# curl -XPUT --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/line_numbers_dis/_doc/_mapping -H 'Content-Type: application/json' -d '
# {
#  "_doc":{
#    "properties": {
#         "name": {"type": "text"},
#         "line_number_count": {"type": "integer"},
#         "count": {"type": "integer"}
#       }
#   }
# }'

# curl -XPUT --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/segment_numbers_dis/_doc/_mapping -H 'Content-Type: application/json' -d '
# {
#  "_doc":{
#    "properties": {
#         "segment_number_count": {"type": "integer"},
#         "count": {"type": "integer"}
#       }
#   }
# }'

# curl -XPUT --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/before_file_dis/_doc/_mapping -H 'Content-Type: application/json' -d '
# {
#  "_doc":{
#    "properties": {
#         "file_type": {"type": "text"},
#         "count": {"type": "integer"}
#       }
#   }
# }'

# curl -XPUT --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/after_file_dis/_doc/_mapping -H 'Content-Type: application/json' -d '
# {
#  "_doc":{
#    "properties": {
#         "file_type": {"type": "text"},
#         "count": {"type": "integer"}
#       }
#   }
# }'


# curl -XPUT --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/single_analyze/_doc/_mapping -H 'Content-Type: application/json' -d '
# {
#     "_doc":{
#         "properties": {
#             "reduce_line_count": {"type": "integer"},
#             "add_line_count": {"type": "integer"},
#             "control_count": {"type": "integer"},
#             "var_count": {"type": "integer"},
#             "nodes_count": {"type": "integer"},
#             "edges_count": {"type": "integer"}
#         }
#     }
# }'

# curl -XPUT --cacert http_ca.crt -u elastic:TYtEtszTSAmsh9VJZxV4 https://localhost:9200/single_result/_doc/_mapping -H 'Content-Type: application/json' -d '
# {
#  "_doc":{
#    "properties": {
#         "prob_all": {"type": "float"},
#         "prob_vul": {"type": "float"},
#         "prob_sim": {"type": "float"},
#         "nodes_count": {"type": "integer"},
#         "edges_count": {"type": "integer"},
#         "analyze_text": {"type": "text"}
#     }
#   }
# }'





