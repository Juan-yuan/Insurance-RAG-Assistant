import os
import random
import sys

from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker, RRFRanker

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from runtime_env import require_env


def operate_db():
    client = MilvusClient(uri=require_env('MILVUS_URI'))
    database_name = require_env('MILVUS_DATABASE_NAME')
    databases = client.list_databases()
    if database_name not in databases:
        client.create_database(db_name=database_name)
        print(f'Created database: {database_name} successfully')
    client.using_database(db_name=database_name)
    print(f'Using database: {database_name} successfully')
    return client

client = operate_db()

def operate_table():
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name='scalar', datatype=DataType.VARCHAR, max_length=256, description='Scalar field')
    client.create_collection(collection_name='demo_v1', schema=schema)
    print('Created Collection: demo_v1 successfully')

def operate_index():
    prepare_indexs = client.prepare_index_params()
    prepare_indexs.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)
    res = client.list_indexes(collection_name='demo_v1')
    print(f'res: {res}')
    res1 = client.describe_index(collection_name='demo_v1', index_name='vector_index')
    print(f'res1: {res1}')
    print('-' * 30)
    print(f'Collection load state: {client.get_load_state(collection_name="demo_v1")}')
    client.load_collection(collection_name='demo_v1')
    print(f'Collection load state: {client.get_load_state(collection_name="demo_v1")}')
    client.release_collection(collection_name='demo_v1')
    print(f'Collection load state: {client.get_load_state(collection_name="demo_v1")}')
    client.drop_index(collection_name='demo_v1', index_name='vector_index')
    prepare_indexs.add_index(field_name='scalar', index_type='', index_name='scalar_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)
    res = client.list_indexes(collection_name='demo_v1')
    print(f'res: {res}')

def operate_entity():
    if not client.has_collection(collection_name="demo_v2"):
        client.create_collection(collection_name='demo_v2', dimension=5, metric_type='IP')
    data = [
        {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                             0.9029438446296592], "color": "pink_8682"},
        {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501,
                             0.838729485096104], "color": "red_7025"},
        {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185,
                             0.20785793220625592], "color": "orange_6781"},
        {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995,
                             0.95791889146345], "color": "pink_9298"},
        {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184,
                             0.30337481143159106], "color": "red_4794"},
        {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383,
                             -0.1446277761879955], "color": "yellow_4222"},
        {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192,
                             -0.8984947637863987], "color": "red_9392"},
        {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709,
                             0.5378064918413052], "color": "grey_8510"},
        {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872,
                             -0.6140360785406336], "color": "white_9381"},
        {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717,
                             -0.6980531615588608], "color": "purple_4976"}
    ]
    client.upsert(collection_name='demo_v2', data=data)
    client.create_partition(collection_name='demo_v2', partition_name='partitionA')
    data = [
        {"id": 10, "vector": [-0.5570353903748935, -0.8997887893201304, -0.7123782431855732, -0.6298990746450119,
                              0.6699215060604258], "color": "red_1202"},
        {"id": 11, "vector": [0.6319019033373907, 0.6821488267878275, 0.8552303045704168, 0.36929791364943054,
                              -0.14152860714878068], "color": "blue_4150"},
        {"id": 12, "vector": [0.9483947484855766, -0.32294203351925344, 0.9759290319978025, 0.8262982148666174,
                              -0.8351194181285713], "color": "orange_4590"},
        {"id": 13, "vector": [-0.5449109892498731, 0.043511240563786524, -0.25105249484790804, -0.012030655265886425,
                              -0.0010987671273892108], "color": "pink_9619"},
        {"id": 14, "vector": [0.6603339372951424, -0.10866551787442225, -0.9435597754324891, 0.82302442649483184,
                              -0.7986720938400362], "color": "orange_4863"},
        {"id": 15, "vector": [-0.8825129181091456, -0.9204557711667729, -0.935350065513425, 0.5484069690287079,
                              0.24448151140671204], "color": "orange_7984"},
        {"id": 16, "vector": [0.6285586391568163, 0.5389064528263487, -0.3163366239905099, 0.22036279378888013,
                              0.15077052220816167], "color": "blue_9010"},
        {"id": 17, "vector": [-0.20151825016059233, -0.905239387635804, 0.6749305353372479, -0.7324272081377843,
                              -0.33007998971889263], "color": "blue_4521"},
        {"id": 18, "vector": [0.2432286610792349, 0.01785636564206139, -0.651356982731391, -0.35848148851027895,
                              -0.7387383128324057], "color": "orange_2529"},
        {"id": 19, "vector": [0.055512329053363674, 0.7100266349039421, 0.4956956543575197, 0.24541352586717702,
                              0.4209030729923515], "color": "red_9437"}
    ]
    client.upsert(collection_name='demo_v2', data=data, partition_name='partitionA')
    client.upsert(collection_name='demo_v2', data=data)

def operate_query():
    result1 = client.search(
        collection_name='demo_v2',
        data=[[0.19886812567710876,0.060235604643821716,0.697696328163147,0.2614474594593048,0.8387295007705688]],
        limit=2,
        search_params={'metric_type': 'IP'},
        output_fields=['id', 'vector', 'color']
    )
    print(f'result1: {result1}')
    print('♣' * 30)
    result2 = client.search(
        collection_name='demo_v2',
        data=[
                [0.19886812567710876, 0.060235604643821716, 0.697696328163147, 0.2614474594593048, 0.8387295007705688],
                [0.31720051169395447, 0.971904456615448, -0.369811475276947, -0.48608946800231934, 0.9579188823699951]
            ],
        limit=2,
        search_params={'metric_type': 'IP'},
        output_fields=['id', 'vector']
    )
    print(f'result2: {result2}')
    print('♣' * 30)
    result3 = client.search(
        collection_name='demo_v2',
        data=[[0.6319019198417664,0.682148814201355,0.8552303314208984,0.36929792165756226,-0.14152860641479492]],
        limit=5,
        search_params={'metric_type': 'IP'},
        partition_names=['partitionA']
    )
    print(f'result3: {result3}')
    print('♣' * 30)
    result4 = client.search(
        collection_name='demo_v2',
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],
        limit=5,
        search_params={'metric_type': 'IP'},
        output_fields=['vector', 'color']
    )
    print(f'result4: {result4}')
    print('♣' * 30)
    result5 = client.search(
        collection_name='demo_v2',
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],
        limit=5,
        search_params={'metric_type': 'IP'},
        output_fields=['color'],
        filter='color like "red%"'
    )
    print(f'result5: {result5}')
    print('♣' * 30)
    search_params = {
        'metric_type': 'IP',
        'params': {
            "radius": 0.8,
            "range_filter": 1
        }
    }
    result6 = client.search(
        collection_name='demo_v2',
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],
        limit=5,
        search_params=search_params,
        output_fields=['color'],
    )
    print(f'result6: {result6}')

def complex_query_add_data():
    schema = client.create_schema(enable_dynamic_field=False)
    schema.add_field(field_name='film_id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='filmVector', datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name='posterVector', datatype=DataType.FLOAT_VECTOR, dim=5)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name='filmVector', index_type='IVF_FLAT', metric_type='L2', params={'nlist': 128})
    index_params.add_index(field_name='posterVector', index_type='', metric_type='COSINE')
    client.create_collection(collection_name='demo_v3', schema=schema, index_params=index_params)
    entities = []
    for _ in range(1000):
        film_id = random.randint(1, 10000)
        film_vector = [random.random() for _ in range(5)]
        poster_vector = [random.random() for _ in range(5)]
        entity = {'film_id': film_id, 'filmVector': film_vector, 'posterVector': poster_vector}
        entities.append(entity)
    print(f'Entity count: {len(entities)}')
    print(f'First entity: {entities[0]}')
    client.upsert(collection_name='demo_v3', data=entities)
    print('Data inserted successfully!')

def complex_query():
    query_filmVector = [[0.5485517382621765, 0.27578410506248474, 0.43596649169921875, 0.038949452340602875, 0.09114820510149002]]
    search_param_1 = {
        'data': query_filmVector,
        'anns_field': 'filmVector',
        'param': {
            'metric_type': 'L2',
            'params': {'nprobe': 10}
        },
        'limit': 2,
    }
    request1 = AnnSearchRequest(**search_param_1)
    query_posterVector = [[0.13992096483707428,0.6470130085945129,0.9526867270469666,0.407039999961853,0.13641506433486938]]
    search_param_2 = {
        'data': query_posterVector,
        'anns_field': 'posterVector',
        'param': { 'metric_type': 'COSINE'},
        'limit': 2,
    }
    request2 = AnnSearchRequest(**search_param_2)
    seqs = [request1, request2]
    ranker = WeightedRanker(0.7, 0.3)
    outputs = client.hybrid_search(
        collection_name='demo_v3',
        reqs=seqs,
        ranker=ranker,
        limit=2,
        output_fields=['filmVector', 'posterVector']
    )
    print(f'Hybrid search results: {outputs}')
    print('♥️' * 20)
    for hits in outputs:
        print('TopK results:')
        for hit in hits:
            print(hit)

if __name__ == '__main__':
    operate_entity()
