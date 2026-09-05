"""
Example:
    Demonstrates interaction with the Milvus database through pymilvus.
"""
import random

# Import packages
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker, RRFRanker


# todo 1. Define database operation function, handle connection and database management.
def operate_db():
    # 1. Configure database connection.
    # Option 1: Local file database connection.
    # client = MilvusClient(uri='ai_milvus.db')

    # Option 2: Connect to local Milvus service (default port: 19530).
    # In actual development, replace with the Linux server IP.
    # client = MilvusClient(uri='http://127.0.0.1:19530')
    client = MilvusClient(uri='http://localhost:19530')

    # 2. Handle the 'milvus_demo' database
    # 2.1 Get the current database list -> similar to: MySQL SHOW DATABASES;
    databases = client.list_databases()
    # print(f'Current database list: {databases}')      # ['default', 'itcast']

    # 2.2 Check whether 'milvus_demo' exists.
    if 'milvus_demo' not in databases:
        # Does not exist, so create the database.
        client.create_database(db_name='milvus_demo')
        print('Created database: milvus_demo successfully')
    else:
        # Exists, so use the database.
        client.using_database(db_name='milvus_demo')
        print('Using database: milvus_demo successfully')

    # 3. Return the client instance.
    return client


# todo 2. Call the function above and get the database client instance.
client = operate_db()


# todo 3. Collection operations -> equivalent to table operations in Milvus.
def operate_table():
    # 1. Define the Collection Schema, equivalent to a table structure.
    # The schema defines the fields, data types, and constraints in the Collection.
    # Arg 1: auto_id=False: primary key fields must be assigned manually.
    # Arg 2: enable_dynamic_field=True: allows dynamic fields to be added.
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # 2. Add specific fields to the schema, equivalent to table columns.
    # Arg 1: field name.     Arg 2: data type.     Arg 3: whether it is a primary key.
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    # Arg 1: field name.     Arg 2: data type.     Arg 3: vector dimension.
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=5)
    # Arg 1: field name.     Arg 2: data type.     Arg 3: maximum length.     Arg 4: field description.
    schema.add_field(field_name='scalar', datatype=DataType.VARCHAR, max_length=256, description='Scalar field')

    # 3. Create the Collection using the schema defined above.
    # Arg 1: collection name.     Arg 2: schema.
    client.create_collection(collection_name='demo_v1', schema=schema)

    # 4. Print a message to confirm creation.
    print('Created Collection: demo_v1 successfully')


# todo 4. Milvus index operations.
def operate_index():
    # --------------------------------- Scenario 1: Add an index to a vector field ---------------------------------
    # 1. Initialize the index parameter configuration object.
    # Note: vector and scalar indexes are configured through this object.
    prepare_indexs = client.prepare_index_params()

    # 2. Configure index information for the vector field.
    # Arg 1: field_name='vector'   Create an index for the vector field.
    # Arg 2: index_type=''         Index type. In real development, specify one such as IVF_FLAT.
    # Arg 3: metric_type='COSINE'  Vector distance metric, such as L2, COSINE, or IP.
    # Arg 4: index_name='vector_index'  Index name.
    prepare_indexs.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')

    # 3. Create the index.
    # Arg 1: collection_name='demo_v1'       Target Collection.
    # Arg 2: index_params=prepare_indexs     Index parameter configuration object.
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)

    # --------------------------------- Scenario 2: View index information ---------------------------------
    # 1. List all indexes in the Collection.
    res = client.list_indexes(collection_name='demo_v1')
    # 2. Print index information.
    print(f'res: {res}')        # res: ['vector_index']

    # 3. Get detailed information about the specified index.
    res1 = client.describe_index(collection_name='demo_v1', index_name='vector_index')
    print(f'res1: {res1}')
    print('-' * 30)

    # --------------------------------- Scenario 3: Load and release Collection ---------------------------------
    # 1. Check the current Collection load state.
    print(f'Collection load state: {client.get_load_state(collection_name="demo_v1")}')

    # 2. Load the Collection into memory.
    client.load_collection(collection_name='demo_v1')
    print(f'Collection load state: {client.get_load_state(collection_name="demo_v1")}') # Loaded

    # 3. Release the Collection from memory.
    client.release_collection(collection_name='demo_v1')
    print(f'Collection load state: {client.get_load_state(collection_name="demo_v1")}') # NotLoad

    # --------------------------------- Scenario 4: Delete an index ---------------------------------
    # Arg 1: collection_name='demo_v1'      Target Collection.
    # Arg 2: index_name='vector_index'      Target index name.
    client.drop_index(collection_name='demo_v1', index_name='vector_index')

    # --------------------------------- Scenario 5: Add an index to a scalar field ---------------------------------
    prepare_indexs.add_index(field_name='scalar', index_type='', index_name='scalar_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)

    # List all indexes to verify that the scalar index was created successfully.
    res = client.list_indexes(collection_name='demo_v1')
    print(f'res: {res}')        # res: ['vector_index', 'scalar_index']


# todo 5. Entity data operations -> insert, update, and delete data in a Milvus Collection.
def operate_entity():
    # ------------------------------ Scenario 1: Create Collection ------------------------------
    # 1. Create a Collection directly without defining a Schema first.
    # Arg 1: collection_name='demo_v2'   Collection name.
    # Arg 2: dimension=5                 Vector dimension.
    # Arg 3: metric_type='IP'            Vector distance metric, such as L2, COSINE, or IP.
    if not client.has_collection(collection_name="demo_v2"):
        client.create_collection(collection_name='demo_v2', dimension=5, metric_type='IP')


    # ------------------------------ Scenario 2: Add data to Collection (upsert = update + insert) ------------------------------
    # 1. Prepare the data to be added.
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

    # 2. Execute the upsert operation -> update if the same primary key exists, otherwise insert.
    client.upsert(collection_name='demo_v2', data=data)

    # ------------------------------ Scenario 3: Create a partition in the Collection ------------------------------
    # Arg 1: collection_name='demo_v2'     Target Collection.
    # Arg 2: partition_tag='partitionA'    Partition name.
    client.create_partition(collection_name='demo_v2', partition_name='partitionA')

    # ------------------------------ Scenario 4: Add data to a specific partition ------------------------------
    # 1. Prepare data to be added to the partition.
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

    # 2. Add data to the partition -> upsert.
    # Arg 1: target Collection,   Arg 2: data to add,   Arg 3: target partition.
    client.upsert(collection_name='demo_v2', data=data, partition_name='partitionA')

    # 3. Additionally, add the same data to the default partition.
    client.upsert(collection_name='demo_v2', data=data)

    # ------------------------------ Scenario 5: Delete data from Collection ------------------------------
    # 1. Delete data by filter condition -> without a partition, the entire Collection is searched.
    # res = client.delete(collection_name='demo_v2', filter='id in [12, 5, 6]')
    # print(f'res: {res}')        # Print deletion result.

    # 2. Delete by primary key + specified partition -> only delete from the specified partition.
    # Arg 1: Collection name,   Arg 2: primary key list,   Arg 3: target partition.
    # res2 = client.delete(collection_name='demo_v2', ids=[13, 5], partition_name='partitionA')
    # print(f'res2: {res2}')      # Print deletion result.


# todo 6. Collection query operations -> query Entity objects.
def operate_query():
    # ---------------------------------- Scenario 1: Single vector search ----------------------------------
    # 1. Call search() to perform a single vector search.
    result1 = client.search(
        collection_name='demo_v2',                  # Target Collection.
        data=[[0.19886812567710876,0.060235604643821716,0.697696328163147,0.2614474594593048,0.8387295007705688]],      # Query vector.
        limit=2,                                    # Return at most 2 similar results.
        search_params={'metric_type': 'IP'},        # Similarity metric -> inner product (IP).
        output_fields=['id', 'vector', 'color']     # Fields to return.
    )
    # 2. Print query results.
    print(f'result1: {result1}')
    print('♣' * 30)

    # ---------------------------------- Scenario 2: Batch vector search ----------------------------------
    # 1. Call search() to perform batch vector search.
    result2 = client.search(
        collection_name='demo_v2',  # Target Collection.
        data=[
                [0.19886812567710876, 0.060235604643821716, 0.697696328163147, 0.2614474594593048, 0.8387295007705688],
                [0.31720051169395447, 0.971904456615448, -0.369811475276947, -0.48608946800231934, 0.9579188823699951]
            ],           # Query vectors.
        limit=2,  # Return at most 2 similar results.
        search_params={'metric_type': 'IP'},  # Similarity metric -> inner product (IP).
        output_fields=['id', 'vector']        # Fields to return.
    )
    # 2. Print query results.
    print(f'result2: {result2}')        # Returns 4 results: 2 query vectors * 2 results per vector.
    print('♣' * 30)

    # ---------------------------------- Scenario 3: Partition search ----------------------------------
    # 1. Call search() to perform a single vector search.
    result3 = client.search(
        collection_name='demo_v2',  # Target Collection.
        data=[[0.6319019198417664,0.682148814201355,0.8552303314208984,0.36929792165756226,-0.14152860641479492]],
        # Query vector.
        limit=5,  # Return at most 5 similar results.
        search_params={'metric_type': 'IP'},  # Similarity metric -> inner product (IP).
        partition_names=['partitionA']        # Specify the partition to search.
    )
    # 2. Print query results.
    print(f'result3: {result3}')
    print('♣' * 30)

    # ---------------------------------- Scenario 4: Search with specified output fields ----------------------------------
    # 1. Call search() to perform a single vector search.
    result4 = client.search(
        collection_name='demo_v2',  # Target Collection.
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],
        # Query vector.
        limit=5,  # Return at most 5 similar results.
        search_params={'metric_type': 'IP'},    # Similarity metric -> inner product (IP).
        output_fields=['vector', 'color']       # Fields to return.
    )
    # 2. Print query results.
    print(f'result4: {result4}')
    print('♣' * 30)

    # ---------------------------------- Scenario 5: Search with filter condition ----------------------------------
    # 1. Call search() to perform a single vector search.
    result5 = client.search(
        collection_name='demo_v2',  # Target Collection.
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],  # Query vector.
        limit=5,  # Return at most 5 similar results.
        search_params={'metric_type': 'IP'},  # Similarity metric -> inner product (IP).
        output_fields=['color'],              # Fields to return.
        filter='color like "red%"'
    )
    # 2. Print query results.
    print(f'result5: {result5}')
    print('♣' * 30)

    # ---------------------------------- Scenario 6: Range search ----------------------------------
    # 1. Define range search parameters.
    search_params = {
        'metric_type': 'IP',        # Similarity metric -> inner product (IP).
        'params': {
            "radius": 0.8,          # Lower bound: similarity >= 0.8.
            "range_filter": 1       # Upper bound: similarity <= 1.
        }                           # Together they define the range [0.8, 1].
    }

    # 2. Call search() to perform a single vector search.
    result6 = client.search(
        collection_name='demo_v2',  # Target Collection.
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],  # Query vector.
        limit=5,  # Return at most 5 similar results.
        search_params=search_params,          # Search parameters.
        output_fields=['color'],              # Fields to return.
    )
    # 2. Print query results.
    print(f'result6: {result6}')


# todo 7. Complex query (multi-vector field hybrid search) -> prepare test data.
def complex_query_add_data():
    # ------------------------------------------ Scenario 1: Define Collection Schema ------------------------------------------
    # 1. Create Schema object.
    schema = client.create_schema(enable_dynamic_field=False)       # Dynamic fields are disabled.

    # 2. Add fields to the schema.
    # 2.1 Primary key film_id, int64.
    schema.add_field(field_name='film_id', datatype=DataType.INT64, is_primary=True)
    # 2.2 Add the first vector field filmVector, float vector with dimension 5.
    schema.add_field(field_name='filmVector', datatype=DataType.FLOAT_VECTOR, dim=5)
    # 2.3 Add the second vector field posterVector, float vector with dimension 5.
    schema.add_field(field_name='posterVector', datatype=DataType.FLOAT_VECTOR, dim=5)


    # ------------------------------------------ Scenario 2: Define indexes for multiple vector fields ------------------------------------------
    # 1. Initialize index parameter configuration object.
    index_params = client.prepare_index_params()
    # 2. Add an index to filmVector.
    # Arg 1: field,  Arg 2: index type,  Arg 3: metric type,  Arg 4: IVF index parameters.
    index_params.add_index(field_name='filmVector', index_type='IVF_FLAT', metric_type='L2', params={'nlist': 128})
    # 3. Add an index to posterVector.
    # Arg 1: field,  Arg 2: index type,  Arg 3: metric type.
    index_params.add_index(field_name='posterVector', index_type='', metric_type='COSINE')


    # ------------------------------------------ Scenario 3: Create a Collection with multiple vector fields ------------------------------------------
    # Arg 1: collection name,   Arg 2: Collection Schema,   Arg 3: index parameter configuration.
    client.create_collection(collection_name='demo_v3', schema=schema, index_params=index_params)


    # ------------------------------------------ Scenario 4: Generate and insert Entity data ------------------------------------------
    # 1. Initialize Entity list.
    entities = []

    # 2. Generate 1000 Entity records.
    for _ in range(1000):
        # 2.1 Generate field values for each Entity.
        film_id = random.randint(1, 10000)              # Randomly generate film ID.
        film_vector = [random.random() for _ in range(5)]     # Generate a random 5-dimensional film vector.
        poster_vector = [random.random() for _ in range(5)]   # Generate a random 5-dimensional poster vector.

        # 2.2 Build the Entity dictionary. Fields must match the Schema.
        entity = {'film_id': film_id, 'filmVector': film_vector, 'posterVector': poster_vector}
        # 2.3 Add the Entity dictionary to the Entity list.
        entities.append(entity)

    # 3. Print Entity count and the first Entity to verify the generated data.
    print(f'Entity count: {len(entities)}')
    print(f'First entity: {entities[0]}')

    # 4. Execute upsert to insert/update data in batches.
    # Arg 1: Collection name,   Arg 2: Entity list.
    client.upsert(collection_name='demo_v3', data=entities)

    # 5. Print result.
    print('Data inserted successfully!')


# todo 8. Define function to implement hybrid search.
def complex_query():
    # ---------------------------------- Scenario 1: Multi-vector field hybrid search ----------------------------------
    # 1. Configure search parameters for the filmVector field.
    # 1.1 Define query vector.
    query_filmVector = [[0.5485517382621765, 0.27578410506248474, 0.43596649169921875, 0.038949452340602875, 0.09114820510149002]]

    # 1.2 Configure search parameters.
    search_param_1 = {
        'data': query_filmVector,           # Query vector.
        'anns_field': 'filmVector',         # Vector field used for search.
        'param': {
            'metric_type': 'L2',            # Search distance metric -> Euclidean distance (L2).
            'params': {'nprobe': 10}        # IVF index parameter -> search the top 10 clusters.
        },
        'limit': 2,                        # Return at most 2 results.
    }

    # 1.3 Create the first search request object.
    request1 = AnnSearchRequest(**search_param_1)

    # 2. Configure search parameters for the posterVector field.
    # 2.1 Define query vector.
    query_posterVector = [[0.13992096483707428,0.6470130085945129,0.9526867270469666,0.407039999961853,0.13641506433486938]]
    # 2.2 Configure search parameters.
    search_param_2 = {
        'data': query_posterVector,             # Query vector.
        'anns_field': 'posterVector',           # Vector field used for search.
        'param': { 'metric_type': 'COSINE'},    # Vector similarity metric -> cosine similarity.
        'limit': 2,                             # Return at most 2 results.
    }
    # 2.3 Create the second search request object.
    request2 = AnnSearchRequest(**search_param_2)

    # 3. Configure hybrid search by combining the two search results.
    # 3.1 Combine multiple search requests.
    seqs = [request1, request2]     # Search requests for filmVector and posterVector.
    # 3.2 Define result ranker. filmVector weight 0.7, posterVector weight 0.3.
    ranker = WeightedRanker(0.7, 0.3)
    # ranker = RRFRanker()            # Optional: another ranking algorithm.

    # 4. Execute hybrid search.
    outputs = client.hybrid_search(
        collection_name='demo_v3',          # Target Collection.
        reqs=seqs,                          # List of hybrid search requests.
        ranker=ranker,                      # Hybrid search ranker.
        limit=2,                            # Return at most 2 results.
        output_fields=['filmVector', 'posterVector']        # Fields to return.
    )
    # 5. Print hybrid search results.
    print(f'Hybrid search results: {outputs}')
    print('♥️' * 20)

    # ---------------------------------- Scenario 2: Iterate through and print search result details ----------------------------------
    # 1. Iterate through each query result.
    for hits in outputs:
        print('TopK results:')
        # 2. Iterate through each result and print details.
        for hit in hits:
            print(hit)





# todo n. Test code
if __name__ == '__main__':
    # 1. Test: database operation function.
    # operate_db()

    # 2. Test: Collection operation function.
    # operate_table()

    # 3. Test: index operation function.
    # operate_index()

    # 4. Test: Entity data operations -> insert, update, delete.
    operate_entity()

    # 5. Test: Entity data operations -> query.
    # operate_query()

    # 6. Test: Entity data operations -> hybrid query.
    # Prepare data.
    # complex_query_add_data()

    # Hybrid query.
    # complex_query()

    # 7. Test: Delete the specified Collection.
    # client.drop_collection(collection_name='demo_v1')
