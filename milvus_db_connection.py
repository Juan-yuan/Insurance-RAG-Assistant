"""
案例:
    演示通过 pymilvus库与Milvus数据库交互.
"""
import random

# 导包
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker, RRFRanker


# todo 1. 定义数据库操作函数, 处理连接和数据库管理.
def operate_db():
    # 1. 配置数据库连接.
    # 写法1: 本地文件数据库连接方式.
    # client = MilvusClient(uri='ai_milvus.db')

    # 写法2: 实际使用的连接方式: 连接本地Milvus服务(默认端口: 19530), 实际开发中, 换成Linux服务器的IP即可.
    # client = MilvusClient(uri='http://127.0.0.1:19530')
    client = MilvusClient(uri='http://localhost:19530')

    # 2. 处理'milvus_demo'数据库
    # 2.1 获取当前所有数据库列表 -> 类似于: MySQL的 show databases;
    databases = client.list_databases()
    # print(f'当前数据库列表: {databases}')      # ['default', 'itcast']

    # 2.2 判断要创建的数据库'milvus_demo'是否存在, 存在就使用, 不存在就创建.
    if 'milvus_demo' not in databases:
        # 不存在, 就创建数据库
        client.create_database(db_name='milvus_demo')
        print('创建数据库: milvus_demo 成功')
    else:
        # 存在, 就使用数据库
        client.using_database(db_name='milvus_demo')
        print('使用数据库: milvus_demo 成功')

    # 3. 返回客户端实例.
    return client


# todo 2. 调用上述函数, 获取数据库客户端实例.
client = operate_db()


# todo 3. collection集合(相当于Milvus的表) -> 即: Milvus表的操作.
def operate_table():
    # 1. 定义集合的Schema(相当于: 表结构), schema用于规定集合中包含哪些字段, 各字段的数据类型和约束条件.
    # 参1: auto_id=False: 主键字段需要手动赋值, 不自动生成.
    # 参2: enable_dynamic_field=True: 允许集合中添加动态字段, 即: 不需要提前定义字段. 这些动态添加的字段会以 json格式存在 $meta中.
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # 2. 像schema中添加具体字段(相当于表中的列)
    # 参1: 字段名称.     参2: 字段数据类型.     参3: 是否是主键字段.
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    # 参1: 字段名称.     参2: 字段数据类型.     参3: 字段维度(词向量维度).
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=5)
    # 参1: 字段名称.     参2: 字段数据类型.     参3: 字段最大长度.     参4: 字段描述.
    schema.add_field(field_name='scalar', datatype=DataType.VARCHAR, max_length=256, description='标量字段')

    # 3. 创建集合(即: Milvus中的表) -> 使用上边定义的schema创建名为'demo_v1'的集合.
    # 参1: 集合名称.     参2: schema(表结构)
    client.create_collection(collection_name='demo_v1', schema=schema)

    # 4. 打印提示信息, 确保完成创建.
    print('创建集合(Milvus表): demo_v1 成功')


# todo 4. Milvus的索引操作.
def operate_index():
    # --------------------------------- 场景1: 为向量字段添加索引 ---------------------------------
    # 1. 初始化索引参数配置对象, 用于设置索引相关参数.       注意: 后续的向量索引, 标量索引都是通过它来进行设置的.
    prepare_indexs = client.prepare_index_params()

    # 2. 配置向量字段的索引信息.
    # 参1: field_name='vector'   指定为vector字段创建索引.
    # 参2: index_type=''         索引类型,这里用的是默认的, 实际开发中需要指定, 例如: IVF_FLAT等...
    # 参3: metric_type='COSINE'  向量距离度量方式, 例如: L2(欧式距离), COSINE(余弦相似度), IP(内积)等...
    # 参4: index_name='vector_index'  索引名称.
    prepare_indexs.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')

    # 3. 执行索引创建操作.
    # 参1: collection_name='demo_v1'       指定操作的集合(Milvus表)为: demo_v1
    # 参2: index_params=prepare_indexs     指定索引参数配置对象.
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)

    # --------------------------------- 场景2: 查看索引信息 ---------------------------------
    # 1. 列出集合中所有索引
    res = client.list_indexes(collection_name='demo_v1')
    # 2. 打印索引信息
    print(f'res: {res}')        # res: ['vector_index']

    # 3. 获取指定索引的详细信息
    res1 = client.describe_index(collection_name='demo_v1', index_name='vector_index')
    print(f'res1: {res1}')
    print('-' * 30)

    # --------------------------------- 场景3: 集合的加载和释放 ---------------------------------
    # 1. 查看集合当前加载状态(默认未加载, 状态为: <LoadState: NotLoad>)
    print(f'集合加载状态: {client.get_load_state(collection_name="demo_v1")}')

    # 2. 集合加载到内存.
    client.load_collection(collection_name='demo_v1')
    print(f'集合加载状态: {client.get_load_state(collection_name="demo_v1")}') # Loaded

    # 3. 集合释放(从内存中卸载, 删除索引前必须执行)
    client.release_collection(collection_name='demo_v1')
    print(f'集合加载状态: {client.get_load_state(collection_name="demo_v1")}') # NotLoad

    # --------------------------------- 场景4: 删除索引 ---------------------------------
    # 参1: collection_name='demo_v1'      目标集合名称
    # 参2: index_name='vector_index'      目标索引名称
    client.drop_index(collection_name='demo_v1', index_name='vector_index')

    # --------------------------------- 场景5: 为标量字段添加索引 ---------------------------------
    prepare_indexs.add_index(field_name='scalar', index_type='', index_name='scalar_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)

    # 列出所有索引, 验证标量索引是否创建成功
    res = client.list_indexes(collection_name='demo_v1')
    print(f'res: {res}')        # res: ['vector_index', 'scalar_index']


# todo 5.表数据(实体对象Entity)的增删改 -> 即: 往Milvus的Collection集合中 增删改数据.
def operate_entity():
    # ------------------------------ 场景1: 创建Collection集合(Milvus表) ------------------------------
    # 1. 直接创建集合, 无需提前定义Schema(表结构) -> 语法糖,简化方式.
    # 参1: collection_name='demo_v2'   集合(Milvus表)的名称
    # 参2: dimension=5                 向量维度
    # 参3: metric_type='IP'            向量距离度量方式, 例如: L2(欧式距离), COSINE(余弦相似度), IP(内积)等...
    if not client.has_collection(collection_name="demo_v2"):
        client.create_collection(collection_name='demo_v2', dimension=5, metric_type='IP')


    # ------------------------------ 场景2: 往Collection集合中添加数据(upsert = update + insert) ------------------------------
    # 1. 准备要添加的数据(实体列表)
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

    # 2. 执行upsert操作 -> 功能: 存在相同主键(id)则更新, 不存在则插入, 实现: 去重 + 新增.
    # 参1: collection_name='demo_v2'       目标集合名称
    # 参2: data=data                       要添加的数据
    client.upsert(collection_name='demo_v2', data=data)

    # ------------------------------ 场景3: 在Collection集合中创建分区 ------------------------------
    # 参1: collection_name='demo_v2'     目标集合名称
    # 参2: partition_tag='partitionA'    分区标签(名)
    client.create_partition(collection_name='demo_v2', partition_name='partitionA')

    # ------------------------------ 场景4: 往指定分区中添加数据 ------------------------------
    # 1. 准备要往分区中添加的数据
    data = [
        {"id": 10, "vector": [-0.5570353903748935, -0.8997887893201304, -0.7123782431855732, -0.6298990746450119,
                              0.6699215060604258], "color": "red_1202"},
        {"id": 11, "vector": [0.6319019033373907, 0.6821488267878275, 0.8552303045704168, 0.36929791364943054,
                              -0.14152860714878068], "color": "blue_4150"},
        {"id": 12, "vector": [0.9483947484855766, -0.32294203351925344, 0.9759290319978025, 0.8262982148666174,
                              -0.8351194181285713], "color": "orange_4590"},
        {"id": 13, "vector": [-0.5449109892498731, 0.043511240563786524, -0.25105249484790804, -0.012030655265886425,
                              -0.0010987671273892108], "color": "pink_9619"},
        {"id": 14, "vector": [0.6603339372951424, -0.10866551787442225, -0.9435597754324891, 0.8230244263466688,
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

    # 2. 往分区中添加数据 -> upsert
    # 参1: 目标集合名称,   参2: 要添加的数据,   参3: 目标分区名称
    client.upsert(collection_name='demo_v2', data=data, partition_name='partitionA')

    # 3. 额外动作, 把上述数据也添加到默认分区中
    client.upsert(collection_name='demo_v2', data=data)

    # ------------------------------ 场景5: 从集合中删除数据 ------------------------------
    # 1. 按过滤条件删除数据 -> 不指定分区时, 默认遍历整个集合.
    # res = client.delete(collection_name='demo_v2', filter='id in [12, 5, 6]')
    # print(f'res: {res}')        # 打印删除结果(删除的行数)  仅供参考.


    # 2. 按主键 + 指定分区删除 -> 仅删除指定分区中的数据
    # 参1: 集合名称,   参2: (要删除的)主键列表,   参3: 目标分区名称
    # res2 = client.delete(collection_name='demo_v2', ids=[13, 5], partition_name='partitionA')
    # print(f'res2: {res2}')      # 打印删除结果(删除的行数)  仅供参考.


# todo 6. 表数据(Milvus集合)操作函数 -> 实体对象 查询
def operate_query():
    # ---------------------------------- 场景1: 单一向量查询(根据单个向量找相似结果) ----------------------------------
    # 1. 调用search()方法执行单一向量搜索.
    result1 = client.search(
        collection_name='demo_v2',                  # 目标搜索的集合名称
        data=[[0.19886812567710876,0.060235604643821716,0.697696328163147,0.2614474594593048,0.8387295007705688]],      # 查询向量
        limit=2,                                    # 最多返回2条相似结果
        search_params={'metric_type': 'IP'},        # 相似度度量方式 -> 内积(IP)
        output_fields=['id', 'vector', 'color']     # 指定返回结果中包含的字段, 类似于SQL语句: select id, vector, color from demo_v2
    )
    # 2. 打印查询结果
    print(f'result1: {result1}')
    print('♣' * 30)

    # ---------------------------------- 场景2: 批量向量查询(根据多个向量 同时 找相似结果) ----------------------------------
    # 1. 调用search()方法执行批量向量搜索.
    result2 = client.search(
        collection_name='demo_v2',  # 目标搜索的集合名称
        data=[
                [0.19886812567710876, 0.060235604643821716, 0.697696328163147, 0.2614474594593048, 0.8387295007705688],
                [0.31720051169395447, 0.971904456615448, -0.369811475276947, -0.48608946800231934, 0.9579188823699951]
            ],           # 查询向量
        limit=2,  # 最多返回2条相似结果
        search_params={'metric_type': 'IP'},  # 相似度度量方式 -> 内积(IP)
        output_fields=['id', 'vector']        # 指定返回结果中包含的字段, 类似于SQL语句: select id, vector from demo_v2
    )
    # 2. 打印查询结果
    print(f'result2: {result2}')        # 返回4条:  2条(搜索向量) * 2条(每条搜索向量对应的数据 limit=2)
    print('♣' * 30)

    # ---------------------------------- 场景3: 分区搜索(仅在指定的分区中搜索) ----------------------------------
    # 1. 调用search()方法执行单一向量搜索.
    result3 = client.search(
        collection_name='demo_v2',  # 目标搜索的集合名称
        data=[[0.6319019198417664,0.682148814201355,0.8552303314208984,0.36929792165756226,-0.14152860641479492]],
        # 查询向量
        limit=5,  # 最多返回5条相似结果
        search_params={'metric_type': 'IP'},  # 相似度度量方式 -> 内积(IP)
        partition_names=['partitionA']        # 指定分区名称
    )
    # 2. 打印查询结果
    print(f'result3: {result3}')
    print('♣' * 30)

    # ---------------------------------- 场景4: 指定输出字段搜索(仅返回需要的字段, 减少数据量) ----------------------------------
    # 1. 调用search()方法执行单一向量搜索.
    result4 = client.search(
        collection_name='demo_v2',  # 目标搜索的集合名称
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],
        # 查询向量
        limit=5,  # 最多返回5条相似结果
        search_params={'metric_type': 'IP'},    # 相似度度量方式 -> 内积(IP)
        output_fields=['vector', 'color']       # 指定返回结果中包含的字段, 类似于SQL语句: select vector, color from demo_v2
    )
    # 2. 打印查询结果
    print(f'result4: {result4}')
    print('♣' * 30)

    # ---------------------------------- 场景5: 带过滤条件的搜索(先过滤数据再找相似结果) ----------------------------------
    # 1. 调用search()方法执行单一向量搜索.
    result5 = client.search(
        collection_name='demo_v2',  # 目标搜索的集合名称
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],  # 查询向量
        limit=5,  # 最多返回5条相似结果
        search_params={'metric_type': 'IP'},  # 相似度度量方式 -> 内积(IP)
        output_fields=['color'],              # 指定返回结果中包含的字段, 类似于SQL语句: select vector, color from demo_v2
        filter='color like "red%"'
    )
    # 2. 打印查询结果
    print(f'result5: {result5}')
    print('♣' * 30)

    # ---------------------------------- 场景6: 范围搜索(仅返回相似度在指定范围内的结果) ----------------------------------
    # 1. 定义范围搜索参数.
    search_params = {
        'metric_type': 'IP',        # 相似度度量方式 -> 内积(IP)
        'params': {
            "radius": 0.8,          # 搜索范围下限: 相似度 >= 0.8
            "range_filter": 1       # 搜索范围上限: 相似度 <= 1
        }                           # 它们共同组成范围约束[0.8, 1], 仅返回相似度在该区间的结果.
    }

    # 2. 调用search()方法执行单一向量搜索.
    result6 = client.search(
        collection_name='demo_v2',  # 目标搜索的集合名称
        data=[[0.6319019198417664, 0.682148814201355, 0.8552303314208984, 0.36929792165756226, -0.14152860641479492]],  # 查询向量
        limit=5,  # 最多返回5条相似结果
        search_params=search_params,          # 查询参数
        output_fields=['color'],              # 指定返回结果中包含的字段, 类似于SQL语句: select color from demo_v2
    )
    # 2. 打印查询结果
    print(f'result6: {result6}')


# todo 7.复杂查询(多向量字段混合检索示例) -> 准备测试数据.
def complex_query_add_data():
    # ------------------------------------------ 场景1: 定义集合Schema ------------------------------------------
    # 1. 创建Schema对象
    schema = client.create_schema(enable_dynamic_field=False)       # 不允许动态字段.

    # 2. 往schema中添加字段
    # 2.1 主键film_id(电影id), int64
    schema.add_field(field_name='film_id', datatype=DataType.INT64, is_primary=True)
    # 2.2 添加第1个向量字段filmVector(电影的特征向量), 浮点型, 向量维度为5
    schema.add_field(field_name='filmVector', datatype=DataType.FLOAT_VECTOR, dim=5)
    # 2.3 添加第2个向量字段posterVector(电影海报特征向量), 浮点型, 向量维度为5
    schema.add_field(field_name='posterVector', datatype=DataType.FLOAT_VECTOR, dim=5)


    # ------------------------------------------ 场景2: 定义多个向量字段的索引 ------------------------------------------
    # 1. 初始化索引参数配置对象.
    index_params = client.prepare_index_params()
    # 2. 给filmVector字段添加索引.
    # 参1: 给哪个字段添加索引,  参2: 索引类型,  参3: 索引度量方式(L2: 欧式距离),  参4: 索引参数配置对象(IVF索引参数, 聚类数量为128)
    index_params.add_index(field_name='filmVector', index_type='IVF_FLAT', metric_type='L2', params={'nlist': 128})
    # 3. 给posterVector字段添加索引.
    # 参1: 给哪个字段添加索引,  参2: 索引类型,  参3: 索引度量方式(COSINE: 余弦相似度)
    index_params.add_index(field_name='posterVector', index_type='', metric_type='COSINE')


    # ------------------------------------------ 场景3: 创建 包含多向量字段的集合 ------------------------------------------
    # 参1: 集合名称,   参2: 集合Schema(表结构),   参3: 索引参数配置对象
    client.create_collection(collection_name='demo_v3', schema=schema, index_params=index_params)


    # ------------------------------------------ 场景4: 生成并插入实体数据(批量添加多条记录)  ------------------------------------------
    # 1. 初始化实体列表 -> 用于存储多条数据.
    entities = []

    # 2. 循环生成1000条实体数据.
    for _ in range(1000):
        # 2.1 生成每条实体的字段值.
        film_id = random.randint(1, 10000)              # 随机生成电影id
        film_vector = [random.random() for _ in range(5)]     # 随机生成5维的电影特征向量
        poster_vector = [random.random() for _ in range(5)]   # 随机生成5维的电影海报特征向量

        # 2.2 构建实体字典(字段要和Schema一致)
        entity = {'film_id': film_id, 'filmVector': film_vector, 'posterVector': poster_vector}
        # 2.3 将实体字典添加到实体列表中.
        entities.append(entity)

    # 3. 打印实体数量 和 第1条实体, 验证数据生成结果.
    print(f'实体的数量: {len(entities)}')        # 1000
    print(f'第1条实体: {entities[0]}')

    # 4. 执行upsert操作 批量插入/更新数据.
    # 参1: 集合名称,   参2: 实体列表(要插入的数据)
    client.upsert(collection_name='demo_v3', data=entities)

    # 5. 打印结果.
    print('数据插入成功!')


# todo 8.定义函数, 实现: 混合检索(复杂查询)
def complex_query():
    # ---------------------------------- 场景1: 多向量字段混合检索(合并多个搜索结果) ----------------------------------
    # 1. 配置第1个向量字段filmVector的搜索参数.
    # 1.1 定义查询向量
    query_filmVector = [[0.5485517382621765, 0.27578410506248474, 0.43596649169921875, 0.038949452340602875, 0.09114820510149002]]

    # 1.2 配置搜索参数.
    search_param_1 = {
        'data': query_filmVector,           # 查询向量
        'anns_field': 'filmVector',         # 执行搜索的向量字段为filmVector, anns 近似最近邻的缩写
        'param': {
            'metric_type': 'L2',            # 搜索距离度量方式 -> 欧式距离(L2)
            'params': {'nprobe': 10}        # IVF索引参数 -> 查询前10个最相似的醋
        },
        'limit': 2,                        # 最多返回2条结果
    }

    # 1.3 创建第1个搜索请求对象.
    request1 = AnnSearchRequest(**search_param_1)

    # 2. 配置第2个向量字段posterVector的搜索参数.
    # 2.1 定义查询向量
    query_posterVector = [[0.13992096483707428,0.6470130085945129,0.9526867270469666,0.407039999961853,0.13641506433486938]]
    # 2.2 配置搜索参数.
    search_param_2 = {
        'data': query_posterVector,             # 查询向量
        'anns_field': 'posterVector',           # 执行搜索的向量字段为filmVector, anns 近似最近邻的缩写
        'param': { 'metric_type': 'COSINE'},    # 向量相似度度量 -> 余弦相似度
        'limit': 2,                             # 最多返回2条结果
    }
    # 2.3 创建第2个搜索请求对象.
    request2 = AnnSearchRequest(**search_param_2)

    # 3. 配置混合检索(即: 合并两个字段的搜索结果)
    # 3.1 组合多个搜索请求.
    seqs = [request1, request2]     # 包含filmVector 和 posterVector 的搜索请求
    # 3.2 定义结果排序器(例如: 加权排序, filmVector的权重0.7, posterVector的权重0.3)
    ranker = WeightedRanker(0.7, 0.3)
    # ranker = RRFRanker()            # 可选, 另一种排序算法, 倒数排序融合

    # 4. 执行混合检索.
    outputs = client.hybrid_search(
        collection_name='demo_v3',          # 目标集合名称
        reqs=seqs,                          # 混合检索的搜索请求列表
        ranker=ranker,                      # 混合排序器
        limit=2,                            # 最多返回2条结果
        output_fields=['filmVector', 'posterVector']        # 返回结果中包含的字段.
    )
    # 5. 打印混合检索结果.
    print(f'混合检索结果: {outputs}')
    print('♥️' * 20)

    # ---------------------------------- 场景2: 遍历并打印检索结果详情 ----------------------------------
    # 1. 循环遍历每个查询结果
    for hits in outputs:
        print('TopK 结果:')
        # 2. 遍历每条结果, 打印详情.
        for hit in hits:
            print(hit)





# todo n.测试代码
if __name__ == '__main__':
    # 1. 测试: 数据库操作函数.
    # operate_db()

    # 2. 测试: 集合操作函数.
    # operate_table()

    # 3. 测试: 索引操作函数.
    # operate_index()

    # 4. 测试: 表数据(Milvus集合)操作函数 -> 实体(Entity)的 增删改
    operate_entity()

    # 5. 测试: 表数据(Milvus集合)操作函数 -> 实体(Entity)的 查询
    # operate_query()

    # 6. 测试: 表数据(Milvus集合)操作函数 -> 混合查询
    # 准备数据
    # complex_query_add_data()

    # 混合查询
    # complex_query()

    # 7. 测试: 删除指定的集合(Milvus表)
    # client.drop_collection(collection_name='demo_v1')