import requests,joblib,csv

file = 'nq-test.qa.csv'
data_to_save = 'nq-test-with-colbert-ref-top50'
data_list = []
with open(file, encoding='UTF-8') as fin:
    reader = csv.reader(fin, delimiter='\t')
    for k, row in enumerate(reader):
        print(k)
        temp = {}
        q = row[0]
        answer_list = eval(row[1])
        query_item = q
        url = 'http://localhost:8893/api/search?query=' + query_item + '&k=50'
        response = requests.get(url=url)
        res_dic = response.json()
        corpus_list_topk = res_dic['topk']
        temp['question'] = q
        temp['answer'] = answer_list
        temp['list'] = corpus_list_topk
        if k == 0:
            print(corpus_list_topk)
        data_list.append(temp)
joblib.dump(data_list,data_to_save)
