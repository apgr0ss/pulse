import requests
import pandas as pd


def bing_search(query):
    url = 'https://api.bing.microsoft.com/v7.0/search'
    # query string parameters
    payload = {'q': query, 'count': 20}
    # custom headers
    headers = {'Ocp-Apim-Subscription-Key': ''}
    # make GET request
    r = requests.get(url, params=payload, headers=headers)
    # get JSON response
    return r.json()

def search(query, domain_info):
    if query is None:
        return None
    query_results = {'unique_id':[], 'domain_name':[], 'link':[]}
    unique_id = 0
    invalid_urls = []
    for row in domain_info.iterrows():
        row_id, row_data = row
        full_query = 'site:{0} ({1})'.format(row_data.url, query)
        results = bing_search(full_query)
        if 'webPages' not in results.keys():
            invalid_urls += [row_data.url]
        else:
            for result in results['webPages']['value']:
                query_results['unique_id'] += [unique_id]
                query_results['domain_name'] += [row_data.domain_name]
                query_results['link'] += [result['url']]
                unique_id += 1
    pd.DataFrame(query_results).to_csv('data\\search_results.csv', index=False)
    return invalid_urls
