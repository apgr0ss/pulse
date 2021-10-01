from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3 import Retry
import time
import io
from PyPDF2 import PdfFileReader
import numpy as np
import pandas as pd
import bs4
from fake_headers import Headers


def runGet(url, s):
    try:
        response = s.get(url, timeout=10)
        return response
    except Exception as e:
        return e

def scrape_web(row):
    sites = ''
    skipped_urls = []
    url = row.link
    retries = Retry(total=5,
                    backoff_factor=0.2)
    header = Headers()
    s = Session()
    s.headers.update(header.generate())
    s.mount('http://', HTTPAdapter(retries))
    i = 0
    maxTries = 15
    while i < maxTries:
        response = runGet(url, s)
        if not issubclass(type(response), Exception) and not issubclass(type(response), RequestException) and response.status_code == 200:
            break
        else:
            s.headers.update(header.generate())
            i+=1
            if i == maxTries:
                if issubclass(type(response), Exception) or issubclass(type(response), RequestException):
                    err = response.__class__.__name__
                    print('ERROR associated with {0}: {1}'.format(url, err))
                else:
                    print('Exited with status {0} for url {1}'.format(response.status_code, url))
                rtime = np.random.random()
                time.sleep(rtime)
                return (row.domain_name, '', url)
    if response.headers['content-type'] == 'application/pdf':
        with io.BytesIO(response.content) as f:
            pdf = PdfFileReader(f)
            if pdf.isEncrypted:
                try:
                    pdf.decrypt('')
                except NotImplementedError:
                    print('Could not decrypt pdf from url {0}'.format(url))
                    rtime = np.random.random()
                    time.sleep(rtime)
                    return (row.domain_name, '', url)
            n_pages = pdf.numPages
            page = ''
            for i in range(n_pages):
                page += pdf.getPage(i).extractText()
            to_add = '\n' + page
            sites += to_add
        print('scraped ' + url)
    else:
        sites += '\n' + response.text
        print('scraped ' + url)

    rtime = np.random.random()
    time.sleep(rtime)
    return (row.domain_name, sites, url)

def preprocess_scraped_text(textcol):
    texts = []
    for text in textcol:
        paragraphs = bs4.BeautifulSoup(text).find_all('p')
        paragraphs = [result.get_text().encode('ascii', 'replace').decode('utf-8').replace('\n',' ').replace('?', ' ').replace(',', ' ') for result in paragraphs]
        paragraphs = [paragraph for paragraph in paragraphs if len([token for token in paragraph.split(' ') if len(token) > 0]) > 40]
        texts += [' '.join(paragraphs)]
    return texts

def scrape_web_main(path):
    #urls = pd.read_csv(path)
    #scraped_data = urls.apply(scrape_web, axis=1).values
    #text = pd.DataFrame.from_records(scraped_data, columns = ['domain_name', 'text', 'url'])
    text = pd.read_csv('data\\text_raw.csv')
    text.text = text.text.fillna('')
    text.text = preprocess_scraped_text(text.text)
    text.to_csv('data\\text_preprocessed.csv', index=False)
