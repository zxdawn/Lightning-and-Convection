# Scrapying data from ccs(http://psmis.ccs.org.cn/ship/list.do)

# Download driver and mv to your path
#   Chrome driver:
#       https://sites.google.com/a/chromium.org/chromedriver/downloads
#   FireFox driver:
#       https://github.com/mozilla/geckodriver/releases

import re
import time
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

# Set names of save_files
data_name = 'ship_data.csv'
urls_name = 'urls.csv'

# Set browser options
opts = Options()
opts.set_headless()
assert opts.headless
browser = Chrome(options=opts)

# Main url
domain = 'http://psmis.ccs.org.cn'
url = domain+'/ship/list.do'
browser.get(url)

# Complete url
def complete_link(result):
    result = result.rpartition('..')[2]\
                .replace('cn','en')\
                .replace('Provisionally Classed','Provisionally%20Classed')\
                .replace('Class Suspended','Class%20Suspended')

    return result

# Get link of each ship
def get_links(url,all_links,power):
    soup = BeautifulSoup(url, 'lxml')
    results = soup.findAll('table')[0].tbody.find_all('tr')
    links = [domain+complete_link(result.a['href']) for result in results]
    all_links.extend(links)


# Click to next page and scrapy again
def next_page(browser,all_links,power):
    continue_link = browser.find_element_by_link_text('下一页')
    continue_link.click()
    time.sleep(0.2)
    html = browser.page_source
    get_links(html,all_links,power)


# Scrapy everything and save to ship_data list
def get_data(urls_name,data_name):
    with open(urls_name,'r') as urls:
        # Read urls as list
        urls_list=[url for url in urls]
        # Set headers and empty ship_data
        headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}
        ship_data = []        

        for counter,url in enumerate(urls_list):
            print ('Downloading '+counter)
            while True:
                try:
                    # Read data from url
                    data1 = urllib.request.Request(url, headers=headers)
                    f = urllib.request.urlopen(data1)
                    data = f.read()
                    soup = BeautifulSoup(data, 'lxml')

                    if counter == 0:
                        # Read column_headers
                        column_headers = [re.sub('\s+','',th.getText()).rstrip().replace(':','')\
                            for th in soup.findAll('th')]
                        # Read first ship_data
                        ship_data.append([re.sub('\s+','',td.getText()).rstrip()\
                            for td in soup.findAll('td')])

                    else:
                        # Read residual ship_data
                        ship_data.append([re.sub('\s+','',td.getText()).rstrip()\
                            for td in soup.findAll('td')])

                    if counter == len(urls_list)-1:
                        # Save data to csv file
                        df = pd.DataFrame(ship_data, columns=column_headers)
                        df = df.convert_objects(convert_numeric=True)
                        df.to_csv(data_name,index=False)
                except:
                    # print("Connection of url is refused by the server..")
                    print("Let me sleep for a while ZZzzzz...")
                    time.sleep(10)
                    continue
                break


def get_urls(browser,urls_name):
    all_links = []; power = []
    # Get links of all ships
    html = browser.page_source
    get_links(html,all_links,power)

    # Check if it's the last page
    result = 'continue'
    while result is 'continue':
        try:
            next_page(browser,all_links,power)
        except NoSuchElementException:
            result = 'stop'
            print ('At the end of pages now, please check your results')

    browser.quit()

    # Save urls as csv file "urls_name"
    with open(urls_name, 'w') as myfile:
        for link in all_links:
            myfile.write(link)
            myfile.write('\n')


def main(browser):
    get_urls(browser,urls_name)
    get_data(urls_name,data_name)


if __name__ == '__main__':
    main(browser)