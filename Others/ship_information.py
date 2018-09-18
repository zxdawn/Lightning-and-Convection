# Scrapying data from ccs(http://psmis.ccs.org.cn/ship/list.do)

# Download driver and mv to your path
#   Chrome driver:
#       https://sites.google.com/a/chromium.org/chromedriver/downloads
#   FireFox driver:
#       https://github.com/mozilla/geckodriver/releases

import re
import time
import requests
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

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

    headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}
    for link in links:
        print (link)
        data1 = urllib.request.Request(link, headers=headers)
        f = urllib.request.urlopen(data1)
        data = f.read()

        soup = BeautifulSoup(data, 'lxml')
        power.extend(soup.find('th', text=re.compile('Cylinders,Diameter,Stroke,Power &Revolution of Main Engine:')).find_next_sibling("td"))
        f.close()
        time.sleep(1)

    #example:'\r\n    \t**** \t\r\n    '
    power = [item[7:-8] for item in power]


# Click to next page and scrapy again
def next_page(browser,all_links,power):
    continue_link = browser.find_element_by_link_text('下一页')
    continue_link.click()
    time.sleep(5)
    html = browser.page_source
    get_links(html,all_links,power)


def main(browser):
    all_links = []; power = []

    html = browser.page_source
    get_links(html,all_links,power)

    result = 'continue'
    while result is 'continue':
        try:
            next_page(browser,all_links,power)
        except NoSuchElementException:
            result = 'stop'
            print ('At the end of pages now, please check your results')

    browser.quit()
    print (len(all_links))


if __name__ == '__main__':
    main(browser)
