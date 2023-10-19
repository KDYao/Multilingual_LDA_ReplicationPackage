"""
Download data from BleepingComputer website
Author: Muhammad Parvez
"""

import requests
import time
import datetime
from bs4 import BeautifulSoup
import os
import cloudscraper
import random

from src.dataprocessing.Post import Post
global save_dir

microsoft_windows_support = ['https://www.bleepingcomputer.com/forums/f/233/windows-crashes-and-blue-screen-of-death-bsod-help-and-support/',
                             'https://www.bleepingcomputer.com/forums/f/229/windows-10-support/',
                             'https://www.bleepingcomputer.com/forums/f/209/windows-8-and-windows-81/',
                             'https://www.bleepingcomputer.com/forums/f/167/windows-7/',
                             'https://www.bleepingcomputer.com/forums/f/83/windows-server/',
                             'https://www.bleepingcomputer.com/forums/f/248/legacy-windows-operating-systems/']

alternative_operating_systems_support = ['https://www.bleepingcomputer.com/forums/f/11/linux-unix/',
                                         'https://www.bleepingcomputer.com/forums/f/172/mac-os/',
                                         'https://www.bleepingcomputer.com/forums/f/10/dospdaother/']

hardware = ['https://www.bleepingcomputer.com/forums/f/7/internal-hardware/',
            'https://www.bleepingcomputer.com/forums/f/138/external-hardware/',
            'https://www.bleepingcomputer.com/forums/f/139/system-building-and-upgrading/',
            'https://www.bleepingcomputer.com/forums/f/171/questions-and-advice-for-buying-a-new-computer/']

security = ['https://www.bleepingcomputer.com/forums/f/22/virus-trojan-spyware-and-malware-removal-help/',
            'https://www.bleepingcomputer.com/forums/f/239/ransomware-help-tech-support/',
            'https://www.bleepingcomputer.com/forums/f/25/anti-virus-anti-malware-and-privacy-software/',
            'https://www.bleepingcomputer.com/forums/f/238/backup-imaging-and-disk-management-software/',
            'https://www.bleepingcomputer.com/forums/f/222/firewall-software-and-hardware/',
            'https://www.bleepingcomputer.com/forums/f/231/encryption-methods-and-programs/',
            'https://www.bleepingcomputer.com/forums/f/45/general-security/',
            'https://www.bleepingcomputer.com/forums/f/103/am-i-infected-what-do-i-do/']

internet_and_networking = ['https://www.bleepingcomputer.com/forums/f/14/web-browsingemail-and-other-internet-applications/',
                           'https://www.bleepingcomputer.com/forums/f/21/networking/',
                           'https://www.bleepingcomputer.com/forums/f/38/web-site-development/',
                           'https://www.bleepingcomputer.com/forums/f/218/service-providers/',
                           'https://www.bleepingcomputer.com/forums/f/230/network-streaming-devices/']

software = ['https://www.bleepingcomputer.com/forums/f/16/business-applications/',
            'https://www.bleepingcomputer.com/forums/f/57/all-other-applications/',
            'https://www.bleepingcomputer.com/forums/f/27/tips-and-tricks/',
            'https://www.bleepingcomputer.com/forums/f/37/graphics-design-and-photo-editing/',
            'https://www.bleepingcomputer.com/forums/f/65/audio-and-video/',
            'https://www.bleepingcomputer.com/forums/f/26/programming/',
            'https://www.bleepingcomputer.com/forums/f/243/virtual-machines-and-vm-programs/']

tablets_and_mobile_devices = ['https://www.bleepingcomputer.com/forums/f/227/which-tablet-should-i-buy/',
                              'https://www.bleepingcomputer.com/forums/f/184/apple-ios/',
                              'https://www.bleepingcomputer.com/forums/f/216/android-os/',
                              'https://www.bleepingcomputer.com/forums/f/223/windows-phone/',
                              'https://www.bleepingcomputer.com/forums/f/217/windows-tablets-microsoft-surface/']

gaming = ['https://www.bleepingcomputer.com/forums/f/20/computer-gaming/',
          'https://www.bleepingcomputer.com/forums/f/204/game-consoles/']

gadgets = ['https://www.bleepingcomputer.com/forums/f/175/cell-phones/',
           'https://www.bleepingcomputer.com/forums/f/176/ipod-zune-mp3-players/',
           'https://www.bleepingcomputer.com/forums/f/177/gps-devices/']

bleepingComputer_applications_and_guides = ['https://www.bleepingcomputer.com/forums/f/6/tutorials/',
                                            'https://www.bleepingcomputer.com/forums/f/85/windows-startup-programs-database/',
                                            'https://www.bleepingcomputer.com/forums/f/92/mini-guides-and-how-tos-simple-answers-to-common-questions/']

general_topics = ['https://www.bleepingcomputer.com/forums/f/5/general-chat/',
                  'https://www.bleepingcomputer.com/forums/f/64/introductions/',
                  'https://www.bleepingcomputer.com/forums/f/3/bleepingcomputer-announcements-comments-suggestions/',
                  'https://www.bleepingcomputer.com/forums/f/2/archived-news/',
                  'https://www.bleepingcomputer.com/forums/f/214/bleepingcomputer-offers-and-deals/',
                  'https://www.bleepingcomputer.com/forums/f/219/it-certifications-and-careers/',
                  'https://www.bleepingcomputer.com/forums/f/69/the-speak-easy/',
                  'https://www.bleepingcomputer.com/forums/f/58/forum-games/',
                  'https://www.bleepingcomputer.com/forums/f/59/photo-albums-images-and-videos/',
                  'https://www.bleepingcomputer.com/forums/f/35/tests-and-scribbles/']

categories = [microsoft_windows_support, alternative_operating_systems_support, hardware, security,
              internet_and_networking, software, tablets_and_mobile_devices, gaming, gadgets,
              bleepingComputer_applications_and_guides, general_topics]


def find_post_links_in_category(category_link, savedir, page_num,scraper):
    text = scraper.get(category_link).text
    soup = BeautifulSoup(text, 'html.parser')
    #print(soup.prettify())
    links = soup.find_all('a')
    filtered_links = []
    print("Total links: "+str(len(links)))
    for link in links:
        if link.has_attr('itemprop'):
            filtered_links.append(link.get('href'))

    next_link_href = None
    next_page_num  = None
    last_link_href = None
    last_page_num  = None
    next_link = soup.find('li', class_='next')
    last_link = soup.find('li', class_='last')

    #write the content to a file
    with open(os.path.join(save_dir, str(page_num)+".txt", 'w')) as f:
        f.write("\n".join(filtered_links))
    f.close()
    
    #https://www.bleepingcomputer.com/forums/f/167/windows-7/page-2?prune_day=100&sort_by=Z-A&sort_key=last_post&topicfilter=all
    if next_link is not None:
        next_link_href = next_link.find_next('a')['href']
        start_index = int(next_link_href.index('/page-')+len('/page-'))
        end_index = int(next_link_href.index('?prune_day='))
        next_page_num = int(next_link_href[start_index:end_index])
    waiting_time = random.randint(4,6) + random.randint(1,3)
    if next_page_num is not None:
        print("Waiting time: " + str(waiting_time) + " Next page number:" + str(next_link_href))
        time.sleep(waiting_time)
        print("Link: "+next_link_href)
        find_post_links_in_category(next_link_href, save_dir, next_page_num, scraper)


def parse(post_id,page_num,html_content):
    post_list = []
    soup = BeautifulSoup(html_content, 'html.parser')
    title_element = soup.find("title")
    keyword_element = soup.find("meta", attrs={'name': "keywords"})

    meta_elements = soup.find_all("meta")
    element_counter = -1
    for elem in meta_elements:
        if elem.has_attr("name") and elem["name"]=="description "and elem.has_attr("content"):
            post_content = elem["content"]

    div_elements = soup.find_all("div",{'class':'post_body'})
    for elem in div_elements:

        #collect the post date
        commentTimeElement = elem.find("abbr", attrs={'itemprop': 'commentTime', 'class': 'published', 'title':True})
        inner_elem = elem.find("div", attrs={'itemprop':'commentText','class':'post entry-content'})
        if inner_elem is not None:
            element_counter = element_counter + 1
            post_content = inner_elem.text.replace("\n", " ")
            internal_links = []
            external_links = []

            #collect links
            links = inner_elem.findAll("a", attrs={"href": True})
            for link in links:
                if link["href"].startswith("http://www.bleepingcomputer.com"):
                    internal_links.append(link["href"])
                else:
                   external_links.append(link["href"])

            post = Post()
            post.id = post_id
            if commentTimeElement is not None:
                date_time_obj = str(commentTimeElement["title"])
                post.published=datetime.datetime.strptime(date_time_obj[0:date_time_obj.rindex("-")], '%Y-%m-%dT%H:%M:%S')

            #print("Page Number: "+str(page_num)+" Element Counter: "+str(element_counter)+" Title: "+title_element.text+"  ")
            if page_num ==1 and element_counter == 0:
                post.title = title_element.text
                post.post_type = 1
                if keyword_element is not None:
                    post.tags = keyword_element["content"]
            else:
                post.post_type = 2
            post.text = post_content
            post.internal_links = internal_links
            post.external_links = external_links
            post_list.append(post)
    return post_list

def content_collector(scraper, post_link, post_number, next_page_num):
    print("Post Link: "+str(post_link))
    target_path = save_dir + str(post_number) + "_" + str(
        next_page_num) + ".txt"
    download_page = False #track whether we need to download the page or not
    text = None
    if os.path.exists(target_path):
        fr = open(target_path, mode='r')
        # read all lines at once
        text = fr.read()
        temp_post_list = parse(post_number, next_page_num, text)
        if len(temp_post_list)==0:
            download_page = True
        # close the file
        fr.close()

    if  os.path.exists(target_path) is False or (os.path.exists(target_path) is True and download_page is True):
        text = scraper.get(post_link).text
        waiting_time = random.randint(4,6) + random.randint(1,3)
        print("Download the page {}".format(post_link))
        time.sleep(waiting_time)
        fw = open(target_path,"w")
        fw.write(text)
        fw.close()
    else :
        print("No need to download the page: {}".format(target_path))

    soup = BeautifulSoup(text, 'html.parser')
    next_link_href = None
    next_page_num = None
    last_link_href = None
    last_page_num = None
    next_link = soup.find('li', class_='next')
    last_link = soup.find('li', class_='last')

    if next_link is not None:
        next_link_href = next_link.find_next('a')['href']
        start_index = int(next_link_href.index('/page-') + len('/page-'))
        next_page_num = int(next_link_href[start_index:len(next_link_href)])

        if next_page_num is not None:
            content_collector(scraper,next_link_href, post_number,next_page_num)

if __name__ == '__main__':
    # Change here to define your raw data save location
    scraper = cloudscraper.create_scraper()
    category_link = 'https://www.bleepingcomputer.com/forums/f/103/am-i-infected-what-do-i-do/'
    section_name = category_link.split('/')[-2]


    # Save index of a given forum
    save_dir = os.path.join(*['data', 'raw_data', 'page_index', section_name])
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    find_post_links_in_category(category_link, save_dir, 1, scraper)


    # Load saved indexes and download posts
    scraper = cloudscraper.create_scraper()
    list_files = [(f.path, f.name) for f in os.scandir(save_dir) 
    if f.is_file() and f.name.endswith(".txt") is True]
    file_counter = 0
    for (file_path,file_name) in list_files:
        file_counter = file_counter + 1
        file_number = int(file_name[0:len(file_name)-len(".txt")])
        print("File Number:"+str(file_number)+" File Counter: "+str(file_counter)+"/"+str(len(list_files)))
        with open(file_path) as f:
            content = f.readlines()
            for post_link in content:
                if post_link.startswith("https://www.bleepingcomputer.com/forums/t/"):
                    start_index = len("https://www.bleepingcomputer.com/forums/t/")
                    end_index = post_link.index("/", start_index)
                    post_number = post_link[start_index:end_index]
                    print("start processing :" + post_link + " " + file_name)
                    content_collector(scraper, post_link, post_number, 1)