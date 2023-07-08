################################################################################
### Step 1
################################################################################

import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import time

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

HEADERS = {
   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36' 
}
# Define root domain to crawl
"""Notes:
  Variable ign_wiki_path is used in get_domain_hyperlinks to reject urls that do not contain the variable. This may be incorrect on some sites as their paths may be structured differently.
  Remove the path check in get_domain_hyperlinks if you no longer desire this behaviour.
  if url_obj.netloc == local_domain and url_obj.path == ign_wiki_path:
    clean_link = link 
"""
domain = "ign.com"
ign_wiki_path = 'wikis/elden-ring'
full_url = f'https://www.ign.com/{ign_wiki_path}'

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

################################################################################
### Step 2
################################################################################

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            
            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

################################################################################
### Step 3
################################################################################

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain and url_obj.path == ign_wiki_path:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


################################################################################
### Step 4
################################################################################

def remove_newlines(str):
    str = str.replace('\n', ' ')
    str = str.replace('\\n', ' ')
    str = str.replace('  ', ' ')
    str = str.replace('  ', ' ')
    return str

def sanitize_file_name(file_name):
    # Define a regular expression pattern to match invalid characters
    invalid_chars = r'[\/:*?"<>|]'

    # Use the re.sub() function to replace invalid characters with an empty string
    sanitized_name = re.sub(invalid_chars, '', file_name)

    return sanitized_name

def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("crawler_text/"):
            os.mkdir("crawler_text/")

    if not os.path.exists("crawler_text/"+local_domain+"/"):
            os.mkdir("crawler_text/" + local_domain + "/")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress
        try:
          time.sleep(1) # being nice to the server
          # Save text from the url to a <url>.txt file
          with open('crawler_text/'+local_domain+'/'+sanitize_file_name(url[8:].replace("/", "_") + ".txt"), "w", encoding="UTF-8") as f:

              # Get the text from the URL using BeautifulSoup
              response = requests.get(url, headers=HEADERS)
              if response.ok:
                soup = BeautifulSoup(response.text, "html.parser")

                # Get the text but remove the tags
                text = soup.get_text()

                # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                if ("You need to enable JavaScript to run this app." in text):
                    print("Unable to parse page " + url + " due to JavaScript being required")
                
                # Otherwise, write the text to the file in the text directory
                f.write(remove_newlines(text))
              else:
                  print(f'Unable to scrape url: {url}')

          # Get the hyperlinks from the URL and add them to the queue
          for link in get_domain_hyperlinks(local_domain, url):
              if link not in seen:
                  queue.append(link)
                  seen.add(link)
        except Exception as error:
            print(f'Exception occured scraping url: {url}\n Exception: {error}')

crawl(full_url)

