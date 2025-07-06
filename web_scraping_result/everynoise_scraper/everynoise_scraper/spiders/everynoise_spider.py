from scrapy.spiders import Spider
import scrapy
from bs4 import BeautifulSoup
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService


import colorlog

handler = colorlog.StreamHandler()

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)

handler.setFormatter(colorlog.ColoredFormatter('%(red)s%(levelname)s: %(message)s'))


class EverynoiseSpiderSpider(scrapy.Spider):
    name = "everynoise_spider"
    allowed_domains = ["everynoise.com"]
    start_urls = ["https://everynoise.com/everynoise1d.html"]

    def parse(self, response):   
        logger.info("*********************** Starting parse function ***********************")
    
        # Launch headless Chrome
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(service=ChromeService(), options=options)
        driver.get(response.url)

        #relative_links = driver.find_elements(by=By.XPATH, value="/html/body/table/tbody/tr[1]/td[3]/a")
        
        genre_links = driver.find_elements(by=By.XPATH, value="/html/body/table/tbody/tr/td[3]/a")
        logger.info(f"*********************** Found {len(genre_links)} relative links ***********************")
       
    
        for link in genre_links[:5]:  
            logger.info(f"*********************** Processing genre link: {link} ***********************")             
            #genre_name = href.replace("everynoise1d-", "").replace(".html", "")
            full_url = link.get_attribute("href")
            logger.info(f"*********************** Processing genre link: {full_url} ***********************")   
            yield {
                #"genre" : genre_name,
                "link": full_url  
            }
            # logger.info(f"*********************** Start Function get_songs_from_genre(link) ***********************")
            # song_list = self.get_songs_from_genre(href)
            # logger.info(f"*********************** Found {len(song_list)} songs in genre {genre_name} at {full_url} ***********************")
            
        logger.info("*********************** Scraping complete ***********************")
        driver.quit()
        
    def get_songs_from_genre(self, link):

        logger.info(f"********************  Scraping songs from {link} ********************")
        driver = webdriver.Chrome()
        driver.get(self.link)
        logger.info(f"********************  Page loaded, switching to iframe")

        # Switch to iframe
        iframe = driver.find_element(By.XPATH, "//iframe[@id='spotify']")
        driver.switch_to.frame(iframe)
        
        # Scrape data inside iframe
        # Find if 'Lush Life' appears anywhere in the iframe's HTML
        iframe_html = driver.page_source
        
        logger.info(f"********************  Scraping song names from iframe")
        # Get all elements with the specified class and print their text
        elements = driver.find_elements(By.CSS_SELECTOR, ".tracklist-name, .TracklistRow__track-name")

        logger.info(f"********************  Found {len(elements)} song elements in iframe")
        song_names = []
        for el in elements: 
            song_names.append(el.text)

        logger.info(f"********************  Found {len(song_names)} songs in iframe")
        
        # Switch back to main content
        driver.switch_to.default_content()
        
        driver.quit()

        return song_names

        


 
