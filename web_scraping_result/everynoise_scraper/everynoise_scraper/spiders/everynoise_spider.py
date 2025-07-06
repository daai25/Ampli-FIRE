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
        # Launch headless Chrome
        options = webdriver.ChromeOptions()
        #options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(service=ChromeService(), options=options)
        driver.get(response.url)

        genre_links = driver.find_elements(by=By.XPATH, value="/html/body/table/tbody/tr/td[3]/a")
        logger.info(f"*********************** Found {len(genre_links)} relative links ***********************")
       
        for link in genre_links[:3]:  
            #genre_name = href.replace("everynoise1d-", "").replace(".html", "")
            full_url = link.get_attribute("href")
            song_names = self.get_songs_from_genre(link)
            logger.info(f"*********************** Found {len(song_names)} songs in genre: {full_url} ***********************")
            yield {
                #"genre" : genre_name,
                "link": full_url,
                "songs": song_names  
            }
        
        # song_names = self.get_songs_from_genre(genre_links[0])
        # logger.info(f"*********************** Found {len(song_names)} songs in the first genre ***********************")
                  


        logger.info("*********************** Scraping complete ***********************")
        driver.quit()


    def get_songs_from_genre(self, link):
        driver = webdriver.Chrome()
        driver.get(link.get_attribute("href"))
        
        # Switch to iframe
        iframe = driver.find_element(By.XPATH, "//iframe[@id='spotify']")
        driver.switch_to.frame(iframe)
        
        # Scrape data inside iframe
        # Find if 'Lush Life' appears anywhere in the iframe's HTML
        iframe_html = driver.page_source
        
        # Get all elements with the specified class and print their text
        elements = driver.find_elements(By.CLASS_NAME, "TracklistRow_title__1RtS6")
        song_names = [el.text for el in elements]
        # for el in elements:
        #     logger.info(el.text)
        
        # Switch back to main content
        driver.switch_to.default_content()
        
        driver.quit()

        return song_names
    


        


 
