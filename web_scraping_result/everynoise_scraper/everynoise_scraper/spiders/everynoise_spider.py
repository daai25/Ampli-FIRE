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

        for link in genre_links[:3]:  
            logger.info(f"Processing genre link: {link}")
            #full_url = link.get_attribute("href")
            song_names, artist_names = self.get_songs_from_genre(link)
     
            for song_name, artist_name in zip(song_names, artist_names):
                yield {
                    #"link": full_url,
                    "genre": link.text.strip(),
                    "song name": song_name,
                    "artist name": artist_name
                }
        
        driver.quit()

        
    def get_songs_from_genre(self, link):
        driver = webdriver.Chrome()
        driver.get(link.get_attribute("href"))
        
        # Switch to iframe
        iframe = driver.find_element(By.XPATH, "//iframe[@id='spotify']")
        driver.switch_to.frame(iframe)
       
        # Get all elements with the specified class and print their text
        elements = driver.find_elements(By.CLASS_NAME, "TracklistRow_title__1RtS6")
        song_names = [el.text for el in elements]

        elements = driver.find_elements(By.CLASS_NAME, "TracklistRow_subtitle___DhJK")
        artist_names = [el.text for el in elements]

        # Switch back to main content
        driver.switch_to.default_content()
        
        driver.quit()

        return song_names, artist_names
    


        


 
