# from scrapy import Spider, Request
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, NoSuchElementException
# import requests
# import time

# import logging

# class EverynoiseSpider(Spider):
#     name = "everynoise_spider"
#     allowed_domains = ["everynoise.com"]
#     start_urls = ["https://everynoise.com/everynoise1d.html"]

#     def __init__(self, *args, **kwargs):
#         super(EverynoiseSpider, self).__init__(*args, **kwargs)

#         chrome_options = Options()
#         chrome_options.add_argument("--headless")  # comment out for debugging
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         chrome_options.add_argument("--disable-gpu")
#         chrome_options.add_argument("--window-size=1920,1080")
#         chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

#         try:
#             service = Service(ChromeDriverManager().install())
#             self.driver = webdriver.Chrome(service=service, options=chrome_options)
#             self.driver.implicitly_wait(10)
#         except Exception as e:
#             self.logger.error(f"Failed to initialize Chrome driver: {e}")
#             raise

#     def parse(self, response):
#         # Get genre links from the table
#         genre_rows = response.css("table tr")
        
#         for row in genre_rows[1:11]:  # Skip header row, limit to 10 for testing
#             genre_cell = row.css("td:nth-child(3)")  # Genre name is in 3rd column
#             if genre_cell:
#                 genre_name = genre_cell.css("::text").get()
#                 if genre_name and genre_name.strip():
#                     genre_name = genre_name.strip()
#                     # Construct the genre page URL with proper URL encoding
#                     genre_url = f"https://everynoise.com/engenremap-{genre_name.replace(' ', '%20').replace('&', '%26')}.html"
                    
#                     self.logger.info(f"Found genre: {genre_name}, URL: {genre_url}")
                    
#                     yield Request(
#                         url=genre_url,
#                         callback=self.parse_genre,
#                         meta={"genre_name": genre_name},
#                         dont_filter=True
#                     )

#     def parse_genre(self, response):
#         genre_name = response.meta["genre_name"]

#         self.logger.info(f"Scraping genre: {genre_name}")
        
#         try:
#             self.driver.get(response.url)
            
#             # Wait for page to load
#             WebDriverWait(self.driver, 15).until(
#                 EC.presence_of_element_located((By.ID, "spotify"))
#             )
            
#             # Switch to iframe
#             iframe = self.driver.find_element(By.ID, "spotify")
#             self.driver.switch_to.frame(iframe)

#             # Wait for Spotify content to load with multiple possible selectors
#             track_selectors = [
#                 "[data-testid='tracklist-row']",
#                 ".track-row",
#                 ".TracklistRow",
#                 "div[role='row']"
#             ]
            
#             track_rows = None
#             for selector in track_selectors:
#                 try:
#                     WebDriverWait(self.driver, 10).until(
#                         EC.presence_of_element_located((By.CSS_SELECTOR, selector))
#                     )
#                     track_rows = self.driver.find_elements(By.CSS_SELECTOR, selector)[:5]
#                     break
#                 except TimeoutException:
#                     continue
            
#             if not track_rows:
#                 self.logger.warning(f"No track rows found for {genre_name}")
#                 self.driver.switch_to.default_content()
#                 return

#             # Get first 5 song titles and artists
#             for track_row in track_rows:
#                 try:
#                     # Try different selectors for song title
#                     title_element = None
#                     title_selectors = [
#                         "[data-testid='internal-track-link'] div",
#                         "[data-testid='internal-track-link']",
#                         ".track-name",
#                         ".TracklistRow__track-name",
#                         "a[href*='track'] div",
#                         ".tracklist-name",
#                         "[role='gridcell'] a div"
#                     ]
                    
#                     for selector in title_selectors:
#                         try:
#                             title_element = track_row.find_element(By.CSS_SELECTOR, selector)
#                             if title_element.text.strip():
#                                 break
#                         except NoSuchElementException:
#                             continue
                    
#                     # Try different selectors for artist
#                     artist_element = None
#                     artist_selectors = [
#                         "[data-testid='track-row-column-artist'] a",
#                         "[data-testid='track-row-column-artist']",
#                         ".artist-name",
#                         ".TracklistRow__artists",
#                         ".track-artist",
#                         "a[href*='artist']",
#                         ".tracklist-artist"
#                     ]
                    
#                     for selector in artist_selectors:
#                         try:
#                             artist_element = track_row.find_element(By.CSS_SELECTOR, selector)
#                             if artist_element.text.strip():
#                                 break
#                         except NoSuchElementException:
#                             continue
                    
#                     if title_element and artist_element:
#                         title = title_element.text.strip()
#                         artist = artist_element.text.strip()
                        
#                         if title and artist:
#                             yield {
#                                 "genre": genre_name,
#                                 "title": title,
#                                 "artist": artist,
#                                 "url": response.url
#                             }
#                     else:
#                         self.logger.warning(f"Could not find title or artist elements for {genre_name}")
                        
#                 except Exception as e:
#                     self.logger.warning(f"Error extracting track data for {genre_name}: {e}")
#                     continue

#             self.driver.switch_to.default_content()

#         except TimeoutException:
#             self.logger.error(f"Timeout waiting for page elements for {genre_name}")
#         except Exception as e:
#             self.logger.error(f"Failed scraping {genre_name}: {e}")
#             # Switch back to default content even on error
#             try:
#                 self.driver.switch_to.default_content()
#             except:
#                 pass

#     def closed(self, reason):
#         if hasattr(self, 'driver'):
#             self.driver.quit()
