from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import os

#Path to the json with the links and genre of the songs
folder = "C:\\Users\\annab\\Ampli-FIRE\\web_scraping_result\\everynoise_scraper\\everynoise_scraper\\"

#name of the json file, that the scraped data would be saved to
filename = os.path.join(folder, "music_data.json")

driver = webdriver.Chrome()

#read through the json
with open("enaoscraper\enaoscraper\music_genre_everynoice.json", "r") as file:
    music_json = json.load(file)

size = len(music_json)

music = []
http_link_everynoise = "https://everynoise.com/"

#scraping all the data we need for every link that is in the json we read through
for i in range(size):
    playlist_link = music_json[i]["link"]
    http_link_to_playlist = http_link_everynoise + playlist_link
    driver.get(http_link_to_playlist)
     
    # Switch to iframe
    iframe = driver.find_element(By.XPATH, "//iframe[@id='spotify']")
    driver.switch_to.frame(iframe)
 
    # Scrape data inside iframe
    iframe_html = driver.page_source
 
    # Get all songs with the artists to each song for every genre there is
    song_names = driver.find_elements(By.CLASS_NAME, "TracklistRow_title__1RtS6")
    song_artist = driver.find_elements(By.CLASS_NAME, "TracklistRow_subtitle___DhJK")
    for name_el, artist_el in zip(song_names, song_artist):
        current_genre = music_json[i]["genre"]
        current_name = name_el.text
        current_artist = artist_el.text.strip()

        music.append({
        "genre": current_genre,
        "song_name": current_name,
        "artist": current_artist
        })
#putting all the data into a json file
with open(filename, "w") as file:
    json.dump(music, file, indent=4)
 
# Switch back to main content
driver.switch_to.default_content()
 
driver.quit()