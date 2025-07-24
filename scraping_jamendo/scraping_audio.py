from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import requests
import os
import undetected_chromedriver as uc


def scrape_jamendo_songs():
    driver = uc.Chrome()
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Optional: Hide browser
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Skip the broken homepage search — go straight to results page
    search_url = f"https://www.jamendo.com/explore/playlists"
    driver.get(search_url)
    time.sleep(10)  # Wait for results to load
    count = 0
    album_count = 0
    not_download = 0

    def scroll_down():
        # Scroll down to the bottom of the page
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down by 1000 pixels
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait for the page to load content
            time.sleep(2)

            # Calculate new height and check if the bottom is reached
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    scroll_down()

    links = driver.find_elements(By.XPATH, "//a[@class='link-wrap']")
    # Extract URLs first (because once you navigate, original elements become stale)
    album_urls = [link.get_attribute("href") for link in links]
    print(album_urls)
    for i in album_urls:
        album_count += 1

    lst_songs = []
    music_dir = 'music'
    os.makedirs(music_dir, exist_ok=True)

    time.sleep(30)

    for u in album_urls:
        driver.get(u)
        time.sleep(3)
        count += 1
        print(f'{count}/{album_count}')

        scroll_down()

        links = driver.find_elements(By.XPATH, "//a[@class='link-wrap js-trackrow-playlistpage-link']")
        # Extract URLs first (because once you navigate, original elements become stale)
        song_urls = [link.get_attribute("href") for link in links]
        print(song_urls)

        for url in song_urls:
            driver.get(url)
            time.sleep(5)  # Wait for page to load; adjust as needed

            try:
                meta_audio = driver.find_element(By.CSS_SELECTOR, 'meta[property="og:audio"]')
                audio_url = meta_audio.get_attribute("content")
                print("Audio URL:", audio_url)

                song_name = driver.find_element(By.CSS_SELECTOR, "h1.primary > span").text.lower()
                print("Song Name:", song_name)

                artist_name = driver.find_element(By.CSS_SELECTOR, "a.secondary > span").text.lower()
                print("Artist Name:", artist_name)

                dct = {
                    "song_name": song_name,
                    "artist": artist_name,
                    "audio_url": audio_url
                }
                lst_songs.append(dct)
                with open('more_data.json', 'w') as json_file:
                    json.dump(lst_songs, json_file, indent=4)

                down_button = driver.find_element(By.CSS_SELECTOR,
                                                  "i[class='icon icon-download'")
                down_button.click()
                time.sleep(3)

                load_button = driver.find_element(By.CSS_SELECTOR,
                                                  'div[class="js-download-button"]')
                load_button.click()

            except Exception as e:
                print(f"Error scraping {url}: {e}")
                not_download+=1
                print(not_download)


        """section = driver.find_element(By.XPATH, "//section[.//h1[text()='Description']]")
        description = section.find_element(By.TAG_NAME, "p").text
        print("Description:", description)"""

    driver.quit()


if __name__ == "__main__":
    scrape_jamendo_songs()
