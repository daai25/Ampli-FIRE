from selenium import webdriver
from selenium.webdriver.common.by import By
 

def find_songs_from_genre(string_link):
    driver = webdriver.Chrome()
    driver.get(string_link)
    
    # Switch to iframe
    iframe = driver.find_element(By.XPATH, "//iframe[@id='spotify']")
    driver.switch_to.frame(iframe)
    
    # Scrape data inside iframe
    # Find if 'Lush Life' appears anywhere in the iframe's HTML
    iframe_html = driver.page_source
    
    # Get all elements with the specified class and print their text
    elements = driver.find_elements(By.CLASS_NAME, "TracklistRow_title__1RtS6")
    for el in elements:
        print(el.text)
    
    # Switch back to main content
    driver.switch_to.default_content()
    
    driver.quit()


