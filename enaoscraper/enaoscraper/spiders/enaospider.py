import scrapy
from scrapy_playwright.page import PageMethod


class EnaospiderSpider(
    scrapy.Spider):
    name = "enaospider"
    allowed_domains = ["everynoise.com"]
    start_urls = ["https://everynoise.com/everynoise1d.html"]


    def parse(self, response):
        # self.logger.info("Parsing response from %s", response.url)

        genre_links = response.css("td.note a")
        # self.logger.info("Found %d genre links", len(genre_links))

        # if not genre_links:
        #    self.logger.warning("No genres found. Check selectors or rendering.")

        for a in genre_links:
            yield {
                "genre": a.css("::text").get(),
                "link": a.css("::attr(href)").get(),
            }


    def start_requests(self):
        """
        Generate the initial requests with Playwright configuration.
        This method enables JavaScript rendering for each URL.
        """
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={
                    # Enable Playwright for this request (JavaScript execution)
                    'playwright': True,
    
                    # Define page methods to execute in the browser
                    'playwright_page_methods': [
                        # Wait for quote elements to appear (JavaScript loaded content)
                        PageMethod('wait_for_selector', 'div.canvas a', state='attached'),
    
                        # Wait until no network requests are ongoing (page fully loaded)
                        PageMethod('wait_for_load_state', 'networkidle'),
                    ],
                }
            )
