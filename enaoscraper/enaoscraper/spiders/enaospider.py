import scrapy
from scrapy_playwright.page import PageMethod


class EnaospiderSpider(scrapy.Spider):
    name = "everynoise_spider"
    allowed_domains = ["everynoise.com"]
    start_urls = ["https://everynoise.com/"]

    def parse(self, response):
        custom_settings = {
            'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
            'PLAYWRIGHT_BROWSER_TYPE': 'firefox',

            'DOWNLOAD_HANDLERS': {
                'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
                'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            },

            'PLAYWRIGHT_LAUNCH_OPTIONS': {
                'headless': True,
                'timeout': 30 * 1000,
            },

            'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60 * 1000,

            'PLAYWRIGHT_CONTEXTS': {
                'default': {
                    'viewport': {'width': 1920, 'height': 1080},
                    'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                },
            },

            'RETRY_TIMES': 3,
            'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],

            'DOWNLOAD_DELAY': 1,
            'RANDOMIZE_DOWNLOAD_DELAY': True,
            'CLOSESPIDER_TIMEOUT_NO_ITEM': 120,
            'ROBOTSTXT_OBEY': False,
        }

    def parse(self, response):
        #self.logger.info("Parsing response from %s", response.url)
    
        genre_links = response.css("div.canvas a")
        #self.logger.info("Found %d genre links", len(genre_links))

        #if not genre_links:
        #    self.logger.warning("No genres found. Check selectors or rendering.")

        for a in genre_links:
            yield {
                "genre": response.css('tr td.note a::text').get(),
                "link": response.urljoin(a.attrib["href"]),
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
