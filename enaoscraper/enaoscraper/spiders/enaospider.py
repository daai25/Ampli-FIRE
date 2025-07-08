import scrapy


class EnaospiderSpider(scrapy.Spider):
    name = "enaospider"
    allowed_domains = ["everynoise.com"]
    start_urls = ["https://everynoise.com/everynoise1d.html"]

    def parse(self, response):
        print(response.css('iframe #__next > div > div > div.TrackList_backgroundColorContainer__vm8ks.TrackListWidget_trackListContainer__zpYQe > div > div > ol > li.TracklistRow_trackListRow__vrAAd.TracklistRow_isCurrentTrack__N2KN6.TracklistRow_isPlayable__U6o2r > h3'))
        yield {
                # #__next > div > div > div.TrackList_backgroundColorContainer__vm8ks.TrackListWidget_trackListContainer__zpYQe > div > div > ol > li.TracklistRow_trackListRow__vrAAd.TracklistRow_isCurrentTrack__N2KN6.TracklistRow_isPlayable__U6o2r > h3
                'name':  response.css('#__next > div > div > div.TrackList_backgroundColorContainer__vm8ks.TrackListWidget_trackListContainer__zpYQe > div > div > ol > li.TracklistRow_trackListRow__vrAAd.TracklistRow_isCurrentTrack__N2KN6.TracklistRow_isPlayable__U6o2r > h3').get(),
                'artist': response.css('ol li.TracklistRow_trackListRow__vrAAd.TracklistRow_isPlayable__U5o2r h4::text').get(),
                'genre': response.css('tr td.note a::text').get(),
        } 

