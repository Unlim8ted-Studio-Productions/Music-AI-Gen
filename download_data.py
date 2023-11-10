import requests
from bs4 import BeautifulSoup
import os


def crawl_and_download_music(url, download_directory, max_downloads):
    # Fetch the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Extract links to music files
    music_links = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].endswith((".mp3", ".wav", ".flac", ".mid", ".MID"))
    ]
    # Create the download directory if it doesn't exist                            ############################
    os.makedirs(download_directory, exist_ok=True)  ############################
    ############################
    # Download the music files up to the specified maximum                            ############################
    for i, link in enumerate(music_links[:max_downloads]):  ############################
        music_url = (
            link if link.startswith("http") else f"{url}/{link}"
        )  ############################
        response = requests.get(music_url)  ############################
        with open(
            os.path.join(download_directory, f"music_{i+1}.mid"), "wb"
        ) as file:  ############################
            file.write(response.content)  ############################
            ############################                                           #
            ############################                                        # * * #


if __name__ == "__main__":  ############################                        #   #
    # Example usage                            ############################    #      #
    crawl_and_download_music(  ############################                    #        #
        "https://bushgrafts.com/midi/",  ############################             |  |             @@@@@@@@@@######################
        "data\\",  ############################                              @@@@@@@@@@######################
        1000,  ############################                              @@@@@@@@@@######################
    )  ############################                              @@@@@@@@@@######################
    ###########################                              @@@@@@@@@@##
    crawl_and_download_music(  ############################
        "https://bushgrafts.com/jazz/",  ############################
        "data\\",  ############################
        1000,  ############################
    )  ############################

# import requests
# from bs4 import BeautifulSoup
# import os
# import sys
# import time
# import json
# import re
#
#
# class DMjazzCrawler:
#    BASE_URL = "http://www.bushgrafts.com/jazz"
#    ROOT = "archive"
#
#    def __init__(self, sleep_time=0.1, log=True):
#        self.sleep_time = sleep_time
#        self.log = log
#
#    def _request_url(self, url, doctype="html"):
#        # set header
#        response = requests.get(url, headers={"Cache-Control": "max-age=0"})
#
#        # sleep
#        time.sleep(self.sleep_time)
#
#        # return
#        if doctype == "html":
#            soup = BeautifulSoup(response.text, "html.parser")
#            return soup
#        elif doctype == "content":
#            return response.content
#        else:
#            return response
#
#    def _log_print(self, log, quite=False):
#        if not quite:
#            print(log)
#
#        if self.log:
#            with open("log.txt", "a") as f:
#                print(log, file=f)
#
#    def fetch_song(self):
#        self.soup = self._request_url(self.BASE_URL + "/midi.htm")
#        a_list = dmc.soup.find_all("a")
#        midi_list = []
#        name_list = []
#
#        cnt = 0
#        for idx, a in enumerate(a_list):
#            str_ = a.get("href")
#            if str_ and (str_ not in midi_list) and (".mid" in str_):
#                song_name = re.sub("\s+", " ", a.text.replace("\r\n", "")).strip(" ")
#                if song_name:
#                    midi_fn = str_.split("/")[1]
#                    midi_list.append(midi_fn)
#                    name_list.append(song_name)
#                    print("%3d | %-40s %s" % (idx, song_name, midi_fn))
#                    cnt += 1
#
#        self._log_print("Total: %d" % cnt)
#
#        return dict(zip(midi_list, name_list))
#
#    def crawl_song(self, song_dict):
#        for idx, k in enumerate(song_dict.keys()):
#            url = self.BASE_URL + "/Midi%20site/" + k
#            print("%3d %s" % (idx, url))
#            content = self._request_url(url, doctype="content")
#
#            with open(("data\\", k), "wb") as f:
#                f.write(content)
#
#    def run(self):
#        song_dict = self.fetch_song()
#
#        if not os.path.exists(self.ROOT):
#            os.makedirs(self.ROOT)
#        with open(os.path.join(self.ROOT, "archive.json"), "w") as f:
#            json.dump(song_dict, f)
#
#        self.crawl_song(song_dict)
#
#
# class DMMIDICrawler:
#    BASE_URL = "http://www.bushgrafts.com/midi"
#    ROOT = "archive"
#
#    def __init__(self, sleep_time=0.1, log=True):
#        self.sleep_time = sleep_time
#        self.log = log
#
#    def _request_url(self, url, doctype="html"):
#        # set header
#        response = requests.get(url, headers={"Cache-Control": "max-age=0"})
#
#        # sleep
#        time.sleep(self.sleep_time)
#
#        # return
#        if doctype == "html":
#            soup = BeautifulSoup(response.text, "html.parser")
#            return soup
#        elif doctype == "content":
#            return response.content
#        else:
#            return response
#
#    def _log_print(self, log, quite=False):
#        if not quite:
#            print(log)
#
#        if self.log:
#            with open("log.txt", "a") as f:
#                print(log, file=f)
#
#    def fetch_song(self):
#        self.soup = self._request_url(self.BASE_URL + "/midi.htm")
#        a_list = dmc.soup.find_all("a")
#        midi_list = []
#        name_list = []
#
#        cnt = 0
#        for idx, a in enumerate(a_list):
#            str_ = a.get("href")
#            if str_ and (str_ not in midi_list) and (".mid" in str_):
#                song_name = re.sub("\s+", " ", a.text.replace("\r\n", "")).strip(" ")
#                if song_name:
#                    midi_fn = str_.split("/")[1]
#                    midi_list.append(midi_fn)
#                    name_list.append(song_name)
#                    print("%3d | %-40s %s" % (idx, song_name, midi_fn))
#                    cnt += 1
#
#        self._log_print("Total: %d" % cnt)
#
#        return dict(zip(midi_list, name_list))
#
#    def crawl_song(self, song_dict):
#        for idx, k in enumerate(song_dict.keys()):
#            url = self.BASE_URL + "/Midi%20site/" + k
#            print("%3d %s" % (idx, url))
#            content = self._request_url(url, doctype="content")
#
#            with open(("data\\", k), "wb") as f:
#                f.write(content)
#
#    def run(self):
#        song_dict = self.fetch_song()
#
#        if not os.path.exists(self.ROOT):
#            os.makedirs(self.ROOT)
#        with open(os.path.join(self.ROOT, "archive.json"), "w") as f:
#            json.dump(song_dict, f)
#
#        self.crawl_song(song_dict)
#
#
# if __name__ == "__main__":
#    dmc = DMjazzCrawler()
#    dmc.run()
#    dmc = DMMIDICrawler()
#    dmc.run()
#
