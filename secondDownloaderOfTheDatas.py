import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_songs(url, output_folder, max_songs):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all the download links on the page
    download_links = soup.find_all("a", class_="playicn")

    # Limit the number of songs to download
    download_links = download_links[:max_songs]

    for link in download_links:
        song_url = urljoin(url, link["href"])
        song_response = requests.get(song_url)

        # Get the filename from the URL
        filename = os.path.join(output_folder, os.path.basename(song_url))

        # Save the file
        with open(filename, "wb") as file:
            file.write(song_response.content)

        print(f"Downloaded: {filename}")

    print(f"{len(download_links)} songs downloaded successfully.")

# Example usage:
url = "https://www.jamendo.com/community/chiptune"#https://freemusicarchive.org/genre/Chiptune/"
output_folder = "data"
max_songs = 200  # Specify the maximum number of songs to download

download_songs(url, output_folder, max_songs)
