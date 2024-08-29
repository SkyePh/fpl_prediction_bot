from bs4 import BeautifulSoup
import pandas as pd
from playwright.sync_api import sync_playwright


url = "https://fbref.com/en/comps/9/stats/Premier-League-Stats#all_stats_standard"


def playwright_func():
    with sync_playwright() as play:

        #launch a browser
        browser = play.chromium.launch(headless=True)
        page = browser.new_page()

        #nav to url
        page.goto(url)

        #wait to fully load (this table needs it)
        page.wait_for_load_state('networkidle')

        #parse the page now that its loaded
        html = page.content()

        browser.close()

        return html


def scrape_stats(html):

    # parse the page
    soup = BeautifulSoup(html, 'lxml')

    # find table with player stats with table ID
    table = soup.find('table', {'id': 'stats_standard'})

    if table:

        # extract headers. th = table header. [1:] skips the first empty column header
        headers = [th.text for th in table.find('thead').find_all('th')][1:]

        # extract data from the rows. tbody = table body. tr = table row
        rows = table.find('tbody').find_all('tr')

        # make list to store player data
        player_data = []

        # loop through each row and extract data and add them to list
        for row in rows:
            if row.find('th', {'scope': 'row'}):  # makes sure its a player row

                # get player data. td = table data
                player_stats = [td.text.strip() for td in row.find_all('td')]

                # Debugging: Print out the row and its length
                print(f"Row {len(player_data) + 1}: {player_stats}, Length: {len(player_stats)}")

                # Handle rows with missing data by padding with None or trimming extra columns
                if len(player_stats) < len(headers):
                    # Pad missing columns with None
                    player_stats.extend([None] * (len(headers) - len(player_stats)))
                    print(f"Padded row {len(player_data) + 1}: {player_stats}")
                elif len(player_stats) > len(headers):
                    # Trim extra columns
                    player_stats = player_stats[:len(headers)]
                    print(f"Trimmed row {len(player_data) + 1}: {player_stats}")

                player_data.append(player_stats)

        #checks if headers and player_data are valid
        if headers and player_data:

            # make a data frame (df) to store data with pandas
            df = pd.DataFrame(player_data, columns=headers)

            # show the data frame
            print(df)

            # save everything to a CSV file
            df.to_csv('prem_player_stats.csv', index=False)

        else:
            print("No valid headers or player data found.")
    else:
        #raise error
        print("Table with this ID cannot be found")


#main
if __name__ == '__main__':

    html = playwright_func()

    scrape_stats(html)