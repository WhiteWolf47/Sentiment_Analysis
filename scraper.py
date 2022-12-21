# Importing libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# scraping the required text from the websites
def get_info(url):

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"
    }
    res = requests.get(url, headers = headers)
    html_page = res.content

    soup = BeautifulSoup(html_page, 'html.parser')

    imp = soup.find_all('p')
    output = ''
    for i in imp:
        output += '{} '.format(i.text)

    return output

# Main function
if __name__ == '__main__':
    # Reading the input file
    df = pd.read_csv('Input.xlsx - Sheet1.csv')

    # getting the text from the websites
    df['Output'] = df['URL'].apply(get_info)

    # Writing the text to a .txt files
    for i in range(len(df)):
        #print(df['Output'][i])
        with open(f"articles/{df['URL_ID'][i]}.txt", 'a') as f:
            f.write(str(df['Output'][i]))
        f.close()


    