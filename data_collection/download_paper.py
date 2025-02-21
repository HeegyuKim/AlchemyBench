import pandas as pd
import os
import requests
from pprint import pprint
import glob
import time
from tqdm.auto import tqdm

files = list(glob.glob("s2api-result/*.jsonl"))

dfs = [pd.read_json(file, lines=True) for file in files]
df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=["paperId"])
print(f"Total papers: {len(df)}")

df["venue_domain"] = df.publicationVenue.apply(lambda x: x['url'].split("/")[2] if x and "url" in x else None)
df["pdf_url"] = df.openAccessPdf.apply(lambda x: x['url'] if isinstance(x, dict) else None)
df["pdf_url_domain"] = df.pdf_url.apply(lambda x: x.split("/")[2] if x else None)
df["doi"] = df.externalIds.apply(lambda x: x['DOI'] if x and "DOI" in x else None)
df["filename"] = df.apply(lambda x: f"pdfs/{x.paperId}.pdf", axis=1)

df = df[df.filename.apply(lambda x: not os.path.exists(x))]
print(f"Total papers (not downloaded): {len(df)}")

available_domains = ["pubs.rsc.org", "mdpi.com", "nature.com", "link.springer.com"]

def check_domain(url):
    if url is None:
        return False
    
    for domain in available_domains:
        if domain in url:
            return True
    return False

new_df = df[df.pdf_url.apply(check_domain)]
del df
df = new_df
print(f"Total papers (not downloaded, available domains): {len(df)}")



os.makedirs("pdfs", exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
}
# available_domains = ["pubs.acs.org"]
rows = df[df.openAccessPdf.notna()]
for i, row in tqdm(rows.iterrows(), total=len(rows)):
    # print(row)
    url = row.openAccessPdf['url']
    filename = row.filename
    print(url)

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"Failed to download {url}")
        continue

    with open(filename, 'wb') as f:
        f.write(r.content)

    print(f"Downloaded {filename}")
    
    time.sleep(3)



