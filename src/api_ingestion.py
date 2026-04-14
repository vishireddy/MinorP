import os
import re
import requests
from bs4 import BeautifulSoup

RAW_DATA_DIR = "data/raw"

# Comprehensive curated list of 35 major Indian Acts with direct PDF links from indiacode.nic.in
KNOWN_ACTS = {
    # Fundamental Rights & Social Justice
    "Right to Information Act 2005":                "https://www.indiacode.nic.in/bitstream/123456789/2033/1/A2005-22.pdf",
    "Right to Education Act 2009":                  "https://www.indiacode.nic.in/bitstream/123456789/2249/1/200906.pdf",
    "Aadhaar Act 2016":                             "https://www.indiacode.nic.in/bitstream/123456789/2349/1/201618.pdf",
    "Protection of Human Rights Act 1993":          "https://www.indiacode.nic.in/bitstream/123456789/1587/1/199410.pdf",
    "SC ST Prevention of Atrocities Act 1989":      "https://www.indiacode.nic.in/bitstream/123456789/2091/1/198933.pdf",
    "Protection of Civil Rights Act 1955":          "https://www.indiacode.nic.in/bitstream/123456789/2081/1/195522.pdf",
    "Maintenance of Parents and Senior Citizens Act 2007": "https://www.indiacode.nic.in/bitstream/123456789/15476/1/maintenance_and_welfare_of_parents.pdf",
    "National Commission for Women Act 1990":       "https://www.indiacode.nic.in/bitstream/123456789/1616/1/199020.pdf",
    "Persons with Disabilities Act 2016":           "https://www.indiacode.nic.in/bitstream/123456789/15504/1/rpwda.pdf",

    # Criminal Law & National Security
    "Bharatiya Nyaya Sanhita 2023":                 "https://www.indiacode.nic.in/bitstream/123456789/20062/1/bns-2023.pdf",
    "Bharatiya Nagarik Suraksha Sanhita 2023":      "https://www.indiacode.nic.in/bitstream/123456789/20065/1/bnss-2023.pdf",
    "Bharatiya Sakshya Adhiniyam 2023":             "https://www.indiacode.nic.in/bitstream/123456789/20066/1/bsa-2023.pdf",
    "Unlawful Activities Prevention Act 1967":      "https://www.indiacode.nic.in/bitstream/123456789/1503/1/196737.pdf",
    "Armed Forces Special Powers Act 1958":         "https://www.indiacode.nic.in/bitstream/123456789/1472/1/195828.pdf",
    "Prevention of Money Laundering Act 2002":      "https://www.indiacode.nic.in/bitstream/123456789/2036/1/200315.pdf",
    "National Investigation Agency Act 2008":       "https://www.indiacode.nic.in/bitstream/123456789/2223/1/200934.pdf",

    # Data, Technology & Media
    "Digital Personal Data Protection Act 2023":    "https://www.indiacode.nic.in/bitstream/123456789/19967/1/dpdp_act_2023.pdf",
    "Telecom Act 2023":                             "https://www.indiacode.nic.in/bitstream/123456789/20087/1/telecom-act2023.pdf",
    "Copyright Act 1957":                           "https://www.indiacode.nic.in/bitstream/123456789/1367/1/195714.pdf",
    "Patents Act 1970":                             "https://www.indiacode.nic.in/bitstream/123456789/1392/1/197039.pdf",

    # Taxation, Trade & Finance
    "Central GST Act 2017":                         "https://www.indiacode.nic.in/bitstream/123456789/13048/1/cgst_act.pdf",
    "Foreign Exchange Management Act 1999":         "https://www.indiacode.nic.in/bitstream/123456789/1805/1/199942.pdf",
    "Reserve Bank of India Act 1934":               "https://www.indiacode.nic.in/bitstream/123456789/2472/1/193402.pdf",
    "Securities and Exchange Board of India Act 1992": "https://www.indiacode.nic.in/bitstream/123456789/2009/1/199215.pdf",
    "Insolvency and Bankruptcy Code 2016":          "https://www.indiacode.nic.in/bitstream/123456789/2345/1/201631.pdf",

    # Environment & Welfare
    "Environment Protection Act 1986":              "https://www.indiacode.nic.in/bitstream/123456789/4316/1/198629.pdf",
    "Wildlife Protection Act 1972":                 "https://www.indiacode.nic.in/bitstream/123456789/1726/1/197253.pdf",
    "Forest Rights Act 2006":                       "https://www.indiacode.nic.in/bitstream/123456789/2070/1/200702.pdf",
    "MGNREGA 2005":                                 "https://www.indiacode.nic.in/bitstream/123456789/2043/1/200542.pdf",
    "National Food Security Act 2013":              "https://www.indiacode.nic.in/bitstream/123456789/6062/1/national_food_security_act.pdf",
    "Disaster Management Act 2005":                 "https://www.indiacode.nic.in/bitstream/123456789/2034/1/200553.pdf",

    # Governance & Constitutional Bodies
    "Contempt of Courts Act 1971":                  "https://www.indiacode.nic.in/bitstream/123456789/1582/1/197170.pdf",
    "Official Languages Act 1963":                  "https://www.indiacode.nic.in/bitstream/123456789/1378/1/196319.pdf",
    "Inter-State River Water Disputes Act 1956":    "https://www.indiacode.nic.in/bitstream/123456789/1480/1/195633.pdf",
    "Finance Commission Miscellaneous Act 1951":    "https://www.indiacode.nic.in/bitstream/123456789/1463/1/195140.pdf",
}


def slugify(name: str) -> str:
    """Converts an Act name to a safe filename slug."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')


def _download_from_url(url: str, filepath: str) -> tuple[bool, str]:
    """Core download function that handles both direct PDF URLs and India Code HTML pages."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers, timeout=60, stream=True)
    response.raise_for_status()
    
    content_type = response.headers.get("Content-Type", "")
    
    # If it's a direct PDF link, stream it straight to disk
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, filepath
    
    # Otherwise, it's an HTML page — scrape the PDF download link from the India Code page
    soup = BeautifulSoup(response.text, "html.parser")
    
    # India Code bitstream links are usually in <a> tags with /bitstream/ or ending in .pdf
    pdf_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/bitstream/" in href and href.lower().endswith(".pdf"):
            pdf_link = href if href.startswith("http") else f"https://www.indiacode.nic.in{href}"
            break
    
    if not pdf_link:
        # Try any .pdf link on the page
        for a in soup.find_all("a", href=True):
            if a["href"].lower().endswith(".pdf"):
                pdf_link = a["href"] if a["href"].startswith("http") else f"https://www.indiacode.nic.in{a['href']}"
                break
    
    if not pdf_link:
        return False, "Could not find a PDF download link on this India Code page."
    
    # Download the actual PDF
    pdf_resp = requests.get(pdf_link, headers=headers, timeout=60, stream=True)
    pdf_resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in pdf_resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return True, filepath


def download_act_pdf(act_name: str, url: str, is_amendment: bool = False) -> tuple[bool, str]:
    """Downloads a PDF from India Code and saves it to data/raw/."""
    prefix = "amendment" if is_amendment else "base"
    safe_name = slugify(act_name)
    filename = f"{prefix}_{safe_name}.pdf"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    if os.path.exists(filepath):
        return True, f"Already exists: {filename}"
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    try:
        return _download_from_url(url, filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False, str(e)


def fetch_from_any_url(url: str, custom_name: str, is_amendment: bool = False) -> tuple[bool, str]:
    """
    Scraper feature: accepts any India Code or direct PDF URL pasted by the admin.
    Returns (success, filepath_or_error).
    """
    prefix = "amendment" if is_amendment else "base"
    safe_name = slugify(custom_name) if custom_name else slugify(url.split("/")[-1].replace(".pdf", ""))
    filename = f"{prefix}_{safe_name}.pdf"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    if os.path.exists(filepath):
        return True, f"File already exists: {filename}"
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    try:
        return _download_from_url(url, filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False, str(e)


def get_available_acts() -> dict:
    return KNOWN_ACTS


def search_acts(query: str) -> dict:
    query = query.lower()
    return {name: url for name, url in KNOWN_ACTS.items() if query in name.lower()}
