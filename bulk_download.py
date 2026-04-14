"""
Bulk downloader for Indian Acts - tries multiple strategies:
1. Session-based download (India Code with cookies)
2. Alternative official ministry/regulatory body URLs
"""
import os
import time
import requests
from bs4 import BeautifulSoup

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

ACTS = {
    # --- Additional acts with alternative official source URLs ---
    # SEBI (sebi.gov.in)
    "SEBI Act 1992":                    "https://www.sebi.gov.in/legal/acts/jun-2020/securities-and-exchange-board-of-india-act-1992_47454.html",
    "Securities Contracts Act 1956":    "https://www.sebi.gov.in/legal/acts/aug-2020/securities-contracts-regulation-act-1956_47509.html",
    # RBI (rbi.org.in)
    "RBI Act 1934":                     "https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/RBIA1934170311.pdf",
    "Banking Regulation Act 1949":      "https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/BREA010112.pdf",
    "FEMA 1999":                        "https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/FEMA42006.pdf",
    "Payment and Settlement Systems Act 2007": "https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/PSSA2007251218.pdf",
    # Ministry of Corporate Affairs (mca.gov.in)
    "Companies Act 2013":               "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
    "Insolvency and Bankruptcy Code 2016": "https://www.mca.gov.in/Ministry/pdf/IBC_2016_12012017.pdf",
    "Limited Liability Partnership Act 2008": "https://www.mca.gov.in/Ministry/pdf/LimitedLiabilityPartnershipAct2008.pdf",
    # Ministry of Labour (labour.gov.in)
    "Industrial Disputes Act 1947":     "https://labour.gov.in/sites/default/files/TheIndustrialDisputesAct1947.pdf",
    "Minimum Wages Act 1948":           "https://labour.gov.in/sites/default/files/TheMinimumWagesAct1948_0.pdf",
    "Payment of Wages Act 1936":        "https://labour.gov.in/sites/default/files/ThePaymentofWagesAct1936.pdf",
    "Factories Act 1948":               "https://labour.gov.in/sites/default/files/TheFactoriesAct1948.pdf",
    "Employees Provident Funds Act 1952": "https://labour.gov.in/sites/default/files/TheEmployeesProvidentFundsMiscProvisionsAct1952.pdf",
    # IRDAI (irdai.gov.in)
    "Insurance Act 1938":               "https://irdai.gov.in/ADMINCMS/cms/Uploaded_Files/Insurance%20Act%201938.pdf",
    "IRDA Act 1999":                    "https://irdai.gov.in/ADMINCMS/cms/Uploaded_Files/IRDA%20Act%201999.pdf",
    # CCI (cci.gov.in)
    "Competition Act 2002":             "https://www.cci.gov.in/sites/default/files/the_competition_act.pdf",
    # TRAI (trai.gov.in)
    "TRAI Act 1997":                    "https://trai.gov.in/sites/default/files/TRAI_act.pdf",
    # Ministry of Health (mohfw.gov.in)
    "Drugs and Cosmetics Act 1940":     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdfdocuments/acts_rules/NewDrugsandCosmeticsAct1940.pdf",
    # FSSAI (fssai.gov.in)
    "Food Safety and Standards Act 2006": "https://www.fssai.gov.in/upload/uploadfiles/files/Food_Safety_Stadards_Act_2006.pdf",
    # Ministry of Environment
    "Air Prevention and Control of Pollution Act 1981": "https://cpcb.nic.in/air-prevention-and-control-of-pollution-act-1981/",
    "Water Prevention and Control of Pollution Act 1974": "https://cpcb.nic.in/water-prevention-and-control-of-pollution-act/",
    # Election Commission
    "Representation of People Act 1950": "https://eci.gov.in/files/file/9028-representation-of-the-people-act-1950/",
    # India Code (direct bitstream, try with session)
    "Arms Act 1959":                    "https://www.indiacode.nic.in/bitstream/123456789/1544/1/195954.pdf",
    "Narcotics Drugs and Psychotropic Substances Act 1985": "https://www.indiacode.nic.in/bitstream/123456789/1886/1/198561.pdf",
    "Motor Vehicles Act 1988":          "https://www.indiacode.nic.in/bitstream/123456789/6370/1/motor-vehicles-act.pdf",
    "Code of Civil Procedure 1908":     "https://www.indiacode.nic.in/bitstream/123456789/17510/1/cpc-1908.pdf",
    "Transfer of Property Act 1882":    "https://www.indiacode.nic.in/bitstream/123456789/2263/1/188204.pdf",
    "Indian Contract Act 1872":         "https://www.indiacode.nic.in/bitstream/123456789/2187/1/187209.pdf",
    "Arbitration and Conciliation Act 1996": "https://www.indiacode.nic.in/bitstream/123456789/1628/1/199626.pdf",
    "Specific Relief Act 1963":         "https://www.indiacode.nic.in/bitstream/123456789/1368/1/196347.pdf",
    "Hindu Marriage Act 1955":          "https://www.indiacode.nic.in/bitstream/123456789/15697/1/the-hindu-marriage-act-1955.pdf",
    "Indian Evidence Act 1872":         "https://www.indiacode.nic.in/bitstream/123456789/2187/3/187209_3.pdf",
}


def is_real_pdf(filepath):
    with open(filepath, "rb") as f:
        return f.read(5).startswith(b"%PDF")


def slugify(name):
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def download(session, name, url):
    outfile = os.path.join(RAW_DIR, f"base_{slugify(name)}.pdf")
    if os.path.exists(outfile) and is_real_pdf(outfile):
        print(f"  SKIP  {name} (already exists)")
        return True

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://www.indiacode.nic.in/",
    }

    try:
        r = session.get(url, headers=headers, timeout=45, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        
        # If we get HTML, try scraping for a PDF link
        if "html" in content_type.lower():
            soup = BeautifulSoup(r.text, "html.parser")
            pdf_link = None
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    pdf_link = href if href.startswith("http") else url.rsplit("/", 1)[0] + "/" + href
                    break
            if not pdf_link:
                print(f"  FAIL  {name}: HTML page, no PDF link found")
                return False
            # Retry with direct PDF link
            r = session.get(pdf_link, headers=headers, timeout=45, stream=True)
            r.raise_for_status()

        with open(outfile, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        if is_real_pdf(outfile):
            size_kb = os.path.getsize(outfile) // 1024
            print(f"  OK    {name} ({size_kb} KB)")
            return True
        else:
            os.remove(outfile)
            print(f"  FAIL  {name}: Downloaded file is not a real PDF")
            return False

    except Exception as e:
        if os.path.exists(outfile):
            os.remove(outfile)
        print(f"  FAIL  {name}: {e}")
        return False


if __name__ == "__main__":
    s = requests.Session()
    # Warm up session with India Code homepage
    try:
        s.get("https://www.indiacode.nic.in/", timeout=10,
              headers={"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"})
    except Exception:
        pass

    success, fail = 0, 0
    for name, url in ACTS.items():
        time.sleep(1)  # Polite rate limiting
        ok = download(s, name, url)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"Downloaded: {success} | Failed: {fail}")
    
    # Show final real PDF count
    all_pdfs = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    real_pdfs = [f for f in all_pdfs if is_real_pdf(os.path.join(RAW_DIR, f))]
    print(f"Total real PDFs in data/raw: {len(real_pdfs)}")
