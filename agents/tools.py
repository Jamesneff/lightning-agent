import os

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
_SERPER_URL = "https://google.serper.dev/search"
_TIMEOUT = httpx.Timeout(total=20.0, connect=5.0, read=15.0, write=5.0)


def _serper_search(query: str, num_results: int = 5) -> list[dict]:
    """Shared Serper search helper."""
    response = httpx.post(
        _SERPER_URL,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": num_results},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("organic", [])


@tool
def get_funding_info(company_name: str) -> str:
    """Search for funding rounds, total raised, investors, and round stage for a company."""
    queries = [
        f"{company_name} funding raised seed round investors",
        f"{company_name} startup funding crunchbase pitchbook",
        f"{company_name} raises million venture capital",
    ]
    all_results = []
    seen_links = set()
    for query in queries:
        for r in _serper_search(query, num_results=3):
            link = r.get("link", "")
            if link not in seen_links:
                seen_links.add(link)
                all_results.append(r)

    if not all_results:
        return f"No funding information found for {company_name}."

    lines = [f"Funding search results for {company_name}:"]
    for r in all_results:
        lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')} [{r.get('link', '')}]")
    return "\n".join(lines)


@tool
def get_founder_info(company_name: str) -> str:
    """Search for founder names, roles, and backgrounds for a company."""
    queries = [
        f"{company_name} founder CEO co-founder",
        f"{company_name} founded by linkedin background",
        f"{company_name} team founder previously worked",
    ]
    all_results = []
    seen_links = set()
    for query in queries:
        for r in _serper_search(query, num_results=3):
            link = r.get("link", "")
            if link not in seen_links:
                seen_links.add(link)
                all_results.append(r)

    if not all_results:
        return f"No founder information found for {company_name}."

    lines = [f"Founder search results for {company_name}:"]
    for r in all_results:
        lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')} [{r.get('link', '')}]")
    return "\n".join(lines)


@tool
def get_company_overview(url: str) -> str:
    """Scrape a company homepage to extract product description and what they do."""
    try:
        response = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title else ""
        paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))[:1500]
        return f"Homepage: {url}\nTitle: {title}\nContent: {paragraphs}"
    except Exception as e:
        return f"Could not scrape {url}: {e}"


@tool
def get_recent_news(company_name: str) -> str:
    """Search for recent news, announcements, launches, and traction signals for a company."""
    results = _serper_search(f"{company_name} news announcement launch 2024 2025")
    if not results:
        return f"No recent news found for {company_name}."
    lines = [f"Recent news for {company_name}:"]
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')} [{r.get('link', '')}]")
    return "\n".join(lines)


@tool
def get_sec_filing(company_name: str) -> str:
    """Search SEC EDGAR for Form D filings which indicate early-stage fundraises."""
    results = _serper_search(f"{company_name} site:sec.gov form D", num_results=3)
    if not results:
        return f"No SEC Form D filing found for {company_name}."
    lines = [f"SEC filing results for {company_name}:"]
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')} [{r.get('link', '')}]")
    return "\n".join(lines)
