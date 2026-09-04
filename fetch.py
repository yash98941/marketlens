"""Fetching article text from a URL.

The original version of this project passed whatever URL the user typed
straight to requests. That is a server side request forgery hole: on a hosted
box someone can enter http://169.254.169.254/latest/meta-data/ and read the
cloud metadata service, or http://127.0.0.1:5432 to poke at internal
services, and the server happily fetches it and shows them the result.

So every URL goes through check_url first. It has to be http or https, the
hostname has to resolve to a public address, and redirects are followed one
hop at a time so a public URL cannot bounce to a private one.
"""

import ipaddress
import re
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

TIMEOUT = 10
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
HEADERS = {"User-Agent": "marketlens/1.0"}

DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]

# Wikipedia and similar sites leave footnote markers like [12] in the text.
# They carry no meaning and get picked up as sentences of their own.
FOOTNOTE = re.compile(r"\[\s*(?:\d+|citation needed|edit|note \d+)\s*\]", re.I)


def is_public(host):
    """True only if every address this host resolves to is on the public net."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        # Covers loopback, link local (cloud metadata), and the private ranges.
        if (address.is_private or address.is_loopback or address.is_reserved
                or address.is_link_local or address.is_multicast
                or address.is_unspecified):
            return False
    return bool(infos)


def check_url(url):
    """Raises if the URL is not safe to fetch. Returns the parsed URL."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http and https are allowed, got: " + str(parsed.scheme))
    if not parsed.hostname:
        raise ValueError("no hostname in URL")
    if not is_public(parsed.hostname):
        raise ValueError("refusing to fetch a private or local address: " + parsed.hostname)

    return parsed


def fetch(url):
    """Downloads a page, checking safety again after every redirect."""
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        response = requests.get(url, timeout=TIMEOUT, headers=HEADERS,
                                allow_redirects=False, stream=True)

        if response.is_redirect or response.is_permanent_redirect:
            url = requests.compat.urljoin(url, response.headers["location"])
            continue

        response.raise_for_status()

        kind = response.headers.get("content-type", "")
        if "html" not in kind and "text" not in kind:
            raise ValueError("not a text page: " + kind)

        # Read a bounded amount so a huge file cannot exhaust memory.
        body = response.raw.read(MAX_BYTES, decode_content=True)
        return body.decode(response.encoding or "utf-8", errors="replace")

    raise ValueError("too many redirects")


def clean_paragraph(text):
    """Drops footnote markers and tidies the spacing they leave behind."""
    text = FOOTNOTE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text(html):
    """Pulls the readable part out of a page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    # Paragraphs give cleaner text than get_text over the whole tree, which
    # tends to glue menu items onto the first sentence.
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [clean_paragraph(p) for p in paragraphs]
    paragraphs = [p for p in paragraphs if len(p.split()) > 8]

    if not paragraphs:
        paragraphs = [clean_paragraph(soup.get_text(" ", strip=True))]

    return title, "\n\n".join(paragraphs)


def load_article(url):
    """Returns a dict with the url, title and body text."""
    title, text = extract_text(fetch(url))
    return {"url": url, "title": title, "text": text}
