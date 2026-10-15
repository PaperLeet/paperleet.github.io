"""Bounded arXiv metadata retrieval; submitted links are never arbitrary fetch URLs."""
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET


MAX_METADATA_BYTES = 256 * 1024
_ID = re.compile(r"(?:[0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5}|[a-z][a-z0-9.-]{0,39}/[0-9]{2}(?:0[1-9]|1[0-2])[0-9]{3})(?:v[1-9][0-9]{0,3})?")
_ATOM = "{http://www.w3.org/2005/Atom}"


class SourceUnavailable(RuntimeError):
    """Retryable metadata failure, never a scientific rejection of the paper."""


def canonical_arxiv_url(value):
    if not isinstance(value, str) or len(value) > 512:
        raise ValueError("Enter an arXiv abstract or PDF link")
    value = value.strip()
    if not value or re.search(r"[\x00-\x20\x7f\\%?#]", value):
        raise ValueError("Enter an arXiv link without query parameters or fragments")
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"https", "http"} or parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        raise ValueError("Enter a valid arXiv abstract or PDF link")
    match = re.fullmatch(r"/(abs|pdf)/(.+)", parsed.path)
    identifier = match[2] if match else ""
    if match and match[1] == "pdf" and identifier.endswith(".pdf"):
        identifier = identifier[:-4]
    if not _ID.fullmatch(identifier):
        raise ValueError("Enter a valid arXiv abstract or PDF link")
    return "https://arxiv.org/abs/" + identifier


def arxiv_identity(value):
    """All versions, PDF/abstract links and aliases share the same duplicate key."""
    return re.sub(r"v[0-9]+$", "", canonical_arxiv_url(value).removeprefix("https://arxiv.org/abs/"))


def normalize_metadata(value, requested_url):
    """Validate a trusted loader's output before using it as the model's source data."""
    requested_url = canonical_arxiv_url(requested_url)
    if not isinstance(value, dict) or value.get("status", "ready") != "ready":
        raise SourceUnavailable("arXiv metadata is not available yet. Retry this proposal later.")
    url = canonical_arxiv_url(value.get("url", value.get("paper_url", "")))
    if arxiv_identity(url) != arxiv_identity(requested_url):
        raise SourceUnavailable("arXiv returned metadata for a different paper.")
    requested_version = re.search(r"v([0-9]+)$", requested_url)
    returned_version = re.search(r"v([0-9]+)$", url)
    version = int(returned_version[1]) if returned_version else value.get("version")
    if requested_version and version != int(requested_version[1]):
        raise SourceUnavailable("arXiv returned metadata for a different paper version.")
    def text(key, limit, required=False):
        item = value.get(key, "")
        if not isinstance(item, str) or (required and not item.strip()):
            raise SourceUnavailable("arXiv returned incomplete paper metadata.")
        return " ".join(item.split())[:limit]
    authors = value.get("authors", [])
    if not isinstance(authors, list) or any(not isinstance(author, str) for author in authors):
        raise SourceUnavailable("arXiv returned invalid author metadata.")
    result = {"arxiv_id": arxiv_identity(url), "url": requested_url, "paper_url": requested_url,
              "title": text("title", 2000, required=True), "abstract": text("abstract", 12000),
              "authors": [" ".join(author.split())[:300] for author in authors[:100] if author.strip()],
              "published": text("published", 80), "status": "ready"}
    if isinstance(version, int) and not isinstance(version, bool) and 1 <= version <= 9999:
        result["version"] = version
        result["resolved_url"] = "https://arxiv.org/abs/" + result["arxiv_id"] + "v" + str(version)
    else:
        result["resolved_url"] = url
    return result


def parse_arxiv_metadata(raw, requested_url):
    if not isinstance(raw, bytes) or len(raw) > MAX_METADATA_BYTES or b"<!" in raw or b"\x00" in raw:
        raise SourceUnavailable("arXiv returned invalid metadata.")
    try:
        feed = ET.fromstring(raw)
        entries = feed.findall(_ATOM + "entry")
        if feed.tag != _ATOM + "feed" or len(entries) != 1:
            raise ValueError("Expected one arXiv entry")
        entry = entries[0]
        def text(name):
            node = entry.find(_ATOM + name)
            return "" if node is None else " ".join("".join(node.itertext()).split())
        authors = []
        for author in entry.findall(_ATOM + "author")[:100]:
            name = author.find(_ATOM + "name")
            if name is not None:
                authors.append("".join(name.itertext()))
        return normalize_metadata({"url": text("id"), "title": text("title"), "abstract": text("summary"),
                                   "authors": authors, "published": text("published")}, requested_url)
    except (ET.ParseError, ValueError, TypeError):
        raise SourceUnavailable("arXiv returned invalid or mismatched paper metadata.") from None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_arxiv_metadata(value):
    """Fetch metadata only from the fixed official API, with no redirects or PDF execution."""
    url = canonical_arxiv_url(value)
    identifier = url.removeprefix("https://arxiv.org/abs/")
    endpoint = "https://export.arxiv.org/api/query?" + urlencode({"id_list": identifier, "max_results": 1})
    request = Request(endpoint, headers={"Accept": "application/atom+xml", "User-Agent": "PaperLeet-paper-harness/1.0"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=20) as response:
            if response.geturl() != endpoint:
                raise SourceUnavailable("arXiv metadata redirected unexpectedly.")
            raw = response.read(MAX_METADATA_BYTES + 1)
        return parse_arxiv_metadata(raw, url)
    except (HTTPError, URLError, TimeoutError, OSError):
        raise SourceUnavailable("arXiv metadata could not be fetched. Retry this proposal later.") from None
