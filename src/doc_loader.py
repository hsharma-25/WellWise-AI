import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


DATA_DIR = Path("data")

##describe well patterns using regex
WELL_PATTERN = re.compile(
    r"\b\d{1,2}/\d{1,2}-(?:"
    r"[A-Z]-\d+(?:\s+(?:[A-Z]|T\d+))?"
    r"|"
    r"\d+[A-Z]+(?:T\d+)?"
    r")\b",
    re.IGNORECASE,
)

ABBREVIATED_WELL_PATTERN = re.compile(
    r"\bF-(\d+)\s+([A-Z])\b",
    re.IGNORECASE,
)

##take the text from a report's first page and return the well identifiers if recognized
def extract_well_numbers(first_page_text):
    text = re.sub(r"(?<=-)\s+(?=\d)", "", first_page_text)
    text = re.sub(r"\s+", " ", text)

    well_numbers = set(WELL_PATTERN.findall(text))

    base_wells = [
        well for well in well_numbers
        if re.fullmatch(r"\d{1,2}/\d{1,2}-[A-Z]-\d+", well)
    ]

    for match in ABBREVIATED_WELL_PATTERN.finditer(text):
        number, sidetrack = match.groups()

        for base in base_wells:
            if base.endswith(f"-{number}"):
                well_numbers.add(f"{base} {sidetrack.upper()}")

    return sorted(well_numbers) if well_numbers else ["Unknown"]


docs = []

pdf_files = sorted(DATA_DIR.glob("*.PDF"))

for pdf_path in pdf_files:
    report_pages = PyMuPDFLoader(
        str(pdf_path),
        mode="page",
    ).load()

    if not report_pages:
        continue

    well_numbers = extract_well_numbers(
        report_pages[0].page_content
    )

    for doc in report_pages:
        doc.metadata["report_name"] = pdf_path.name
        doc.metadata["well_numbers"] = ", ".join(well_numbers)

    docs.extend(report_pages)