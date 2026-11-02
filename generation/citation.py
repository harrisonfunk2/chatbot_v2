import re


def verify_citations(answer, citation_map):
    """Check that cited source numbers exist in the citation map."""

    citation_numbers = sorted({
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer) # ex. [1], [2], etc
    })

    valid_citations = [
        number for number in citation_numbers
        if number in citation_map
    ]

    invalid_citations = [
        number for number in citation_numbers
        if number not in citation_map
    ]

    return valid_citations, invalid_citations