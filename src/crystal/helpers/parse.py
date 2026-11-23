# noqa: D100
def parse_text_in_between_headers(
    section: list[str], start_header: str, end_header: str
) -> list[str]:
    """Parses the text between specified start and end headers."""
    desired_lines = []
    in_section_of_interest = False
    for line in section:
        if start_header in line and not in_section_of_interest:
            in_section_of_interest = True
            continue
        if end_header in line and in_section_of_interest:
            break
        if in_section_of_interest:
            desired_lines.append(line)

    return desired_lines
