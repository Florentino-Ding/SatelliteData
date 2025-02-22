import re

# Define the regexes
CODE_PATTERN = re.compile(r"<H1>.+<\/H1>")
NORAD_PATTERN = re.compile(r"<B>NORAD ID<\/B>: [0-9]+ ")
INTL_CODE_PATTERN = re.compile(r"<B>Int'l Code<\/B>: [0-9A-Z\-]+ ")
PERIGEE_PATTERN = re.compile(r"<B>Perigee<\/B>: [0-9.]+ km ")
APOGEE_PATTERN = re.compile(r"<B>Apogee<\/B>: [0-9.]+ km ")
INCLINATION_PATTERN = re.compile(r"<B>Inclination<\/B>: [0-9.]+ &deg ")
PERIOD_PATTERN = re.compile(r"<B>Period<\/B>: [0-9.]+ minutes ")
SEMI_MAJOR_AXIS_PATTERN = re.compile(r"<B>Semi major axis<\/B>: [0-9]+ km ")
RCS_PATTERN = re.compile(r"<B>RCS<\/B>: [a-zA-Z0-9\ ]+ ")
LAUNCH_DATE_PATTERN = re.compile(
    r'<B>Launch date<\/B>: <a href="[a-zA-Z0-9\/\?=\&]+">[0-9a-zA-Z ,]+<\/a>'
)
SOURCE_PATTERN = re.compile(r"<B>Source<\/B>: [a-zA-Z \'\(\)]+")
LAUNCH_SITE_PATTERN = re.compile(r"<B>Launch site<\/B>: [a-zA-Z \(\)]+")
DECAY_DATE_PATTERN = re.compile(r"<B>Decay date<\/B>: [\d-]+")
TYPE_NAME_PATTERN = re.compile(r"<b>Note: This is [a ]?.+<\/b>")
TLE_PATTERN = re.compile(r"<pre>[\w .\n\-]+<\/pre>")


def parse_page(content: str) -> tuple:
    code = CODE_PATTERN.search(content)
    norad = NORAD_PATTERN.search(content)
    intl_code = INTL_CODE_PATTERN.search(content)
    perigee = PERIGEE_PATTERN.search(content)
    apogee = APOGEE_PATTERN.search(content)
    inclination = INCLINATION_PATTERN.search(content)
    period = PERIOD_PATTERN.search(content)
    semi_major_axis = SEMI_MAJOR_AXIS_PATTERN.search(content)
    rcs = RCS_PATTERN.search(content)
    launch_date = LAUNCH_DATE_PATTERN.search(content)
    source = SOURCE_PATTERN.search(content)
    launch_site = LAUNCH_SITE_PATTERN.search(content)
    decay_date = DECAY_DATE_PATTERN.search(content)
    type_name = TYPE_NAME_PATTERN.search(content)
    tle = TLE_PATTERN.search(content)

    # Process the regex results
    code = code.group(0).split("<H1>")[1].split("</H1>")[0] if code else None
    norad = norad.group(0).split(": ")[1] if norad else None
    intl_code = intl_code.group(0).split(": ")[1] if intl_code else None
    perigee = perigee.group(0).split(": ")[1].split(" ")[0] if perigee else None
    apogee = apogee.group(0).split(": ")[1].split(" ")[0] if apogee else None
    inclination = (
        inclination.group(0).split(": ")[1].split(" ")[0] if inclination else None
    )
    period = period.group(0).split(": ")[1].split(" ")[0] if period else None
    semi_major_axis = (
        semi_major_axis.group(0).split(": ")[1].split(" ")[0]
        if semi_major_axis
        else None
    )
    rcs = rcs.group(0).split(": ")[1] if rcs else None
    launch_date = (
        launch_date.group(0).split(": ")[1].split(">")[1].split("<")[0]
        if launch_date
        else None
    )
    source = source.group(0).split(": ")[1] if source else None
    launch_site = launch_site.group(0).split(": ")[1] if launch_site else None
    decay_date = decay_date.group(0).split(": ")[1] if decay_date else "ACTIVE"
    type_name = (
        type_name.group(0).split("This is ")[1].split("</b>")[0].split("a ")[-1]
        if type_name
        else "SATELLITE"
    )
    tle = tle.group(0).split("<pre>")[1].split("</pre>")[0] if tle else None

    return (
        norad,
        code,
        intl_code,
        perigee,
        apogee,
        inclination,
        period,
        semi_major_axis,
        rcs,
        launch_date,
        source,
        launch_site,
        decay_date,
        type_name,
        tle,
    )
