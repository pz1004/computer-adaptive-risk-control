import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
import os
import re
from difflib import SequenceMatcher

def normalize_string(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r'[{}\\]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()

def get_parser():
    parser = BibTexParser(common_strings=True)
    parser.customization = convert_to_unicode
    return parser

def load_bib_file(path):
    with open(path, 'r') as f:
        content = f.read()
    # Hack to fix unquoted months
    for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'june', 'jul', 'july', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec']:
        content = re.sub(rf'(?i)month\s*=\s*{m}\s*,', f'month = "{{{m.capitalize()}}}",', content)
    return bibtexparser.loads(content, parser=get_parser())

main_db = load_bib_file('main.bib')

print("Comparison Results:")
print("="*40)

for entry in main_db.entries:
    key = entry['ID']
    bib_path = f"paper/bibitems/{key}.bib"
    
    if not os.path.exists(bib_path):
        print(f"[{key}] - Could not find fetched bib file.")
        continue
        
    try:
        fetched_db = load_bib_file(bib_path)
        if not fetched_db.entries:
            print(f"[{key}] - Fetched bib file is empty.")
            continue
        fetched_entry = fetched_db.entries[0]
    except Exception as e:
        print(f"[{key}] - Error parsing fetched bib file: {e}")
        continue
        
    modifications_required = False
    diffs = []
    
    fields_to_check = ['title', 'author', 'journal', 'booktitle', 'year', 'volume', 'number', 'pages', 'doi', 'url']
    
    for field in fields_to_check:
        main_val = entry.get(field)
        fetch_val = fetched_entry.get(field)
        
        if main_val and fetch_val:
            if field == 'author':
                norm_main = normalize_string(main_val)
                norm_fetch = normalize_string(fetch_val)
                if SequenceMatcher(None, norm_main, norm_fetch).ratio() < 0.6:
                    modifications_required = True
                    diffs.append(f"  Field '{field}' differs significantly:\n    main.bib: {main_val}\n    fetched : {fetch_val}")
            else:
                norm_main = normalize_string(main_val)
                norm_fetch = normalize_string(fetch_val)
                norm_main = norm_main.replace('--', '-').replace('–', '-')
                norm_fetch = norm_fetch.replace('--', '-').replace('–', '-')
                norm_main = norm_main.replace('http://', 'https://')
                norm_fetch = norm_fetch.replace('http://', 'https://')
                norm_main = norm_main.replace('dx.doi.org', 'doi.org')
                norm_fetch = norm_fetch.replace('dx.doi.org', 'doi.org')

                if norm_main != norm_fetch:
                    modifications_required = True
                    diffs.append(f"  Field '{field}' differs:\n    main.bib: {main_val}\n    fetched : {fetch_val}")
        elif fetch_val and not main_val:
            modifications_required = True
            diffs.append(f"  Field '{field}' missing in main.bib, but present in fetched: {fetch_val}")
    
    if modifications_required:
        print(f"[{key}] requires modifications:")
        for diff in diffs:
            print(diff)
    else:
        print(f"[{key}] is correct.")
    print("-" * 40)
