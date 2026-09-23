import os
import re
import glob
import pymupdf

def update_group1_cv(path):
    print(f"Processing Group 1: {path}")
    doc = pymupdf.open(path)
    page = doc[0]
    xref = page.get_contents()[0]
    raw = doc.xref_stream(xref).decode('latin1')
    
    # Check if already updated
    if 'alvarogarcia.me' in raw and 'LinkedIn' in raw:
        print(f"  Already updated: {path}")
        doc.close()
        return

    # Find the pattern for linkedin in Group 1
    # Pattern: /F41 9.9626 Tf [-33[34]\(link\)...Casas\)]TJ
    pattern = re.compile(r'/F41 9\.9626 Tf \[-33[34]\(link\).*?Casas\)]TJ')
    m = pattern.search(raw)
    if not m:
        print(f"  Pattern not found in {path}")
        doc.close()
        return
    
    old_linkedin = m.group(0)
    # Determine if space before is 333 or 334
    spacing = "-334" if "-334(link)" in old_linkedin else "-333"
    new_linkedin = f"/F41 9.9626 Tf [{spacing}(alvarogarcia.me)]TJ/F38 9.9626 Tf [-333(j)]TJ/F41 9.9626 Tf [{spacing}(LinkedIn)]TJ"
    
    # Also adjust the Td offset to center the line
    # Find the Td offset before email: -115.598 -14.79 Td (or similar)
    td_pattern = re.compile(r'(-1[0-9]{2}\.[0-9]+)\s+(-1[0-9]\.[0-9]+)\s+Td')
    td_match = td_pattern.search(raw[:m.start()])
    
    raw_mod = raw.replace(old_linkedin, new_linkedin)
    
    if td_match:
        old_td = td_match.group(0)
        old_x = float(td_match.group(1))
        # The line was ~388pt wide with old_x ~ -115.6. New line is ~345pt wide (diff ~ 42.5pt).
        # Shifting right by ~21.26pt centers the line: -115.598 + 21.26 = -94.338
        new_x = round(old_x + 21.26, 3)
        new_td = f"{new_x} {td_match.group(2)} Td"
        raw_mod = raw_mod.replace(old_td, new_td, 1)

    doc.update_stream(xref, raw_mod.encode('latin1'))
    
    # Save to temp and reopen to recreate links with accurate bounding boxes
    temp_path = path + ".tmp"
    doc.save(temp_path)
    doc.close()
    
    doc2 = pymupdf.open(temp_path)
    page2 = doc2[0]
    
    # Delete old linkedin links
    for l in page2.get_links():
        if 'linkedin' in l.get('uri', '').lower():
            page2.delete_link(l)
            
    # Add new links
    for w in page2.get_text('words'):
        if w[1] < 70:
            if w[4] == 'alvarogarcia.me':
                page2.insert_link({'kind': pymupdf.LINK_URI, 'from': pymupdf.Rect(w[0], w[1], w[2], w[3]), 'uri': 'https://alvarogarcia.me'})
            elif w[4] == 'LinkedIn':
                page2.insert_link({'kind': pymupdf.LINK_URI, 'from': pymupdf.Rect(w[0], w[1], w[2], w[3]), 'uri': 'https://www.linkedin.com/in/alvaro-garcia-casas/'})
                
    doc2.save(path)
    doc2.close()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"  Successfully updated Group 1: {path}")


def update_group2_cv(path):
    print(f"Processing Group 2: {path}")
    doc = pymupdf.open(path)
    page = doc[0]
    xref = page.get_contents()[0]
    raw = doc.xref_stream(xref).decode('latin1')
    
    # Check if already updated
    if 'alvarogarcia.me' in page.get_text():
        print(f"  Already updated: {path}")
        doc.close()
        return

    # Hex for linkedin in Group 2:
    # [<00480042004d0046>27<0032002f0042004d0058002b0051004b00660042004d0066001b00480070001c>28<006000510040003a001c>28<0060002b0042001c0040002a001c0062001c0062>]TJ
    pattern = re.compile(r'\[<00480042004d0046>.*?0062001c0062>]TJ')
    m = pattern.search(raw)
    if not m:
        print(f"  Pattern not found in {path}")
        doc.close()
        return

    hex_alvaro = '001c00480070001c00600051003b001c0060002b0042001c0058004b0032'
    hex_pipe = '0025'
    hex_linkedin = '00470042004d0046>27<0032002f0041004d'
    new_linkedin_block = f'[<{hex_alvaro}>-333<{hex_pipe}>-333<{hex_linkedin}>]TJ'

    # In Group 2, there is:
    # 1) The main line offset in block 1: /F3 9.96264 Tf -115.59432 -13.37257 Td
    # 2) The separate BT block for LinkedIn: BT /F5 9.96264 Tf 267.51577 23.52435 Td[...
    # Let's adjust both so the whole contact row is centered!
    raw_mod = raw[:m.start()] + new_linkedin_block + raw[m.end():]
    
    # Adjust main Td offset: -115.59432 -> -94.334
    td_pattern = re.compile(r'(-1[0-9]{2}\.[0-9]+)\s+(-1[0-9]\.[0-9]+)\s+Td')
    td_m = td_pattern.search(raw_mod)
    if td_m:
        old_td = td_m.group(0)
        old_x = float(td_m.group(1))
        new_x = round(old_x + 21.26, 3)
        raw_mod = raw_mod.replace(old_td, f"{new_x} {td_m.group(2)} Td", 1)
        
    # Adjust LinkedIn block Td X position:
    # Previously: 267.51577 (or 275.87798). Since the line start moved right by ~21.26, this Td also shifts by 21.26:
    bt_linkedin_pattern = re.compile(r'(BT\s+/F5\s+9\.96264\s+Tf\s+)([0-9]{3}\.[0-9]+)(\s+[0-9]+\.[0-9]+\s+Td\s*\[<001c00480070)')
    bt_m = bt_linkedin_pattern.search(raw_mod)
    if bt_m:
        old_bt_x = float(bt_m.group(2))
        new_bt_x = round(old_bt_x + 21.26, 5)
        raw_mod = raw_mod[:bt_m.start()] + f"{bt_m.group(1)}{new_bt_x}{bt_m.group(3)}" + raw_mod[bt_m.end():]

    doc.update_stream(xref, raw_mod.encode('latin1'))

    temp_path = path + ".tmp"
    doc.save(temp_path)
    doc.close()

    doc2 = pymupdf.open(temp_path)
    page2 = doc2[0]

    for l in page2.get_links():
        if 'linkedin' in l.get('uri', '').lower():
            page2.delete_link(l)

    for w in page2.get_text('words'):
        if w[1] < 70:
            if w[4] == 'alvarogarcia.me':
                page2.insert_link({'kind': pymupdf.LINK_URI, 'from': pymupdf.Rect(w[0], w[1], w[2], w[3]), 'uri': 'https://alvarogarcia.me'})
            elif w[4] == 'LinkedIn':
                page2.insert_link({'kind': pymupdf.LINK_URI, 'from': pymupdf.Rect(w[0], w[1], w[2], w[3]), 'uri': 'https://www.linkedin.com/in/alvaro-garcia-casas/'})

    doc2.save(path)
    doc2.close()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"  Successfully updated Group 2: {path}")


def update_tex_files():
    print("Updating LaTeX source files...")
    tex_files = glob.glob('Proyectos/CVs/*.tex')
    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Replace href to linkedin with website + linkedin
        old_pattern = re.compile(r'\\href\{https://www\.linkedin\.com/in/alvaro-garcia-casas/\}\{linkedin\.com/in/Alvaro-Garcia-Casas\}')
        if old_pattern.search(content):
            new_replacement = (
                r'\\href{https://alvarogarcia.me}{alvarogarcia.me} \\textbar{} \n'
                r'  \\href{https://www.linkedin.com/in/alvaro-garcia-casas/}{LinkedIn}'
            )
            updated_content = old_pattern.sub(new_replacement, content)
            with open(tf, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"  Updated .tex: {tf}")
        else:
            print(f"  Pattern not found or already updated in: {tf}")


if __name__ == '__main__':
    all_pdfs = sorted(glob.glob('Proyectos/CVs/*.pdf'))
    for pdf in all_pdfs:
        doc = pymupdf.open(pdf)
        raw = doc.xref_stream(doc[0].get_contents()[0]).decode('latin1', errors='ignore')
        doc.close()
        if '00480042004d0046' in raw:
            update_group2_cv(pdf)
        else:
            update_group1_cv(pdf)
            
    update_tex_files()
