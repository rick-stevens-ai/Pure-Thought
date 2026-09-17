#!/usr/bin/env python3
"""Build synchronized reading copies and self-contained LaTeX from reviewed sources.

Requires Pandoc; --pdf also requires XeLaTeX. No research computations are run.
Use --check to detect generated-text drift without modifying tracked files.
"""
from pathlib import Path
import argparse
import concurrent.futures
import json
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-09-17'
UPSTREAM = 'https://github.com/rick-stevens-ai/Pure-Thought/blob/main/'
CATALOG = json.loads((ROOT / 'reviews/catalog.json').read_text())
FORMULATION_PATH = ROOT / 'reviews/MATHEMATICAL-FORMULATIONS.md'
parts = re.split(r'^## (\d\d|S\d\d)\s*$', FORMULATION_PATH.read_text(), flags=re.M)
FORMULATIONS = dict(zip(parts[1::2], (x.strip() for x in parts[2::2])))
SOURCES = {}
for group, folder in [('canonical', 'PRDs'), ('supplemental', 'supplemental')]:
    for entry in CATALOG[group]:
        key = f"{entry['n']:02d}" if group == 'canonical' else entry['id']
        SOURCES[key] = (entry, ROOT / folder / entry['file'])
assert set(SOURCES) == set(FORMULATIONS)


def read(rel):
    return (ROOT / rel).read_text()


def section(text, heading):
    return re.split(r'^## ', text.split('## ' + heading + '\n', 1)[1], maxsplit=1, flags=re.M)[0].strip()


def rebase(text, source, destination, web=False):
    def link(m):
        target = m[1]
        if re.match(r'\w+://', target) or target.startswith('#'):
            return m[0]
        path, _, fragment = target.partition('#')
        absolute = (source.parent / path).resolve()
        if web:
            result = UPSTREAM + absolute.relative_to(ROOT).as_posix()
        else:
            import os
            result = Path(os.path.relpath(absolute, destination.parent)).as_posix()
        return '](' + result + ('#' + fragment if fragment else '') + ')'
    return re.sub(r'\]\(([^\s)]+)\)', link, text)


def document(key, destination, web=False, critique=True):
    entry, path = SOURCES[key]
    text = rebase(path.read_text(), path, destination, web)
    form = rebase(FORMULATIONS[key], FORMULATION_PATH, destination, web)
    text += '\n\n## Displayed formulation and claim boundary\n\n' + form + '\n'
    if key.isdigit() and critique:
        cp = ROOT / 'reviews' / entry['file']
        c = rebase(cp.read_text(), cp, destination, web)
        c = c.split('\n', 1)[1]
        c = re.sub(r'^## ', '### ', c, flags=re.M)
        text += '\n## Detailed review and rationale\n' + c
    return text


def summary():
    dest = ROOT / 'COMPLETE-PROBLEM-SUMMARIES.md'
    text = '# Complete problem summaries and mathematical formulations\n\n'
    text += f'Updated {DATE}. All 30 canonical problems and all 12 supplemental topics are included in full, with plain-language guides, assumptions, mathematical targets, outputs and validation gates. The displayed formulations make the report equations and claim directions explicit. These are proposed research tasks, not completed research results.\n\n'
    text += '[Review and critiques](reviews/COMPLETE-REVIEW.md) · [Evidence standard](reviews/EVIDENCE-STANDARD.md) · [Synthesis crosswalk](reviews/SYNTHESIS-CROSSWALK.md) · [PDF reading copy](latex-reports/Complete-Problem-Summaries.pdf)\n\n## Contents\n\n'
    for key, (entry, _) in SOURCES.items():
        text += f"- [{key}. {entry['title']}](#problem-{key.lower()})\n"
    for key in SOURCES:
        text += f'\n\n---\n\n<a id="problem-{key.lower()}"></a>\n\n' + document(key, dest, critique=False)
    return text


def complete_review():
    # Keep the complete canonical texts intact, as required by validate_revision.py.
    text = '# Complete revised problems and detailed critiques\n\n'
    text += f'Updated {DATE}. This reading copy includes all 30 canonical specifications and critiques, 12 supplemental topics, and the displayed mathematical formulations used by the reports. [Overview](REVIEW.md) · [Evidence standard](EVIDENCE-STANDARD.md) · [Complete summaries](../COMPLETE-PROBLEM-SUMMARIES.md)\n\n## Contents\n\n'
    for key, (entry, _) in SOURCES.items():
        text += f"- [{key}. {entry['title']}](#problem-{key.lower()})\n"
    for key, (entry, path) in SOURCES.items():
        text += f'\n\n---\n\n<a id="problem-{key.lower()}"></a>\n\n' + path.read_text()
        text += '\n\n## Displayed formulation and claim boundary\n\n' + FORMULATIONS[key] + '\n'
        if key.isdigit():
            text += '\n\n' + read('reviews/' + entry['file'])
    return text


def synthesis():
    dest = ROOT / 'NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md'
    text = '# Thirty synthesis challenges: revised problem summaries\n\n'
    text += f'Updated {DATE}. This synthesis retains the original synthesis numbering. It is a different portfolio from the canonical 30 PRDs. Every topic is mapped to its revised destination; the 12 distinct supplemental objectives are preserved. Claims and milestones follow the updated review and evidence standard.\n\n'
    text += '[All 42 complete summaries](../COMPLETE-PROBLEM-SUMMARIES.md) · [Full crosswalk](../reviews/SYNTHESIS-CROSSWALK.md) · [Revised synthesis PDF](30-Pure-Thought-Challenges-Report.pdf) · [Preserved original](../archive/original-synthesis/30-Most-Compelling-Pure-Thought-AI-Challenges.md)\n\n'
    rows = re.findall(r'^\| (\d\d) \| (.*?) \| (.*?) \| (.*?) \|$', read('reviews/SYNTHESIS-CROSSWALK.md'), re.M)
    assert len(rows) == 30
    for num, title, targets, treatment in rows:
        text += f'\n# Synthesis {num}. {title}\n\n**Mapping:** {targets}. {treatment}.\n\n'
        keys = re.findall(r'\[(?:PRD )?(\d\d|S\d\d)\]', targets)
        assert keys
        for key in keys:
            entry, path = SOURCES[key]
            source = path.read_text()
            text += f"## {key}. {entry['title']}\n\n"
            question_heading = 'Core question' if key.isdigit() else 'Revised core question'
            text += '**Question.** ' + section(source, question_heading) + '\n\n'
            guide = section(source, 'Plain-language guide')
            motivation = guide.split('### Why this matters\n', 1)[1].split('### What progress would mean', 1)[0].strip()
            text += '**Why it matters.** ' + motivation + '\n\n'
            scope_heading = 'Scope and assumptions' if key.isdigit() else 'Precise scope, target and outputs'
            text += '**Scope and model.** ' + rebase(section(source, scope_heading), path, dest) + '\n\n'
            text += '### Mathematical formulation and claim boundary\n\n' + rebase(FORMULATIONS[key], FORMULATION_PATH, dest) + '\n\n'
            if key.isdigit():
                text += '**Deliverables.** ' + section(source, 'Required outputs and proof obligations') + '\n\n'
                text += '**Validation.** ' + section(source, 'Validation and rejection controls') + '\n\n'
                milestones = section(source, 'Milestones and research extension')
                text += '**Milestones.** ' + milestones.split('\n\nMilestones are')[0] + '\n\n'
                text += f"[Detailed critique](../reviews/{entry['file']}).\n\n"
            else:
                text += '**Critique.** ' + section(source, 'Detailed critique of the synthesis framing') + '\n\n'
                text += '**Validation and milestones.** ' + rebase(section(source, 'Validation and milestones'), path, dest) + '\n\n'
    return text


# Unicode mathematical notation in existing prose is converted to actual TeX glyphs.
# Superscripts/subscripts are grouped; display mathematics retains its operators.
SUPER = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵀ', '0123456789-T'))
SUB = dict(zip('₀₁₂₃₄₅₆₇₈₉₋ₕₖᵢⱼ', '0123456789-hkij'))
MATH = {'−': '-', '×': r'\times', '·': r'\cdot', '±':r'\pm',
        '†':r'\dagger', '′':r'\prime', 'ℏ':r'\hbar', 'ℓ':r'\ell',
        '→':r'\to', '↔':r'\leftrightarrow', '↦':r'\mapsto', '⇌':r'\rightleftharpoons',
        '∂':r'\partial', '∅':r'\varnothing', '∇':r'\nabla', '∈':r'\in',
        '∖':r'\setminus', '∘':r'\circ', '√':r'\surd', '∞':r'\infty',
        '∫':r'\int', '∮':r'\oint', '≈':r'\approx', '≠':r'\ne',
        '≤':r'\le', '≥':r'\ge', '⊂':r'\subset', '⊗':r'\otimes',
        '⊥':r'\perp', '〈':r'\langle', '〉':r'\rangle'}
for char in 'ΓΔΛΣΦΩαβγδεηθκλμνπρστφχψω':
    name = unicodedata.name(char).split()[-1].lower().replace('lamda', 'lambda')
    MATH[char] = '\\' + (name.capitalize() if char.isupper() else name)
ESCAPE = {'\\':r'\textbackslash{}', '&':r'\&', '%':r'\%', '$':r'\$', '#':r'\#',
          '_':r'\_', '{':r'\{', '}':r'\}', '~':r'\textasciitilde{}', '^':r'\textasciicircum{}'}


def notation(s, math=False):
    out = ''
    i = 0
    while i < len(s):
        if s[i:].startswith('⁻¹ᐟ²'):
            out += '^{-1/2}' if math else r'\ensuremath{{}^{-1/2}}'
            i += 4
            continue
        c = s[i]
        if c in SUPER or c in SUB:
            table, operator = (SUPER, '^') if c in SUPER else (SUB, '_')
            j = i
            while j < len(s) and s[j] in table:
                j += 1
            val = operator + '{' + ''.join(table[x] for x in s[i:j]) + '}'
            out += val if math else '\\ensuremath{{}' + val + '}'
            i = j
            continue
        if i + 1 < len(s) and s[i+1] in '\u0302\u0304':
            val = ('\\hat' if s[i+1] == '\u0302' else '\\bar') + '{' + MATH.get(c, c) + '}'
            out += val if math else '\\ensuremath{' + val + '}'
            i += 2
            continue
        if c in MATH:
            out += MATH[c] + ' ' if math else '\\ensuremath{' + MATH[c] + '}'
        elif not math:
            out += ESCAPE.get(c, c)
        else:
            out += c
        i += 1
    return out


def transform(node):
    if isinstance(node, list):
        return [transform(x) for x in node]
    if not isinstance(node, dict):
        return node
    if node.get('t') == 'Math':
        node['c'][1] = notation(node['c'][1], math=True)
        return node
    if node.get('t') == 'Str' and any(c in MATH or c in SUPER or c in SUB or c in '\u0302\u0304' for c in node['c']):
        return {'t':'RawInline', 'c':['latex', notation(node['c'])]}
    # The reading copy uses explicit anchors; LaTeX creates its own bookmarks.
    if node.get('t') in ('RawBlock', 'RawInline') and node['c'][0] == 'html':
        return {'t':node['t'], 'c':['latex', '']}
    if node.get('t') == 'Header':
        node['c'][1][0] = ''  # avoid duplicate labels in combined documents
    return {k: transform(v) for k, v in node.items()}


HEADER = r'''\usepackage{amsmath,amssymb}
\usepackage{xurl}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Pure Thought | Revised research specifications}
\fancyfoot[L]{\small 2026-09-17}
\fancyfoot[R]{\small\thepage}
\setlength{\headheight}{15pt}
\setlength{\emergencystretch}{3em}
\allowdisplaybreaks
'''


def latex(text, title, destination, combined=False):
    text = rebase(text, destination, destination, web=True)
    data = subprocess.check_output(['pandoc', '-f', 'markdown+tex_math_dollars', '-t', 'json'], input=text.encode())
    ast = transform(json.loads(data))
    meta = ast['meta']
    meta['title'] = {'t':'MetaString','c':title}
    meta['author'] = {'t':'MetaString','c':'Pure Thought research portfolio'}
    meta['date'] = {'t':'MetaString','c':'Revised 17 September 2026'}
    meta['header-includes'] = {'t':'MetaBlocks','c':[{'t':'RawBlock','c':['latex',HEADER]}]}
    cmd = ['pandoc', '-f', 'json', '-t', 'latex', '--standalone', '--top-level-division=section',
           '-V', 'documentclass=article', '-V', 'fontsize=11pt', '-V', 'geometry:margin=1in',
           '-V', 'colorlinks=true', '-V', 'urlcolor=blue', '-V', 'linkcolor=blue']
    if combined:
        cmd += ['--toc', '--toc-depth=1']
    result = subprocess.check_output(cmd, input=json.dumps(ast).encode()).decode()
    result = result.replace('\\maketitle', '\\maketitle\n\\thispagestyle{empty}\n\\clearpage')
    if combined:
        result = result.replace('\\tableofcontents', '\\tableofcontents\n\\clearpage')
    return '% Generated by scripts/build_reports.py; edit reviewed Markdown inputs.\n% Compile with XeLaTeX.\n' + result


def generated_texts():
    out = {}
    out[ROOT / 'COMPLETE-PROBLEM-SUMMARIES.md'] = summary()
    out[ROOT / 'reviews/COMPLETE-REVIEW.md'] = complete_review()
    syn = ROOT / 'NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md'
    out[syn] = synthesis()
    for key, (entry, path) in SOURCES.items():
        if key.isdigit():
            matches = list((ROOT / 'latex-reports').glob(key + '-*-Report.tex'))
            assert len(matches) == 1, (key, matches)
            dest = matches[0]
        else:
            dest = ROOT / 'latex-reports' / (key + '-Supplemental-Report.tex')
        content = document(key, dest, web=True)
        evidence = rebase(read('reviews/EVIDENCE-STANDARD.md'), ROOT / 'reviews/EVIDENCE-STANDARD.md', dest, web=True)
        content += '\n\n' + evidence
        out[dest] = latex(content, key + '. ' + entry['title'], dest)
    dest = ROOT / 'latex-reports/Complete-Problem-Summaries.tex'
    body = rebase(out[ROOT / 'COMPLETE-PROBLEM-SUMMARIES.md'], ROOT / 'COMPLETE-PROBLEM-SUMMARIES.md', dest, web=True)
    # The PDF table of contents replaces the Markdown navigation list.
    body = re.sub(r'## Contents\n.*?(?=\n---)', '', body, count=1, flags=re.S)
    body += '\n\n' + rebase(read('reviews/EVIDENCE-STANDARD.md'), ROOT / 'reviews/EVIDENCE-STANDARD.md', dest, web=True)
    out[dest] = latex(body, 'Complete problem summaries and mathematical formulations', dest, combined=True)
    dest = ROOT / 'NOTES/30-Pure-Thought-Challenges-Report.tex'
    body = rebase(out[syn], syn, dest, web=True)
    body += '\n\n' + rebase(read('reviews/EVIDENCE-STANDARD.md'), ROOT / 'reviews/EVIDENCE-STANDARD.md', dest, web=True)
    out[dest] = latex(body, 'Thirty synthesis challenges: revised summaries', dest, combined=True)
    return {path: text.rstrip() + '\n' for path, text in out.items()}


def compile_pdf(path):
    build = ROOT / 'work/report-build' / path.stem
    build.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        proc = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                               '-output-directory=' + str(build), str(path)], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (build / 'compiler-output.txt').write_bytes(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f'{path.name}: see {build}/compiler-output.txt\n' + proc.stdout.decode()[-2500:])
    log = (build / (path.stem + '.log')).read_text(errors='replace')
    if any(issue in log for issue in ['Missing character:', 'Overfull \\hbox', 'Overfull \\vbox']):
        raise RuntimeError(f'Typesetting defect in {path.name}: see build log')
    path.with_suffix('.pdf').write_bytes((build / path.with_suffix('.pdf').name).read_bytes())
    return path.name


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--only', help='Compile only TeX basenames with this prefix')
    args = parser.parse_args()
    generated = generated_texts()
    changed = []
    for path, data in generated.items():
        if not path.exists() or path.read_text() != data:
            changed.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(data)
    if args.check:
        if changed:
            raise SystemExit('Generated text drift:\n' + '\n'.join(changed))
        print(f'{len(generated)} generated text artifacts match their reviewed sources.')
        return
    print(f'Generated {len(generated)} text artifacts; changed {len(changed)}.', flush=True)
    if args.pdf:
        paths = [p for p in generated if p.suffix == '.tex' and (not args.only or p.name.startswith(args.only))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for result in executor.map(compile_pdf, paths):
                print('Built ' + result.replace('.tex', '.pdf'), flush=True)


if __name__ == '__main__':
    main()
