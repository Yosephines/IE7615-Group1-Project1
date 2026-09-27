"""Check the local submission artifacts without training or modifying the submitted run."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import nbformat
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
for p in sorted((ROOT/'notebooks').glob('0*.ipynb')):
    n=nbformat.read(p,4);nbformat.validate(n)
    code=[c for c in n.cells if c.cell_type=='code']
    assert code and all(c.execution_count is not None for c in code),p
    assert not any(o.output_type=='error' for c in code for o in c.outputs),p
    print('Executed notebook:',p.name)
for name,pages in [('group1_proposal.pdf',1),('group1_training_results.pdf',3)]:
    r=PdfReader(ROOT/'output/pdf'/name)
    assert len(r.pages)==pages
    text='\n'.join(p.extract_text() for p in r.pages)
    assert 'Draft for team review' not in text and 'still need to be shared' not in text
    for accuracy in ['33.3%','40.0%','66.7%']:assert accuracy in text
    print('Report:',name,pages,'pages')
for row in json.loads((ROOT/'configs/original_contributions.json').read_text())['files']:
    assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256'],row['file']
assert (ROOT/'results/model_comparison.csv').read_bytes()==(ROOT/'results/group1_repro_v1/model_comparison.csv').read_bytes()
markdown=[ROOT/'README.md']
for folder in ['docs','configs','notebooks','models','data','logs','results']:
    markdown.extend((ROOT/folder).glob('*.md'))
for p in markdown:
    for raw in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        target=unquote(raw.split('#')[0])
        if not target or '://' in target or target.startswith('mailto:'):continue
        assert (p.parent/target).exists(),(p,raw)
print('Local documentation links and original source bytes verified.')
subprocess.run([sys.executable,str(ROOT/'scripts/verify_saved_run.py')],cwd=ROOT,check=True)
print('Submission artifact audit passed.')
