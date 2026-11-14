"""Render the compact repair report from recorded measurements."""
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted

HERE=Path(__file__).resolve().parent


def main():
    r=json.loads((HERE/'results/memory_recall_repair.json').read_text())
    groups={p:[x for x in r['language'] if x['memory_profile']==p] for p in ['addressed_v2','addressed']}
    means={p:{k:mean(x[k] for x in rows) for k in ['test_nll','bytes_per_second','training_seconds']} for p,rows in groups.items()}
    old,new=means['addressed_v2'],means['addressed']
    d0,d1=r['diagnostics']['addressed_v2'],r['diagnostics']['addressed']
    percent=(new['bytes_per_second']/old['bytes_per_second']-1)*100
    language=[['Seed','Previous v2 NLL','Repaired v3 NLL']]+[[str(seed),f"{next(x for x in groups['addressed_v2'] if x['seed']==seed)['test_nll']:.5f}",f"{next(x for x in groups['addressed'] if x['seed']==seed)['test_nll']:.5f}"] for seed in [7,17,29]]+[['Mean',f"{old['test_nll']:.5f}",f"{new['test_nll']:.5f}"]]
    norms=[['Repeated-input probe','Previous v2','Repaired v3'],['Peak matrix norm',f"{d0['max_memory_norm']:.3f}",f"{d1['max_memory_norm']:.3f}"],['Peak read norm',f"{d0['max_read_norm']:.3f}",f"{d1['max_read_norm']:.3f}"],['Cold clone retains binding','No','Yes']]
    saturation=[['Slow scale','Saturated coordinates: v2','Saturated coordinates: v3']]+[[str(a['level']),str(a['saturated_coordinates'])+'/64',str(b['saturated_coordinates'])+'/64'] for a,b in zip(d0['trained_slow_scales'],d1['trained_slow_scales'])]
    def md_table(rows):return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(row)+' |' for row in rows[1:]])
    sections=[
      ('Result', 'Two concrete memory defects are repaired: coherent writes no longer accumulate large read magnitudes, and cold inference no longer discards persistent associations. The end-to-end delayed AB/BA answer task remains at 50% accuracy. This is a partial repair, not a completed long-dependency solution.'),
      ('1. Repeated writes overwhelm the recalled signal', 'Version 2 sums key-value outer products and divides reads by the square root of accumulated key energy. Repeated, aligned writes add coherently; that energy normalizer does not keep their contribution at a constant scale. The fixed decays eventually limit growth, but at levels large enough to saturate downstream tanh states. A 10,000-event constant-input probe reproduces the amplification.'),
      ('2. Starting inference erases stored bindings', 'The native cold-copy path called reset(true), clearing the association store together with recurrent context. Therefore an explicit binding present in a saved or trained model disappeared from the normal evaluation/generation copy. Cold copies now preserve records and handles. clear_memory=True explicitly requests an empty store. The Python lifecycle repair applies to all three checkpoint backends; automatic neural admission remains frozen.'),
      ('Repair mechanics', 'At each slow event, read the existing matrix first. Decay it, predict the observed value from the fixed write key, and write only the residual with rate 0.5 / (1 + ||key||^2). The local backward pass includes this residual dependence exactly. Fixed sketches, rotary clocks, retention half-lives, learned queries and the eight-event training blocks are retained. The old energy normalizer is removed. No extra expert pass or all-history search is introduced.'),
      ('Remaining recall failure', 'Across seeds 7, 17 and 29, both versions remain at 50% on all tested gaps: 16, 48, 96 and 192 distractor bytes, with shifts 0-7. Training uses 6,000 examples, gaps 16/48/96 and shifts 0-3. Preserving a binding or keeping a matrix stable does not establish that the neural decoder learns the delayed answer. The unchanged acceptance test remains failed; neural episodic recall is still disabled by default.'),
      ('Validation and continuation', 'Passed 1,897 native checks, 14 Python tests, AddressSanitizer and UndefinedBehaviorSanitizer. Coverage includes independent numerical derivatives of the residual write, a repeated-content regression, read-before-write causality, channel interventions, retained/cleared bindings, and exact partial-block checkpoint continuation. LeakSanitizer was explicitly disabled on this ptrace-managed host. A checkpoint created before the repair reloads byte-for-byte through the preserved version-2 backend.'),
      ('Checkpoint policy', 'New models write native version 3. Version-1 and version-2 native sources remain unchanged in legacy/ and addressed_v2/. Old matrices are never silently reinterpreted as residual-write matrices. Existing checkpoints continue their original training arithmetic; a fresh version-3 model is required to use the new write rule. Both current and previous four-scale models contain 997,183 fitted scalars.'),
    ]
    language_text=f"Fresh models train on the same first 1,000,000 Tiny Shakespeare targets; the final contiguous 10% supplies 111,539 held-out targets. Execution order alternates by seed on CPU 0. Mean held-out NLL changes by {new['test_nll']-old['test_nll']:+.5f} nats/byte. Mean training throughput is {new['bytes_per_second']:,.0f} versus {old['bytes_per_second']:,.0f} bytes/s ({percent:+.1f}%). This repairs memory behavior with a small measured language-loss regression; it does not demonstrate a language-quality gain."
    diagnostic_text='The trained-state probe uses the same seed-7 delayed-cue training stream, then a warm AB + 192 x + ? prefix. A coordinate is counted as saturated when |state| > 0.95. Level-1 read norm falls from 621.54 to 2.14.'
    reproduce='python multiscale/tests/run_all.py --no-leak-check\npython -m multiscale.compare_recall_repair --data input.txt --out /tmp/recall\npython -m multiscale.build_recall_report'
    md='# SHoEbox memory and recall repair\n\n14 November 2026 | Base: `4e52bee` | Native policy: version 3\n\n'
    for title,body in sections[:2]:md+=f'## {title}\n\n{body}\n\n'
    md+=md_table(norms)+'\n\n'+diagnostic_text+'\n\n'+md_table(saturation)+'\n\n'
    for title,body in sections[2:5]:md+=f'## {title}\n\n{body}\n\n'
    md+='## Matched language results\n\n'+md_table(language)+'\n\n'+language_text+'\n\n'
    for title,body in sections[5:]:md+=f'## {title}\n\n{body}\n\n'
    md+='## Reproduction\n\n```bash\n'+reproduce+'\n```\n\n'
    md+='Recorded measurements: `results/memory_recall_repair.json`; verification: `results/memory_recall_verification.txt`. Native diagnostics remain in a separate executable, outside the production loop.\n\n'
    md+='Hardware: AMD EPYC 9V74; one worker pinned to CPU 0. Python '+r['python']+', NumPy '+r['numpy']+'. C++17, -O3, -march=native, -ffp-contract=off. Compilation, evaluation and checkpoint I/O are excluded from training timing. Dataset SHA-256: `'+groups['addressed'][0]['data_sha256']+'`.\n'
    (HERE/'MEMORY_RECALL_REPAIR.md').write_text(md)
    font_dir=Path('/usr/share/fonts/truetype/dejavu')
    for name,filename in [('ReportSans','DejaVuSans.ttf'),('ReportBold','DejaVuSans-Bold.ttf'),('ReportMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/filename)))
    styles={
      'title':ParagraphStyle('title',fontName='ReportBold',fontSize=21,leading=25,textColor=colors.HexColor('#15314a'),spaceAfter=8),
      'meta':ParagraphStyle('meta',fontName='ReportSans',fontSize=8,leading=11,textColor=colors.HexColor('#526777'),spaceAfter=13),
      'head':ParagraphStyle('head',fontName='ReportBold',fontSize=11,leading=14,textColor=colors.HexColor('#15314a'),spaceBefore=11,spaceAfter=5),
      'body':ParagraphStyle('body',fontName='ReportSans',fontSize=9,leading=12.6,spaceAfter=7),
      'small':ParagraphStyle('small',fontName='ReportSans',fontSize=8,leading=10.5,spaceAfter=7),
      'cell':ParagraphStyle('cell',fontName='ReportSans',fontSize=8,leading=10),
      'code':ParagraphStyle('code',fontName='ReportMono',fontSize=7.1,leading=10,spaceAfter=8),
    }
    story=[]
    def para(text,style='body'):story.append(Paragraph(escape(text),styles[style]))
    def section(i):para(sections[i][0],'head');para(sections[i][1])
    def table(rows,widths):
      cells=[[Paragraph(escape(x),styles['cell']) for x in row] for row in rows]
      t=Table(cells,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9eb4c5')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce5eb'))]));story.append(t);story.append(Spacer(1,7))
    para('SHoEbox memory and recall repair','title');para('14 November 2026  |  Base 4e52bee  |  Version-3 native policy','meta')
    section(0);section(1);table(norms,[248,130,130]);para(diagnostic_text,'small');table(saturation,[110,199,199]);section(2);section(3)
    story.append(PageBreak());para('Qualification and remaining work','title')
    section(4);para('Matched language results','head');table(language,[108,200,200]);para(language_text)
    section(5);section(6);para('Reproduction','head');story.append(Preformatted(reproduce,styles['code']))
    para('Measurements and source hashes: results/memory_recall_repair.json. Hardware: AMD EPYC 9V74, one worker on CPU 0. Python '+r['python']+', NumPy '+r['numpy']+'. C++17 / -O3 / -march=native / -ffp-contract=off. Training timings exclude compilation, evaluation and checkpoint I/O.','small')
    def footer(canvas,doc):
      canvas.setStrokeColor(colors.HexColor('#dce5eb'));canvas.line(43,36,551,36);canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#526777'));canvas.drawString(43,24,'SHoEbox | Memory and recall repair');canvas.drawRightString(551,24,str(doc.page))
    SimpleDocTemplate(str(HERE/'MEMORY_RECALL_REPAIR.pdf'),pagesize=(595.28,841.89),leftMargin=43,rightMargin=44.28,topMargin=35,bottomMargin=48,title='SHoEbox Memory and Recall Repair',author='SHoEbox').build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':main()
