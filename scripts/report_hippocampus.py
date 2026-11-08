"""Write ordinary downloadable Markdown/PDF from the audited hippocampus results."""
import argparse
from html import escape
import json
from pathlib import Path
import re
import reportlab
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether,PageBreak


def pct(value): return f'{100*value:.1f}%'
def pair(value): return f"{value['both_correct']}/{value['conflicting_pairs']} ({pct(value['both_accuracy'])})"
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def write_report(data,out):
    arms=data['arms']; labels={'parent':'Retained parent','A':'A - paired control','B':'B - local anchors','C':'C - shared hippocampus'}
    measure=lambda arm:arms[arm]['measurements']
    passed=arms['C']['promoted']
    conclusion=('The shared-memory arm passes the declared validation promotion gates.' if passed else
        'The shared-memory arm does not pass the declared validation promotion gates. Keep the stronger retention parent as the working model; retain the implemented extension behind its switch.')
    text=f'''# SHoEbox hippocampus implementation

November 8, 2026 (UTC) | Completed implementation and controlled pilot

The explicit hippocampus is implemented: one shared sparse associative service,
owner-local capsules, a dedicated cue lane, conditional regional readers and
causal next-round commits. The extension defaults off. {conclusion}

## Fixed-budget results

All trained arms use the same strong parent, 64 updates, source schedule, AdamW
settings and task/language objectives. The unchanged control's completed endpoint
was reused after {data['control_reuse']['reproduced_updates']} updates reproduced its losses, gradient norms and
source ledgers exactly. Final cohorts are new: seed 1201, document start 80000.
There are 192 conflicting histories per paired cohort; both questions must be
correct about the same unchanged history.

'''
    text+=table(['Model','Unique parameters','Both +0 bytes','Both +32','Both +64'],
        [[labels[a],f"{arms[a]['unique_parameters']:,}",*[pair(measure(a)[f'pairs_{g}']) for g in (0,32,64)]] for a in arms])+'\n\n'
    text+=table(['Model','Single +64','Correction +64','Natural-text NLL'],
        [[labels[a],pct(measure(a)['single_64']['accuracy']),pct(measure(a)['correction_64']['accuracy']),f"{measure(a)['language']['nll']:.6f}"] for a in arms])+'\n\n'
    text+='Single-fact tests use 32 templates; two-fact/correction tests use 16. Natural-text NLL uses 1,024 bytes. Eight documents per ordinary and long-delay stream are included in the machine-readable results.\n\n'
    c=measure('C')
    text+=f"Changing only the query switches C's answer in {c['pairs_0']['query_switches']}/192, {c['pairs_32']['query_switches']}/192 and {c['pairs_64']['query_switches']}/192 conflicting histories at +0/+32/+64. Source-value interventions and each query direction are reported separately below.\n\n"
    text+=table(['Gap','Older correct','Newer correct','Older source switches','Newer source switches'],[
        [g,pct(c[f'pairs_{g}']['older_accuracy']),pct(c[f'pairs_{g}']['newer_accuracy']),
         f"{c[f'pairs_{g}']['source_sensitivity']['older']['source_switches']}/{c[f'pairs_{g}']['source_sensitivity']['older']['groups']}",
         f"{c[f'pairs_{g}']['source_sensitivity']['newer']['source_switches']}/{c[f'pairs_{g}']['source_sensitivity']['newer']['groups']}"] for g in (0,32,64)])+'\n\n'
    text+='The shared arm still predominantly answers with the newer value. The outstanding behavioral exit is reliable query-dependent selection between the retained facts.\n\n'
    text+='## Validation and promotion\n\nValidation uses seed 683/start 3000 and eight templates per cohort. The primary comparison is the fixed update-64 endpoint. B/C validation-selected checkpoints are retained separately.\n\n'
    text+=table(['Arm','Selected update','Selected paired NLL','Pair gate','Retention gate','Correction gate','Language gate'],[
        [a,arms[a]['selected_update'],f"{arms[a]['selected_validation_score']:.6f}",
         *['Pass' if arms[a]['promotion_gates'][k] else 'Fail' for k in ('paired','single','correction','language')]] for a in ('B','C')])+'\n\n'
    text+='Gate columns describe update 64, separately from the validation-selected candidate. Promotion requires at least 50% both-correct validation pairs at gaps 0 and 32, no more than a 2-point single-fact decrease, no more than a 3-point correction decrease, and at most +0.05 natural-text NLL versus the parent on matching validation cohorts. Control A uses its verified fixed endpoint; its intermediate validation selection is not compared.\n\n'
    text+=table(['Arm','Extra state capacity','Retrieval execution'],[
        ['A','None','Original local publication recall.'],
        ['B','64 capsules per region; no global bindings','Dedicated cues and conditional reads within the requesting region.'],
        ['C','64 capsules per region plus 2 x 128 shared bindings','Bounded global directory/postings and targeted owner reads.']])+'\n\n'
    text+='B and C instantiate the same optional parameter schema; B bypasses shared association routing. Unique parameter count therefore matches, while their allocated fast state and executed service work differ.\n\n'
    text+='## What was implemented\n\n'
    text+=table(['Component','Behavior'],[
        ['Regional interface','A shared compact trunk with one independent input-driven cue trace per region; no candidate pooling in the initial cue.'],
        ['Owner anchors','64 fixed-width capsules per region, multiplexing its nodes; generation/version/namespace validation and immutable snapshots.'],
        ['DG / CA3','8-of-256 sparse separation, binary shard directory, bounded postings, associative links and at most two completion steps.'],
        ['Conditional readers / CA1','Owner-local capsule/cue interaction; distinct returned roles; learned accept/null/refine and gated expert-input reinstatement.'],
        ['Admission / revision / feedback','Separate write/erase policies, bounded replacement, provenance, apply-once contributor receipts and attributed publication feedback.'],
        ['Runtime / checkpoints','Read-before-write scheduling, isolated counterfactual forks, named optimizer migration, row reset, detach/device transfer and serializable queues.']])+'\n\n'
    text+='Regional experts, HAT encoding, native recurrence and ordinary publications remain in their existing owners. The service stores compact associations and handles, while experts interpret the retained capsules. No replay buffer, answer dictionary, task parser, frozen backbone or extra answer-supervised binding loss was introduced.\n\n'
    text+='## Structural and gradient exits\n\n'
    text+=f"{data['structural']['targeted_checks']} targeted checks pass. Disabled behavior and the enabled zero-residual warm start preserve logits, RNG and clocks. Tests cover source/write/query/link/reader gradients, stale descriptor masking, correction/eviction, duplicate delivery, reset isolation, counterfactual branches, exact checkpoint/optimizer resume, two independent owners read by a third region, and owner-name permutation.\n\n"
    text+='Third-region, masked-owner and permutation checks are controlled wiring fixtures. The learned comparison uses the native three-region byte task.\n\n'
    text+='Training retains full live credit across recomputed segments and detaches at the optimizer boundary. Sparse recomputation leaves unused owner/page gradients absent, preserving optimizer behavior. The actual update-14 checkpoint reproduces 28-byte logits and loss exactly; the maximum gradient difference across 828 parameter tensors is 3.58e-7. A masked FP64 probe agrees to 1.14e-15 relative gradient error; FP32 accumulation differs by 1.17e-6 relative L2. Historical content gradients are not synthesized from detached handles.\n\n'
    text+='The shared arm resumed from update 14 after host-memory failures. The previously failing update completed at approximately 1.2 GiB process memory with 16-byte recomputation segments. Short calls use direct execution; the execution history records the switch. This changes activation storage, not the learning window or update/data budget.\n\n'
    text+='## Lateral scaling smoke\n\nThe same sparse service was exercised with 64-512 mock regional owners and 4-32 shards. This is a single-process CPU actor simulation. It measures bounded lookup, paging and transport behavior; physical multi-device/GPU execution remains the later H5 milestone. Latencies were sampled while other training was active.\n\n'
    text+=table(['Regions','Shards','p50 ms','p95 ms','Max candidates','Max posting reads','Max readbacks'],[
        [r['regions'],r['shards'],f"{r['fixed_load']['p50_ms']:.3f}",f"{r['fixed_load']['p95_ms']:.3f}",
         r['increasing_load']['max_work']['candidates'],r['increasing_load']['max_work']['posting_reads'],r['increasing_load']['max_work']['readbacks']] for r in data['scaling']])+'\n\n'
    text+='Each request is capped at two shards, 128 actual posting entries, 32 fine candidates and eight target readbacks, including association expansion. The directory also passes a zero-cue tie test without scanning every shard. Queues have explicit overload/expiry and fair service; future messages cannot block unrelated regions.\n\n'
    text+=table(['Regions','Burst requests','Requests/s','Visited owners max','Defer events','Dropped / expired'],[
        [r['regions'],r['increasing_load']['requests'],f"{r['increasing_load']['throughput_requests_s']:.1f}",
         r['increasing_load']['max_work']['regions'],r['increasing_load']['deferred'],
         f"{r['increasing_load']['dropped']} / {r['increasing_load']['expired']}"] for r in data['scaling']])+'\n\n'
    text+='The queue holds 64 requests. Defer counts are service-cycle events and may count one request more than once. Each smoke allocates one anchor/binding per mock owner; it does not saturate total capacity. Returned evidence plus cue occupies at most 256 bytes per request in this probe.\n\n'
    text+=table(['Regions','Persistent tensors KiB','Python metadata KiB','Saved tensors/query KiB','Gradient bytes/query KiB','Owners with gradient'],[
        [r['regions'],f"{r['persistent_tensor_bytes']/1024:.1f}",f"{r['persistent_metadata_estimate_bytes']/1024:.1f}",
         f"{r['one_query_saved_tensor_bytes']/1024:.1f}",f"{r['one_query_gradient_bytes']/1024:.1f}",r['owners_with_gradient']] for r in data['scaling']])+'\n\n'
    text+='Cold capacity uses lazily allocated immutable one-record pages. Python metadata is an object-size estimate. Saved-tensor and gradient allocations above belong to one bounded service query; they are separate from full language-model training activations.\n\n'
    text+='## Inference ablations\n\nEvery intervention uses the same final C weights and held-out histories. Shuffles only exchange causally valid selected anchors; no unavailable or future content is inserted.\n\n'
    ablations={'intact':c,**data['ablations']}
    text+=table(['C intervention','Both +0','Both +32','Both +64','Paired NLL +32'],[
        [name,*[pair(value[f'pairs_{g}']) for g in (0,32,64)],f"{value['pairs_32']['nll']:.6f}"] for name,value in ablations.items()])+'\n\n'
    text+='## Reproducibility and delivery\n\nThe source archive contains the implementation, tests, original specification and experiment commands. The checkpoint archive contains the retained parent, A/B/C endpoint and validation-selected checkpoints, exact corpus, full histories/evaluations, RNG and optimizer state, and SHA-256 checksums. Training source trajectories agree across all arms.\n\n'
    text+=table(['Arm','Weight KiB','Optimizer KiB','Carried state KiB'],[
        [labels[a],f"{arms[a]['unique_weight_bytes']/1024:.1f}",f"{arms[a]['optimizer_tensor_bytes']/1024:.1f}",f"{arms[a]['carried_state_tensor_bytes']/1024:.1f}"] for a in arms])+'\n\n'
    text+=table(['Arm','Saved training tensors MiB','Gradient MiB','Gradient tensors'],[
        [labels[a],f"{arms[a]['training_storage_probe']['saved_tensor_bytes_excluding_weights_inputs']/2**20:.3f}",
         f"{arms[a]['training_storage_probe']['gradient_tensor_bytes']/2**20:.3f}",
         arms[a]['training_storage_probe']['gradient_parameter_tensors']] for a in arms])+'\n\n'
    text+='The training-storage probe uses the same 56 natural-byte positions per model (batch 2, context 28), cold state, direct autograd and no optimizer update. Saved tensor storage excludes model weights and inputs; it includes newly created fast state. Python graph metadata and allocator overhead are excluded. This is a declared representative call, not the peak of a full task update. Full-run process high-water marks are retained in the histories.\n\n'
    text+='Weights and storage are counted uniquely; state-dict aliases do not inflate these totals. H0-H2 structural exits pass, H3 completes the fixed-budget comparison and ablations, and H4 completes the bounded actor smoke. The validation gate determines promotion independently of implementation completion.\n'
    out.mkdir(parents=True,exist_ok=True); md=out/'SHoEbox_Hippocampus_Implementation_Report.md';md.write_text(text)
    return md


def render_pdf(md,pdf):
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    filenames=('DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSansMono.ttf')
    if not all((fonts/name).exists() for name in filenames):
        fonts=Path(reportlab.__file__).parent/'fonts'
        filenames=('Vera.ttf','VeraBd.ttf','Vera.ttf')
    for name,filename in zip(('ReportSans','ReportSans-Bold','ReportMono'),filenames):
        pdfmetrics.registerFont(TTFont(name,str(fonts/filename)))
    pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSans-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall',fontName='ReportSans',fontSize=9,leading=12.5,spaceAfter=7,
        allowWidows=0,allowOrphans=0,textColor=colors.HexColor('#23333b')))
    styles.add(ParagraphStyle(name='Cell',fontName='ReportSans',fontSize=7.5,leading=10))
    styles.add(ParagraphStyle(name='TableHead',fontName='ReportSans-Bold',fontSize=7.5,leading=10,textColor=colors.white))
    styles['Title'].fontName='ReportSans-Bold';styles['Heading2'].fontName='ReportSans-Bold'
    styles['Title'].fontSize=21;styles['Title'].leading=25;styles['Title'].textColor=colors.HexColor('#123d37');styles['Title'].spaceAfter=13
    styles['Heading2'].fontSize=12;styles['Heading2'].leading=16;styles['Heading2'].spaceBefore=12;styles['Heading2'].textColor=colors.HexColor('#123d37')
    styles['Heading2'].keepWithNext=True
    def inline(value):
        value=escape(value).replace('–','-').replace('—','-')
        value=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',value)
        return re.sub(r'`(.*?)`',r'<font name="ReportMono">\1</font>',value)
    flow=[]; lines=md.read_text().splitlines(); i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip(): i+=1;continue
        if line.startswith('# '): flow.append(Paragraph(inline(line[2:]),styles['Title']));i+=1;continue
        if line.startswith('## '):
            if line[3:] in ('Validation and promotion','Structural and gradient exits','Reproducibility and delivery'):
                flow.append(PageBreak())
            flow.append(Paragraph(inline(line[3:]),styles['Heading2']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(set(x)<=set('-: ') for x in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);width=528;first=112 if n>3 else 130
            widths=[first]+[(width-first)/(n-1)]*(n-1)
            contents=[[Paragraph(inline(v),styles['TableHead' if r==0 else 'Cell']) for v in row] for r,row in enumerate(rows)]
            item=Table(contents,colWidths=widths,repeatRows=1,hAlign='LEFT')
            item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#174d43')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf3f0'),colors.white]),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#174d43')),
                ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            flow.append(KeepTogether([item,Spacer(1,8)]));continue
        paragraph=[]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|')):
            paragraph.append(lines[i]);i+=1
        flow.append(KeepTogether([Paragraph(inline(' '.join(paragraph)),styles['BodySmall'])]))
    def footer(canvas,document):
        canvas.setStrokeColor(colors.HexColor('#c5d3cc'));canvas.line(42,38,570,38)
        canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#587069'))
        canvas.drawString(42,25,'SHoEbox | Explicit hippocampus | November 8, 2026 UTC')
        canvas.drawRightString(570,25,str(document.page))
    SimpleDocTemplate(str(pdf),pagesize=letter,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=52,
        title='SHoEbox hippocampus implementation',author='Joshua Rowe / SHoEbox').build(flow,onFirstPage=footer,onLaterPages=footer)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    md=write_report(json.loads(a.summary.read_text()),a.out);render_pdf(md,md.with_suffix('.pdf'));print(md);print(md.with_suffix('.pdf'))

if __name__=='__main__':main()
