"""Evaluate Aditi's checkpoint and build Yosephine's detection-results report."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
IDS = ['7007', '2970', '2336', '7', '4428']
CONF = 0.25
MATCH_IOU = 0.5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + '\n', encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def overlap(a, b):
    area = lambda r: max(0, r[2]-r[0]) * max(0, r[3]-r[1])
    inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def read_boxes(path, predictions=False):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        c, x, y, w, h, *score = map(float, line.split())
        result.append({'class_id': int(c), 'box': [x-w/2, y-h/2, x+w/2, y+h/2],
                       'confidence': score[0] if predictions else 1.0})
    return result


def match_boxes(truth, predictions):
    """One-to-one, same-class matches, taking the highest IoU first."""
    candidates = sorted([(overlap(g['box'], p['box']), i, j)
                         for i, g in enumerate(truth) for j, p in enumerate(predictions)
                         if g['class_id'] == p['class_id'] and overlap(g['box'], p['box']) >= MATCH_IOU],
                        reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for value, i, j in candidates:
        if i not in used_g and j not in used_p:
            pairs.append((i, j, value)); used_g.add(i); used_p.add(j)
    return pairs, used_g, used_p


def fixed_metrics(data, prediction_dir, split):
    counts = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []} for i in range(5)}
    audit, image_counts, raw_predictions = [], [], []
    for label in sorted((data / split / 'labels').glob('*.txt')):
        truth = read_boxes(label)
        all_preds = read_boxes(prediction_dir / label.name, predictions=True)
        predictions = [p for p in all_preds if p['confidence'] >= CONF]
        pairs, used_g, used_p = match_boxes(truth, predictions)
        image_counts.append({'image': label.stem + '.jpg', 'labelled_faces': len(truth),
                             'correct_matches': len(pairs), 'false_positives': len(predictions)-len(pairs),
                             'false_negatives': len(truth)-len(pairs)})
        for p in all_preds:
            raw_predictions.append({'split': split, 'image': label.stem + '.jpg', **p})
        for i, j, value in pairs:
            c = truth[i]['class_id']; counts[c]['tp'] += 1; counts[c]['ious'].append(value)
            audit.append({'image': label.stem+'.jpg', 'status': 'matched', 'identity': IDS[c],
                          'truth_index': i, 'prediction_index': j, 'confidence': predictions[j]['confidence'], 'iou': value})
        for i, g in enumerate(truth):
            if i not in used_g:
                counts[g['class_id']]['fn'] += 1
                audit.append({'image': label.stem+'.jpg', 'status': 'unmatched_label', 'identity': IDS[g['class_id']],
                              'truth_index': i, 'prediction_index': '', 'confidence': '', 'iou': ''})
        for j, p in enumerate(predictions):
            if j not in used_p:
                counts[p['class_id']]['fp'] += 1
                audit.append({'image': label.stem+'.jpg', 'status': 'unmatched_prediction', 'identity': IDS[p['class_id']],
                              'truth_index': '', 'prediction_index': j, 'confidence': p['confidence'], 'iou': ''})
    def summarize(c):
        tp, fp, fn = (c[k] for k in ('tp', 'fp', 'fn'))
        return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp/(tp+fp) if tp+fp else 0,
                'recall': tp/(tp+fn) if tp+fn else 0,
                'mean_matched_iou': sum(c['ious'])/len(c['ious']) if c['ious'] else None}
    total = {k: sum(c[k] for c in counts.values()) for k in ('tp', 'fp', 'fn')}
    total['ious'] = [v for c in counts.values() for v in c['ious']]
    return {'overall': summarize(total), 'per_class': {IDS[i]: summarize(c) for i,c in counts.items()}}, audit, image_counts, raw_predictions


def source_audit(data):
    with (data/'source_split.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    hashes, source_sets = {}, {}
    for row in rows:
        image = ROOT/'data/identities'/row['file']
        value = digest(image)
        assert value == row['sha256'], f'Changed source photo: {row["file"]}'
        assert value not in hashes or hashes[value] == row['split'], 'Source photo shared between splits'
        hashes[value] = row['split']
        source_sets.setdefault(row['split'], set()).add(row['file'])
    manifest_hash = hashlib.sha256()
    for split in ('train','val','test'):
        images = list((data/split/'images').glob('*.jpg'))
        labels = list((data/split/'labels').glob('*.txt'))
        assert {p.stem for p in images} == {p.stem for p in labels}
        for p in sorted(images+labels):
            manifest_hash.update(p.relative_to(data).as_posix().encode())
            manifest_hash.update(p.read_bytes().replace(b'\r\n',b'\n') if p.suffix=='.txt' else p.read_bytes())
    return {'source_counts': {s:len(v) for s,v in source_sets.items()},
            'shared_source_photos': 0, 'dataset_sha256': manifest_hash.hexdigest()}


def make_curves(log, target, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(10,3.15))
    for ax, key, title in zip(axes, ('box','cls','dfl'), ('Face-box loss','Identity loss','Box-refinement loss')):
        ax.plot(log['epoch'], log[f'train/{key}_loss'], label='Training', color='#246b8e', lw=1.7)
        ax.plot(log['epoch'], log[f'val/{key}_loss'], label='Validation', color='#ba6239', lw=1.7)
        ax.axvline(selected,color='#888888',ls=':',lw=1)
        ax.set(title=title,xlabel='Epoch',ylabel='Loss');ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(target,dpi=180,bbox_inches='tight');plt.close(fig)


def build_report(summary, result_dir, report_dir, sweep=None):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.pagesizes import letter

    report_dir.mkdir(parents=True, exist_ok=True)
    test = summary['test']
    scores = test['fixed_confidence']['overall']
    train = summary['training']
    pct = lambda value: f'{100 * value:.1f}%'
    selected = next((row for row in sweep['test'] if row['config_id'] == sweep['selection']['config_id']), None) if sweep else None
    table_rows = [
        ['Metric', 'Baseline', 'Selected settings'] if selected else ['Metric', 'Test result'],
        ['mAP@0.5', pct(test['standard']['map50'])],
        ['mAP@0.5:0.95', pct(test['standard']['map50_95'])],
        ['Precision', pct(scores['precision'])],
        ['Recall', pct(scores['recall'])],
        ['Mean IoU of correct matches', f'{scores["mean_matched_iou"]:.3f}'],
    ]
    if selected:
        for row, value in zip(table_rows[1:], [pct(selected['map50']), pct(selected['map50_95']),
                              pct(selected['precision']), pct(selected['recall']), f"{selected['mean_matched_iou']:.3f}"]):
            row.append(value)
    intro = ('We tested Aditi\'s saved YOLOv8n model on the held-out Milestone 2 dataset: '
             '36 images containing 113 labelled faces.')
    method = ('A correct match needs the right identity and box overlap (IoU) of at least 0.50. '
              'Mean IoU measures only correct matches, excluding misses and false alarms.')
    if selected:
        method += ' mAP uses the full confidence curve, so changing the reporting cutoff alone does not change it.'
    else:
        method += ' Precision and recall use confidence 0.25.'
    interpretation = [
        ('The selected settings produced fewer false alarms, with a small increase in missed faces. '
         'The model weights stayed the same; only the prediction settings were compared.' if selected else
         'The model found and labelled most faces correctly. Some errors were wrong identities or extra predictions around a face.'),
        'These results apply to synthetic scenes made from a small set of repeated photos. The labels were '
        'generated with a face detector. Strong scores here do not guarantee the same performance on real group photos.',
    ]
    training = (f'The model trained for {train["epochs"]} epochs. Epoch {train["selected_epoch"]} had the best '
                'validation score and supplied the checkpoint tested here. The test results were not used to choose it.')
    curves = ('Training loss fell, but validation loss stopped improving as much. '
              'The dotted line marks the selected checkpoint.')
    ta_check = ('The model placed boxes around all nine faces in the TA image. However, the middle-centre '
                'and middle-right faces each received overlapping predictions with different IDs.')
    if selected:
        ta_check = 'Using the selected settings, ' + ta_check[0].lower() + ta_check[1:]
    ta_limit = ('No reference identity labels or boxes were supplied, so this is a visual check rather than '
                'a scored test. The model can output only its trained identities; a confident prediction can still be wrong.')
    sweep_text = ('Dario\'s code compared 24 prediction settings on validation, without retraining. '
                  'The choice balanced precision and recall using F1. Only the baseline and selected settings '
                  'were then checked on test.') if sweep else 'The hyperparameter comparison is not included in this run.'
    sweep_rows = [['Setting', 'Compared', 'Selected'],
                  ['Confidence cutoff', '0.10, 0.25, 0.50, 0.75', f"{selected['confidence']:.2f}"],
                  ['Duplicate-box overlap (NMS IoU)', '0.30, 0.50, 0.70', f"{selected['nms_iou']:.2f}"],
                  ['Duplicate filtering', 'Within each ID / across IDs', 'Within each ID' if not selected['agnostic_nms'] else 'Across IDs']] if selected else []

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Simple', fontName='Helvetica', fontSize=10, leading=14,
                              spaceAfter=9, textColor=colors.HexColor('#25313b')))
    styles.add(ParagraphStyle(name='Note', fontName='Helvetica', fontSize=8.5, leading=11.5,
                              spaceAfter=9, textColor=colors.HexColor('#46535e')))
    styles['Title'].fontName = 'Helvetica-Bold'
    styles['Title'].fontSize = 21
    styles['Title'].leading = 25
    styles['Title'].alignment = TA_LEFT
    styles['Heading2'].fontSize = 13
    styles['Heading2'].spaceBefore = 9
    styles['Heading2'].spaceAfter = 7
    story = []
    def paragraph(text, style='Simple'):
        story.append(Paragraph(text, styles[style]))

    paragraph('Milestone 3 | Detection results', 'Title')
    paragraph('IE 7615 - Group 1 | Evaluation and report: Yosephine Tong', 'Note')
    paragraph(intro)
    cells = [[Paragraph(value, styles['Simple']) for value in row] for row in table_rows]
    table = Table(cells, colWidths=[260, 122, 122] if selected else [340, 164], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6edf1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#dce3e7')),
    ]))
    story.extend([table, Spacer(1, 9)])
    if selected:
        paragraph(f"Confidence cutoffs: baseline 0.25; selected {selected['confidence']:.2f}.", 'Note')
    paragraph(method, 'Note')
    for text in interpretation:
        paragraph(text)
    paragraph('Training and validation loss', 'Heading2')
    paragraph(training)
    story.append(Image(str(result_dir / 'loss_curves.png'), width=504, height=159))
    paragraph(curves, 'Note')
    story.append(PageBreak())
    paragraph('Hyperparameter comparison', 'Title')
    paragraph(sweep_text)
    if sweep_rows:
        grid = Table([[Paragraph(value, styles['Note']) for value in row] for row in sweep_rows],
                     colWidths=[200, 200, 104], hAlign='LEFT')
        grid.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e6edf1')),
                                 ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                 ('LINEBELOW', (0,0), (-1,-1), .3, colors.HexColor('#dce3e7'))]))
        story.extend([grid, Spacer(1, 9)])
    paragraph('TA image check', 'Heading2')
    paragraph(ta_check)
    story.append(Image(str(result_dir / 'ta_image_predictions.jpg'), width=270, height=270, hAlign='CENTER'))
    story.append(Spacer(1, 12))
    paragraph(ta_limit)
    paragraph('Detailed scores and run settings are saved in project_milestone03/results/.', 'Note')
    paragraph('Metric reference: docs.ultralytics.com/modes/val', 'Note')

    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#61717d'))
        canvas.drawString(54, 28, 'Group 1 | Detection evaluation')
        canvas.drawRightString(558, 28, str(doc.page))
    SimpleDocTemplate(str(report_dir / 'milestone03_detection_results.pdf'), pagesize=letter,
                      rightMargin=54, leftMargin=54, topMargin=42, bottomMargin=42,
                      title='Milestone 3 - Detection results', author='IE 7615 Group 1').build(
                          story, onFirstPage=footer, onLaterPages=footer)


def ta_conflicts(ta, shape):
    rows=[]
    for i,a in enumerate(ta['boxes_xyxy']):
        for j,b in enumerate(ta['boxes_xyxy'][:i]):
            if ta['class_ids'][i]!=ta['class_ids'][j] and overlap(a,b)>=0.5:
                x=(a[0]+a[2])/2; y=(a[1]+a[3])/2
                row=min(2,int(y/shape[0]*3));col=min(2,int(x/shape[1]*3))
                location=['Top','Middle','Bottom'][row]+' '+['left','centre','right'][col]
                values=sorted([(ta['confidence'][k],IDS[ta['class_ids'][k]]) for k in (i,j)],reverse=True)
                rows.append({'location':location,'predictions':' / '.join(f'{identity} ({score:.2f})' for score,identity in values)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-zip',type=Path,required=True)
    parser.add_argument('--ta-image',type=Path,required=True)
    parser.add_argument('--training-notebook',type=Path)
    parser.add_argument('--sweep-summary',type=Path, help='Completed sweep run_summary.json to include in the report.')
    args=parser.parse_args()
    config=ROOT/'tmp/m3_config';config.mkdir(parents=True,exist_ok=True);os.environ['YOLO_CONFIG_DIR']=str(config)
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    from ultralytics.utils.metrics import smooth
    SETTINGS.update({'sync':False})
    assert ultralytics.__version__=='8.4.174','Use the pinned Ultralytics version.'
    result_dir=HERE/'results';result_dir.mkdir(parents=True,exist_ok=True)
    detail_dir=ROOT/'tmp/m3_details';detail_dir.mkdir(parents=True,exist_ok=True)
    data=ROOT/'project_milestone02/dataset'
    provenance=source_audit(data)
    # Extract only the files needed for evaluation; never execute the supplied notebook.
    with tempfile.TemporaryDirectory(prefix='m3_eval_',dir=ROOT/'tmp') as folder:
        work=Path(folder)
        with zipfile.ZipFile(args.training_zip) as archive:
            for name in ('weights/best.pt','results.csv','args.yaml'):
                target=work/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
        log=pd.read_csv(work/'results.csv');log.columns=log.columns.str.strip()
        assert len(log)==50 and list(log.epoch)==list(range(1,51))
        model=YOLO(str(work/'weights/best.pt'));assert list(model.names.values())==IDS
        selected=int(log.loc[log['metrics/mAP50-95(B)'].idxmax(),'epoch'])
        stored=model.ckpt['train_metrics'];bestrow=log.loc[log.epoch==selected].iloc[0]
        assert abs(stored['fitness']-bestrow['metrics/mAP50-95(B)'])<1e-5
        assert len(model.ckpt['train_results']['epoch'])==len(log)
        optimizer='Auto (see training log)'
        notebook_hash=None
        if args.training_notebook:
            notebook_hash=digest(args.training_notebook)
            nb=json.loads(args.training_notebook.read_text(encoding='utf-8'))
            transcript=''.join(''.join(o.get('text',[])) for cell in nb['cells'] for o in cell.get('outputs',[]))
            found=re.search(r'AdamW\(lr=([0-9.]+), momentum=([0-9.]+)\)',transcript)
            if found: optimizer=f'AdamW, initial learning rate {found[1]}'
        summary={'checkpoint_sha256':digest(work/'weights/best.pt'),'training_zip_sha256':digest(args.training_zip),
                 'training_notebook_sha256':notebook_hash,'classes':IDS,'dataset':provenance,
                 'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 'environment':{'python':platform.python_version(),'ultralytics':ultralytics.__version__,'torch':torch.__version__,
                                'numpy':np.__version__,'device':'cpu'},
                 'evaluation':{'imgsz':640,'batch':8,'workers':0,'ap_confidence_floor':.001,'nms_iou':.7,
                               'fixed_confidence':CONF,'match_iou':MATCH_IOU,'matching':'same class, one-to-one, descending IoU',
                               'rect':True,'augment':False},
                 'training':{'epochs':len(log),'selected_epoch':selected,'minutes':float(log.time.iloc[-1])/60,
                             'optimizer_description':optimizer,'selected_epoch_validation':stored}}
        spec=yaml.safe_load((data/'data.yaml').read_text());spec['path']=str(data.resolve())
        (work/'data.yaml').write_text(yaml.safe_dump(spec),encoding='utf-8')
        predictions=[]
        for split in ('val','test'):
            metrics=model.val(data=str(work/'data.yaml'),split=split,imgsz=640,batch=8,device='cpu',workers=0,
                              conf=.001,iou=.7,rect=True,augment=False,plots=False,save_txt=True,save_conf=True,
                              project=str(work),name=split,exist_ok=False,verbose=False)
            point=int(smooth(metrics.box.f1_curve.mean(0),.1).argmax())
            fixed,audit,image_counts,raw=fixed_metrics(data,metrics.save_dir/'labels',split);predictions.extend(raw)
            assert sum(r['labelled_faces'] for r in image_counts)==(111 if split=='val' else 113)
            summary[split]={'standard':{'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),
                                       'precision_auto':float(metrics.box.mp),'recall_auto':float(metrics.box.mr),
                                       'automatic_summary_confidence':float(metrics.box.px[point])},
                            'fixed_confidence':fixed,'per_class':metrics.summary()}
            if split=='test':
                save_csv(detail_dir/'test_box_matches.csv',audit);save_csv(detail_dir/'test_image_counts.csv',image_counts)
                save_csv(detail_dir/'per_class_metrics.csv',[{'identity':r['Class'],'faces':r['Instances'],
                          'ap50':r['mAP50'],'ap50_95':r['mAP50-95'],**fixed['per_class'][r['Class']]} for r in metrics.summary()])
        save_json(detail_dir/'predictions.json',predictions)
        (result_dir/'training_history.csv').write_bytes((work/'results.csv').read_bytes())
        (result_dir/'training_args.yaml').write_bytes((work/'args.yaml').read_bytes())
        make_curves(log,detail_dir/'loss_curves.png',selected)
        sweep=json.loads(args.sweep_summary.read_text(encoding='utf-8')) if args.sweep_summary else None
        ta_config={'confidence':CONF,'nms_iou':.7,'agnostic_nms':False}
        if sweep:
            assert sweep['checkpoint_sha256']==summary['checkpoint_sha256'], 'Sweep checkpoint differs.'
            assert sweep['dataset']['dataset_sha256']==provenance['dataset_sha256'], 'Sweep dataset differs.'
            assert sweep['selection']['split']=='val', 'Selection must use validation.'
            ta_config={key:sweep['selection'][key] for key in ta_config}
            summary['report_sweep']={'file':os.path.relpath(args.sweep_summary,ROOT).replace('\\','/'),
                                     'sha256':digest(args.sweep_summary),'selected_settings':ta_config}
        ta=model.predict(source=str(args.ta_image.resolve()),imgsz=640,conf=ta_config['confidence'],
                         iou=ta_config['nms_iou'],agnostic_nms=ta_config['agnostic_nms'],device='cpu',verbose=False)[0]
        ta.save(filename=str(detail_dir/'ta_image_predictions.jpg'))
        summary['ta_image']={'sha256':digest(args.ta_image),'prediction_count':len(ta.boxes),
                             'boxes_xyxy':ta.boxes.xyxy.tolist(),'class_ids':ta.boxes.cls.int().tolist(),
                             'confidence':ta.boxes.conf.tolist(),'reference_labels_available':False,'settings':ta_config}
        summary['ta_image']['overlapping_predictions']=ta_conflicts(summary['ta_image'],ta.orig_shape)
        save_json(result_dir/'metrics.json',summary)
        build_report(summary,detail_dir,ROOT/'output and pdf',sweep=sweep)
        print(json.dumps({'test':summary['test']['fixed_confidence']['overall'],'mAP':summary['test']['standard'],
                          'report':'output and pdf/milestone03_detection_results.pdf'},indent=2))


if __name__=='__main__':
    main()
