"""Build the proposal and results reports from the verified Group 1 run."""
from pathlib import Path
import csv
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"/"pdf";OUT.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom",fontName="Helvetica-Bold",fontSize=22,leading=25,textColor=colors.HexColor("#12304a"),spaceAfter=14))
styles.add(ParagraphStyle(name="SubCustom",fontName="Helvetica-Bold",fontSize=12,leading=15,textColor=colors.HexColor("#146a79"),spaceBefore=12,spaceAfter=6))
styles.add(ParagraphStyle(name="BodyCustom",fontName="Helvetica",fontSize=9.5,leading=13,spaceAfter=7))
styles.add(ParagraphStyle(name="SmallCustom",fontName="Helvetica",fontSize=8,leading=11,spaceAfter=5))
def p(text,style="BodyCustom"):return Paragraph(text,styles[style])
def title(kicker,text):return [p(kicker,"SmallCustom"),p(text,"TitleCustom")]
def section(text):return p(text,"SubCustom")
def table(rows,widths):
    converted=[[p(str(c),"SmallCustom") for c in row] for row in rows]
    t=Table(converted,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7f0f4")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#456579")),
        ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#d9e2e8")),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    return t
def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor("#cbd8e0"));canvas.line(48,40,564,40)
    canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#52616b"))
    canvas.drawString(48,27,"IE 7615 | Group 1")
    canvas.drawRightString(564,27,str(doc.page))
def build(name,story):
    SimpleDocTemplate(str(OUT/name),pagesize=(612,792),leftMargin=48,rightMargin=48,
        topMargin=42,bottomMargin=54,title=name.replace("_"," "),author="IE 7615 Group 1").build(story,onFirstPage=footer,onLaterPages=footer)


import json
RUN="group1_repro_v1"
BASE=ROOT/"results"/RUN
names=["small_custom_cnn","deeper_custom_cnn","resnet18_frozen_backbone"]
labels=["Small CNN","Deeper CNN","ResNet18"]
metrics=[json.loads((BASE/(n+"_metrics.json")).read_text()) for n in names]
selection=json.loads((BASE/"selection.json").read_text())
assert selection["selected_model"]==names[2]
proof=json.loads((BASE/"reproducibility_check.json").read_text())
assert len(proof["checks"])==3 and all(all(v is True for k,v in c.items() if k!="architecture") for c in proof["checks"])
chosen=metrics[2]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(8,7.2),sharex=True)
loss_notes=[]
for ax,name,label,m in zip(axes,names,labels,metrics):
    history=list(csv.DictReader((ROOT/"logs"/RUN/(name+"_history.csv")).open()))
    best=min(history,key=lambda r:float(r["val_loss"]))
    assert int(best["epoch"])==m["selected_epoch"]
    epochs=[int(r["epoch"]) for r in history]
    ax.plot(epochs,[float(r["train_loss"]) for r in history],color="#166b8b",label="Training",linewidth=1.8)
    ax.plot(epochs,[float(r["val_loss"]) for r in history],color="#cc7633",label="Validation",linewidth=1.8)
    ax.axvline(m["selected_epoch"],color="#79848c",linestyle=":",label="Selected epoch")
    ax.set_title(label,loc="left",fontsize=11,fontweight="bold")
    ax.set_ylabel("Loss");ax.grid(alpha=.18);ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8,ncol=3)
    loss_notes.append(f"{label}: lowest validation loss {float(best['val_loss']):.3f} at epoch {m['selected_epoch']}.")
axes[-1].set_xlabel("Epoch");fig.tight_layout()
loss_path=BASE/"training_loss_curves.png";fig.savefig(loss_path,dpi=180);plt.close(fig)
team="Yosephine Tong, Dario Garza and Aditi"
goal="We are building a system that identifies which of five celebrities appears in a photo. Later milestones will add the ability to find several faces in one image. We use Python with PyTorch and torchvision to build and train the models."
data="We use CelebA identity IDs 7007, 2970, 2336, 7 and 4428. Each folder has enough photos to use 23 per person: 17 for learning, 3 for choosing the best model, and 3 for the final test. Using the same number of photos per person keeps the comparison balanced."
main="We compared three image-recognition models. ResNet18 identified the correct person in 10 of 15 test photos (66.7%). The deeper CNN got 6 correct (40.0%), and the small CNN got 5 correct (33.3%)."
models_text="The small and deeper CNNs learn from our photos from the start. ResNet18 already learned useful image features from ImageNet, a large image dataset. We keep that part unchanged and train only its final layer to choose among our five identities."
setup="Each model makes 30 passes through the training photos, called epochs. After each pass, we check it on the validation photos, which are used to choose the model. For each model, we save the version with the lowest validation loss (a measure of prediction error). We then choose the model with the most correct validation predictions and evaluate it on the separate test photos."
preprocess="We convert photos to color (RGB) and crop them to 224 x 224 pixels. During training, random crops and horizontal flips add variety. For validation and testing, we resize the shorter edge to 256 pixels and take a fixed center crop. Pixel values are adjusted using ImageNet's mean and standard deviation."
choice="We chose ResNet18 because it correctly identified 12 of 15 validation photos (80.0%); each custom CNN identified 4 (26.7%). We made this choice before checking the test results. ResNet18's earlier training on ImageNet gives it useful image features when our own training set is small."
tradeoff=f"ResNet18 is larger and slower than the two custom CNNs. It contains about 11.18 million model values, called parameters, but we train only 2,565 in its final layer. It takes about {chosen['inference_ms_per_image']:.1f} milliseconds to make a prediction on our CPU, excluding image preparation. We will use it as our starting model for later milestones because it performed best in this comparison."
limit="The test set is small: just three photos per person. One correct or incorrect prediction changes the overall score by 6.7 percentage points. We need more varied photos to judge how well the model handles different lighting, poses and backgrounds."
repro="The repository includes the photos, the list showing how they were divided, all three trained models, training logs and test predictions."
roles=[["Member","Milestone 1 work"],["Dario","Prepared the initial data and model-training notebook"],["Aditi","Evaluated models, compared results and shared training logs"],["Yosephine","Organized the repository, repeated training and prepared reports"]]
next_steps="Milestone 2 (end of Module 4): combine celebrity photos into new images and label each face with a box. Milestone 3 (mid-Module 5): train YOLOv8 to find those faces. Milestone 4 (end of Module 6): bring the system together and complete the final report."
risks="Our small dataset may not represent the photos the system will see later. Combined images may also look different from real scenes. We will keep training and test images separate and check the face labels carefully."
proposal=title("IE 7615 / GROUP 1 / PROJECT 1","Project proposal")
proposal += [p("<b>Team:</b> "+team+"."),section("What we are building"),p(goal),section("Our photos"),p(data),section("What we found"),p(main),p("ResNet18 also performed best on the validation photos used to choose the model. We will use it in the next stages of the project. The trained models and everything needed to check these results are saved in the repository."),section("Team contributions"),table(roles,[80,436]),section("Next steps"),p(next_steps),section("Main risks"),p(risks)]
build("group1_proposal.pdf",proposal)
report=title("IE 7615 / GROUP 1 / MILESTONE 1","Classification results")
report += [p("<b>Team:</b> "+team+"."),p(main),section("Photos and preparation"),p("We use five CelebA identity IDs: 7007, 2970, 2336, 7 and 4428. The folders contain 121 photos in total. We use 23 per person (115 total), leaving six unused. These are the dataset's ID labels; we have not verified celebrity names."),table([["Photo set","Purpose","Total"],["Training","Teach the models","85"],["Validation","Choose the model","15"],["Test","Measure the final result","15"]],[100,340,76]),Spacer(1,6),p("Each person has 17 training, 3 validation and 3 test photos. The file list is saved so every model uses the same photos. We found no identical image files across these groups."),p(preprocess),section("The three models"),p(models_text),section("How we trained them"),p(setup),p("All models use Adam, a learning rate of 0.001 and batches of 16 images. The small CNN has two convolution layers (32 and 64 channels); the deeper CNN has four (32, 64, 128 and 256). Dropout rates are 0.25 and 0.35. The file shuffle uses seed 47 plus the identity number, so the same photos are selected each time.","SmallCustom"),PageBreak()]
report += title("MODEL COMPARISON","Results and training progress")
report += [table([["Model","Validation","Test","Parameters<br/>(total / trained)","Training"]]+[[label,f'{m["validation_accuracy"]:.1%}',f'{m["test_accuracy"]:.1%}<br/>({m["test_correct"]}/15)',f'{m["total_parameters"]:,}<br/>{m["trainable_parameters"]:,}',f'{m["training_seconds"]:.1f} s'] for label,m in zip(labels,metrics)],[80,72,72,188,104]),p("Parameters are the numbers stored inside a model. Training time includes validation and saving the best version. Measurements use an Intel Core i7-7700K CPU with two threads.","SmallCustom"),section("How prediction error changed during training"),p("Loss measures prediction error; lower is better. Blue shows training photos and orange shows validation photos. The dotted line marks the saved version. These curves come directly from the training logs."),Image(str(loss_path),width=420,height=378),p("The small CNN improves slowly. The deeper CNN learns the training photos better, but its validation error rises and falls. ResNet18 has the lowest validation error. The selected versions are from training passes 30, 21 and 28 for the small CNN, deeper CNN and ResNet18.","SmallCustom"),PageBreak()]
report += title("OUR CHOICE","Why we chose ResNet18")
report += [p(choice),p(tradeoff),Image(str(BASE/(names[2]+"_confusion.png")),width=270,height=225),p("This chart shows the test mistakes. Rows show the actual person; columns show the model's answer. Numbers on the diagonal are correct answers. For IDs 7007, 2970, 2336, 7 and 4428, the model gets 1, 2, 3, 3 and 1 of the three photos correct. It mistakes 4428 for 2970 twice.","SmallCustom"),table([["Prediction time","Small CNN","Deeper CNN","ResNet18"],["Milliseconds per photo"]+[f'{m["inference_ms_per_image"]:.1f}' for m in metrics]],[156,120,120,120]),p("We timed one photo at a time, after a warm-up, and report the median over five passes through the test set. Image loading and preparation are excluded.","SmallCustom"),section("What these results can tell us"),p(limit),section("Checking the results"),p(repro),p("The README explains how to open notebooks 01-05, check the saved models and train them again. These results are saved as group1_repro_v1. Aditi's earlier results are kept separately for reference.","SmallCustom")]
build("group1_training_results.pdf",report)
# Keep the Markdown copies consistent with the PDFs.
proposal_md=['# Project 1 proposal','',f'Team: {team}. IE 7615, Group 1.','','## What we are building','',goal,'','## Our photos','',data,'','## What we found','',main,'','ResNet18 also performed best on validation. We will use it in the next stages of the project. All trained models and the files needed to check our results are included.','','## Team contributions','','| Member | Milestone 1 work |','| --- | --- |']
proposal_md += [f'| {row[0]} | {row[1]} |' for row in roles[1:]]
proposal_md += ['','## Next steps','',next_steps,'','## Main risks','',risks]
(ROOT/'docs/proposal.md').write_text('\n'.join(proposal_md)+'\n',encoding='utf-8')
lines=['# Milestone 1: classification results','',f'Team: {team}.','',main,'','## Photos and preparation','',data,'','In total, we use 85 training, 15 validation and 15 test photos. Six of the 121 available photos are unused. The [saved file list](../configs/splits/group1_repro_v1_seed47.csv) identifies each photo and its group.','',preprocess,'','## Models and training','',models_text,'',setup,'','All models use Adam, learning rate 0.001 and batches of 16. The small CNN has channels 32/64 and dropout 0.25; the deeper CNN has channels 32/64/128/256 and dropout 0.35. The file shuffle uses seed 47 plus the identity number.','','## Comparison','','| Model | Saved training pass | Validation | Test | Parameters: total / trained | Training (s) | Prediction (ms/photo) |','| --- | --- | --- | --- | --- | --- | --- |']
for label,m in zip(labels,metrics):lines.append(f'| {label} | {m["selected_epoch"]} | {m["validation_accuracy"]:.1%} | {m["test_accuracy"]:.1%} ({m["test_correct"]}/15) | {m["total_parameters"]:,} / {m["trainable_parameters"]:,} | {m["training_seconds"]:.1f} | {m["inference_ms_per_image"]:.1f} |')
lines += ['','Parameters are numbers stored inside a model. Times use an Intel Core i7-7700K CPU with two threads. Training includes validation and saving. Prediction time is the median for one photo at a time over five test passes after warm-up; it excludes image loading and preparation.','','## Training progress','','Loss measures prediction error; lower is better. Blue shows training photos, orange shows validation photos, and the dotted line marks the saved version.','','![Training and validation loss](../results/group1_repro_v1/training_loss_curves.png)','','The custom CNNs improve on training photos, but their validation results are weaker. ResNet18 has the lowest validation error.','','## Why we chose ResNet18','',choice,'',tradeoff,'','![Test results by identity](../results/group1_repro_v1/resnet18_frozen_backbone_confusion.png)','','Rows show the actual person; columns show the prediction. Correct counts for IDs 7007, 2970, 2336, 7 and 4428 are 1, 2, 3, 3 and 1 out of three photos each. Identity 4428 is mistaken for 2970 twice.','',limit,'','## Checking the results','',repro,'','See the [README](../README.md) for commands and the [verification record](../results/group1_repro_v1/reproducibility_check.json) for the repeat check. These results are saved as `group1_repro_v1`; Aditi\'s earlier results remain separate.']
(ROOT/'docs/training_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Updated both PDFs and matching Markdown reports.')
