"""Build the training report; keep the team-maintained proposal unchanged."""
from pathlib import Path
import csv
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
ROOT=Path(__file__).resolve().parents[1]
import argparse,sys
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--measure-speed",action="store_true",help="Remeasure prediction speed before building the report")
parser.add_argument("--output-dir",type=Path,default=ROOT.parent/"output and pdf")
args=parser.parse_args()
OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
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
# Use the recorded timings unless explicitly asked to measure again.
from src.pipeline import write_csv,write_json
if args.measure_speed:
    import statistics,time,platform,torch
    from src.pipeline import Experiment,make_model,make_loader,read_csv
    exp=Experiment(ROOT.parent/"data/identities",RUN,device="cpu")
    for name,result in zip(names,metrics):
        checkpoint=torch.load(exp.path("models",name+".pt"),map_location="cpu",weights_only=True)
        model=make_model(name,False).eval();model.load_state_dict(checkpoint["state_dict"])
        images=[x for x,_,_ in make_loader(exp.data_dir,read_csv(exp.manifest),"test",1,47)]
        times=[]
        with torch.no_grad():
            for x in images:model(x)
            for _ in range(5):
                for x in images:
                    start=time.perf_counter();model(x);times.append((time.perf_counter()-start)*1000)
        result["inference_ms_per_image"]=statistics.median(times)
        result["timing_protocol"]="CPU, 2 threads, batch size 1; one warm-up pass, median of 5 full test passes; forward pass only"
        result["hardware"]=platform.processor() or platform.machine()
        write_json(BASE/(name+"_metrics.json"),result)
fields=["architecture","selected_epoch","validation_accuracy","test_accuracy","test_correct","test_images","total_parameters","trainable_parameters","training_seconds","inference_ms_per_image"]
write_csv(BASE/"model_comparison.csv",[{k:m[k] for k in fields} for m in metrics])
write_csv(BASE/"per_class_accuracy.csv",[
    {"architecture":m["architecture"],"identity":identity,"correct":m["confusion_matrix"][i][i],"test_images":sum(m["confusion_matrix"][i]),"accuracy":m["per_class_accuracy"][identity]}
    for m in metrics for i,identity in enumerate(m["ids"])])
chosen=metrics[2]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(8,7.2),sharex=True)
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
axes[-1].set_xlabel("Epoch");fig.tight_layout()
loss_path=BASE/"training_loss_curves.png";fig.savefig(loss_path,dpi=180);plt.close(fig)
team="Yosephine Tong, Dario Garza and Aditi"
main="We compared three image-recognition models. ResNet18 identified the correct person in 10 of 15 test photos (66.7%). The deeper CNN got 6 correct (40.0%), and the small CNN got 5 correct (33.3%)."
models_text="The small and deeper CNNs learn from our photos from the start. ResNet18 already learned useful image features from ImageNet, a large image dataset. We keep that part unchanged and train only its final layer to choose among our five identities."
setup="Each model makes 30 passes through the training photos, called epochs. After each pass, we check it on the validation photos, which are used to choose the model. For each model, we save the version with the lowest validation loss (a measure of prediction error). We then choose the model with the most correct validation predictions and evaluate it on the separate test photos."
preprocess="We convert photos to color (RGB) and crop them to 224 x 224 pixels. During training, random crops and horizontal flips add variety. For validation and testing, we resize the shorter edge to 256 pixels and take a fixed center crop. Pixel values are adjusted using ImageNet's mean and standard deviation."
choice="We chose ResNet18 because it correctly identified 12 of 15 validation photos (80.0%); each custom CNN identified 4 (26.7%). We made this choice before checking the test results. ResNet18's earlier training on ImageNet gives it useful image features when our own training set is small."
tradeoff=f"ResNet18 is larger and slower than the two custom CNNs. It contains about 11.18 million model values, called parameters, but we train only 2,565 in its final layer. It takes about {chosen['inference_ms_per_image']:.1f} milliseconds to make a prediction on our CPU, excluding image preparation. We will use it as our starting model for later milestones because it performed best in this comparison."
limit="The test set is small: just three photos per person. One correct or incorrect prediction changes the overall score by 6.7 percentage points. We need more varied photos to judge how well the model handles different lighting, poses and backgrounds."
repro="The repository includes the photos, the list showing how they were divided, all three trained models, training logs and test predictions."
report=title("IE 7615 / GROUP 1 / MILESTONE 1","Classification results")
report += [p("<b>Team:</b> "+team+"."),p(main),section("Photos and preparation"),p("We use five CelebA identity IDs: 7007, 2970, 2336, 7 and 4428. The folders contain 121 photos in total. We use 23 per person (115 total), leaving six unused. These are the dataset's ID labels; we have not verified celebrity names."),table([["Photo set","Purpose","Total"],["Training","Teach the models","85"],["Validation","Choose the model","15"],["Test","Measure the final result","15"]],[100,340,76]),Spacer(1,6),p("Each person has 17 training, 3 validation and 3 test photos. The file list is saved so every model uses the same photos. We found no identical image files across these groups."),p(preprocess),section("The three models"),p(models_text),section("How we trained them"),p(setup),p("All models use Adam, a learning rate of 0.001 and batches of 16 images. The small CNN has two convolution layers (32 and 64 channels); the deeper CNN has four (32, 64, 128 and 256). Dropout rates are 0.25 and 0.35. The file shuffle uses seed 47 plus the identity number, so the same photos are selected each time.","SmallCustom"),PageBreak()]
report += title("MODEL COMPARISON","Results and training progress")
report += [table([["Model","Validation","Test","Parameters<br/>(total / trained)","Training"]]+[[label,f'{m["validation_accuracy"]:.1%}',f'{m["test_accuracy"]:.1%}<br/>({m["test_correct"]}/15)',f'{m["total_parameters"]:,}<br/>{m["trainable_parameters"]:,}',f'{m["training_seconds"]:.1f} s'] for label,m in zip(labels,metrics)],[80,72,72,188,104]),p("Parameters are the numbers stored inside a model. Training time includes validation and saving the best version. Measurements use an Intel Core i7-7700K CPU with two threads.","SmallCustom"),section("How prediction error changed during training"),p("Loss measures prediction error; lower is better. Blue shows training photos and orange shows validation photos. The dotted line marks the saved version. These curves come directly from the training logs."),Image(str(loss_path),width=420,height=378),p("The small CNN improves slowly. The deeper CNN learns the training photos better, but its validation error rises and falls. ResNet18 has the lowest validation error. The selected versions are from training passes 30, 21 and 28 for the small CNN, deeper CNN and ResNet18.","SmallCustom"),PageBreak()]
report += title("OUR CHOICE","Why we chose ResNet18")
report += [p(choice),p(tradeoff),Image(str(BASE/(names[2]+"_confusion.png")),width=270,height=225),p("This chart shows the test mistakes. Rows show the actual person; columns show the model's answer. Numbers on the diagonal are correct answers. For IDs 7007, 2970, 2336, 7 and 4428, the model gets 1, 2, 3, 3 and 1 of the three photos correct. It mistakes 4428 for 2970 twice.","SmallCustom"),table([["Prediction time","Small CNN","Deeper CNN","ResNet18"],["Milliseconds per photo"]+[f'{m["inference_ms_per_image"]:.1f}' for m in metrics]],[156,120,120,120]),p("We timed one photo at a time, after a warm-up, and report the median over five passes through the test set. Image loading and preparation are excluded.","SmallCustom"),section("What these results can tell us"),p(limit),section("Checking the results"),p(repro),p("The README explains how to open notebooks 01-05, check the saved models and train them again. These results are saved as group1_repro_v1. Aditi's earlier results are kept separately for reference.","SmallCustom")]
build("group1_training_results.pdf",report)
print("Built the training report. The team proposal was not changed.")
